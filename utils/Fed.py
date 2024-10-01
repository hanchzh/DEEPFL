from pyexpat import model
from utils import cfed_options
from models.test import test_img
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_train_test
import torch
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
args = args_parser()


class Fed:
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device('cuda:{}'.format(
            self.args.gpu) if torch.cuda.is_available() and self.args.gpu != -1 else 'cpu')
        self.num_users = args.num_users

    def prepare(self):
        self.load_data_fl()
        self.build_model()

    def load_data_sfl_NOAA_NASA(self):
        # 加载数据集并划分给用户
        if self.args.dataset == 'NOAA_NASA':
            # 导入必要的库
            import xarray as xr
            import numpy as np
            from torch.utils.data import Dataset, DataLoader

            # 假设已经下载并解压了 NOAA 和 NASA 的数据集到指定路径
            # 加载 NOAA 数据集
            dataset_noaa = xr.open_dataset('./data/NOAA/noaa_data.nc')
            # 加载 NASA Ocean Color 数据集
            dataset_nasa = xr.open_dataset('./data/NASA/nasa_ocean_color_data.nc')

            # 对数据集进行预处理，例如选择特定的变量、时间范围、空间范围等
            # 假设我们感兴趣的变量是 'sea_surface_temperature' 和 'chlorophyll_concentration'
            variables = ['sea_surface_temperature', 'chlorophyll_concentration']
            dataset_noaa = dataset_noaa[variables]
            dataset_nasa = dataset_nasa[variables]

            # 合并两个数据集
            dataset_combined = xr.concat([dataset_noaa, dataset_nasa], dim='time')

            # 划分训练集和测试集
            train_time_range = slice('2000-01-01', '2015-12-31')
            test_time_range = slice('2016-01-01', '2020-12-31')
            dataset_train = dataset_combined.sel(time=train_time_range)
            dataset_test = dataset_combined.sel(time=test_time_range)

            # 数据标准化（以训练集的均值和标准差为基准）
            mean = dataset_train.mean(dim='time')
            std = dataset_train.std(dim='time')
            dataset_train = (dataset_train - mean) / std
            dataset_test = (dataset_test - mean) / std

            # 定义自定义 Dataset 类
            class SatelliteDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                    self.times = data['time'].values

                def __len__(self):
                    return len(self.times)

                def __getitem__(self, idx):
                    # 获取指定时间点的数据
                    time = self.times[idx]
                    x = self.data.sel(time=time)
                    # 将数据转换为 numpy 数组
                    x_array = x.to_array().values
                    # 假设我们使用 'sea_surface_temperature' 作为特征，'chlorophyll_concentration' 作为标签
                    features = x_array[0]  # sea_surface_temperature
                    label = x_array[1]  # chlorophyll_concentration
                    return features, label

            # 创建训练集和测试集的 Dataset 对象
            dataset_train = SatelliteDataset(dataset_train)
            dataset_test = SatelliteDataset(dataset_test)

            # 将数据划分给用户
            if self.args.iid:
                # 如果是 IID 的数据划分
                dict_users = satellite_iid(dataset_train, self.args.num_users)
            else:
                # 如果是 Non-IID 的数据划分
                dict_users, dict_users_test = satellite_noniid_train_test(
                    dataset_train, dataset_test, self.args.num_users)
        else:
            exit('Error: unrecognized dataset')

        # 获取数据的形状
        img_size = dataset_train[0][0].shape

        # 保存数据和划分结果
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users
        self.dict_users_test = dict_users_test
        self.img_size = img_size
        return dataset_train, dataset_test, dict_users, dict_users_test

    def satellite_iid(dataset, num_users):
        # 将数据集均匀地分配给每个用户
        num_items = int(len(dataset) / num_users)
        all_idxs = np.arange(len(dataset))
        np.random.shuffle(all_idxs)
        dict_users = {}
        for i in range(num_users):
            dict_users[i] = set(all_idxs[i * num_items:(i + 1) * num_items])
        return dict_users

    def satellite_noniid_train_test(dataset_train, dataset_test, num_users):
        # 按照标签的值对数据进行排序，以实现 Non-IID 划分
        train_labels = np.array([dataset_train[i][1] for i in range(len(dataset_train))])
        test_labels = np.array([dataset_test[i][1] for i in range(len(dataset_test))])

        train_idxs = np.argsort(train_labels)
        test_idxs = np.argsort(test_labels)

        dict_users = {}
        dict_users_test = {}

        num_items_train = int(len(train_idxs) / num_users)
        num_items_test = int(len(test_idxs) / num_users)

        for i in range(num_users):
            dict_users[i] = set(train_idxs[i * num_items_train:(i + 1) * num_items_train])
            dict_users_test[i] = set(test_idxs[i * num_items_test:(i + 1) * num_items_test])

        return dict_users, dict_users_test

    def load_data_sfl_mnist(self):
        # load dataset and split users
        if self.args.dataset == 'mnist':
            trans_mnist = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST(
                './data/mnist/', train=True, download=False, transform=trans_mnist)
            dataset_test = datasets.MNIST(
                './data/mnist/', train=False, download=False, transform=trans_mnist)
            # sample users
            if self.args.iid:
                # 如果是iid 的数据集,key值是用户id，value是用户拥有的图片
                dict_users = mnist_iid(dataset_train, self.args.num_users)
            else:
                # 如果是no-iid 的数据集
                dict_users, dict_users_test = mnist_noniid_train_test(dataset_train, dataset_test, self.args.num_users)
        elif self.args.dataset == 'cifar':
            # cifar 数据集
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10(
                './data/cifar', train=True, download=False, transform=trans_cifar)
            dataset_test = datasets.CIFAR10(
                './data/cifar', train=False, download=False, transform=trans_cifar)
            if self.args.iid:
                dict_users = cifar_iid(dataset_train, self.args.num_users)
            else:
                exit('Error: only consider IID setting in CIFAR10')
        else:
            exit('Error: unrecognized dataset')
        img_size = dataset_train[0][0].shape
        '''我们通过定义不同的数据划分方式将数据分为 iid 和 non-iid 两种，用来模拟测试 FedAvg 在不同场景下的性能。返回的是一个字典类型 dict_users，key值是用户 id，values是用户拥有的图片id。(具体实现方式可以自行研究代码)
        '''
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users
        self.dict_users_test = dict_users_test
        self.img_size = img_size
        return dataset_train, dataset_test, dict_users, dict_users_test
    def load_data_fl(self, set = 0):
        # load dataset and split users
        dataset_trains = []
        dataset_tests = []


        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307+set,), (0.3081+set,))])
        dataset_train = datasets.MNIST(
            './data/mnist/', train=True, download=False, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            './data/mnist/', train=False, download=False, transform=trans_mnist)
        # dataset_trains.append(dataset_test)
        # dataset_tests.append(dataset_train)
        # sample users

        # key值是用户id，value是用户拥有的图片
        dict_users = mnist_iid(dataset_train, self.args.num_users)

        img_size = dataset_train[0][0].shape

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users
        self.img_size = img_size
        return dataset_train, dataset_test, dict_users

    def build_model(self):
        # build model
        if self.args.model == 'cnn' and self.args.dataset == 'cifar':
            net_glob = CNNCifar(args=self.args).to(self.args.device)
        elif self.args.model == 'cnn' and self.args.dataset == 'mnist':
            net_glob = CNNMnist(args=self.args).to(self.args.device)
        elif self.args.model == 'mlp':
            len_in = 1
            for x in self.img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200,
                           dim_out=self.args.num_classes).to(self.args.device)
        else:
            exit('Error: unrecognized model')
        print(net_glob)
        net_glob.train()
        self.net_glob = net_glob
        return net_glob


    def plot(self, curve):
        # plot loss curve
        plt.figure()
        plt.plot(range(len(curve)), curve)
        plt.ylabel('train_loss')
        plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(self.args.dataset,
                                                               self.args.model, self.args.epochs, self.args.frac,
                                                               self.args.iid))

    def plot_acc(self, curve):
        # plot loss curve
        plt.figure()
        plt.plot(range(len(curve)), curve)
        plt.ylabel('acc_test')
        plt.savefig('./save/fed_acc_{}_{}_{}_C{}_iid{}.png'.format(self.args.dataset,
                                                                   self.args.model, self.args.epochs, self.args.frac,
                                                                   self.args.iid))
    def train(self, w_locals,start,end):
        loss_train = []
        for iter in range(args.epochs):
            loss_locals = []
            # 随机选取一部分clients进行aggregate
            for idx in range(args.num_users):
                # 每个迭代轮次本地更新
                self.Locals[idx].net.load_state_dict(w_locals[idx])

                w, loss = self.Locals[idx].train()
                # 复制参与本轮更新的users的所有权重 w_locals

                w_locals[idx] = (copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

            # ___________Weight update__________

            w_glob = cfed_options.sfedAvg(w_locals,start ,end)
            # 把权重更新到global_model

            w_locals = cfed_options.sfed(w_glob, w_locals,start ,end)

            # ___________print loss_____________
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
        print("Training finished")
        self.plot(loss_train)
        # loss_locals = pd.DataFrame(loss_train)
        # loss_locals.to_csv('myfile2.csv')
        # loss_locals = pd.read_csv('myfile2.csv')
        print(loss_locals)
        self.testing(self.Locals[1].net, self.dataset_trains[1], self.dataset_tests[1])
        return loss_train

    def testing(self, net_glob,dataset_train,dataset_test,):
        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        return acc_train, acc_test
    def dis_testing(self, net_glob):
        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, self.dict_users[0], args)
        acc_test, loss_test = test_img(net_glob, self.dict_users[0], args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        return acc_train, acc_test

