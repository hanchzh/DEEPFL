from pyexpat import model
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




def dispatch(w, client_a, client_b):
    # 将build的模型权重复制到全局模型
    # copy weight to net_glob
    # 复制到总的net，每个用户进行更新
    w_overlap = copy.deepcopy(w[0])
    for k in w_overlap.keys():
        w_overlap[k] = (len(client_a) * w[0][k] + len(client_b) * (w[1][k]))
        # 写法正确，接下来要考虑如何判断位于overlap区域的问题
        # i是list的编号，K是layer的层数
        # 每一层都要做div，加一起以后除以总数就行
        w_overlap[k] = torch.div(w_overlap[k], (len(client_a) + len(client_b)))
    return w_overlap


def fedAvg(w):
    w_avg = copy.deepcopy(w[0])
    key = w_avg.keys()
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
def sfed(w, w_users,start = 0,end = 0):
    for w_user in w_users:
        for k in list(w_user.keys())[start:end]:
            w_user[k] = w[k]
    return w_users
def sfedAvg(w,start = 0,end = 0):
    w_avg = copy.deepcopy(w[0])
    for k in list(w_avg.keys())[start:end]:
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def fed_weight(w):
    # 如果位于overlap，权重参数会大一些,这里后面再做修改
    alfa_u = 1
    alfa_v = 1.5
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def test():
    fed_a = Fed(args)
    fed_b = Fed(args)
    dataset_train_a, dict_users_a = fed_a.load_data_fl()
    dataset_train_b, dict_users_b = fed_b.load_data_fl()
    net_glob_a = fed_a.build_model()
    net_glob_b = fed_b.build_model()
    net_local = net_glob_a
    w_glob_a = net_glob_a.state_dict()
    w_glob_b = net_glob_b.state_dict()

    for idx in range(args.num_users):
        local_a = LocalUpdate(args=args, dataset=dataset_train_a, idxs=dict_users_a[idx])
        local_b = LocalUpdate(args=args, dataset=dataset_train_b, idxs=dict_users_b[idx])
        w_a, loss_a = local_a.train(net=copy.deepcopy(net_glob_a).to(args.device))
        w_b, loss_b = local_b.train(net=copy.deepcopy(net_glob_b).to(args.device))

    once_w_a = [w_a, w_b]
    test_avg = fedAvg(once_w_a)
    non_overlap_clients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_overlap = dispatch(w_a, w_b, non_overlap_clients, non_overlap_clients)

    print("That is avg alg {}", test_avg)
    print("That is dispatch {}", test_overlap)


if __name__ == "__main__":
    fed_a = Fed(args)
    curve = [1,13,1.4,5.6,7.1]
    plt.ylabel('{%s}', list(dict(curve=curve).keys()[0]))
