#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from models.test import test_img
from models.Fed import FedAvg
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import MLPAIS, CNNSAR, mnist_iid, mnist_noniid, cifar_iid
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
     
    
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            './data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            './data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            #如果是iid 的数据集
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            #如果是no-iid 的数据集
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        #cifar 数据集
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            './data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            './data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
     elif args.dataset == 'cifar':
        # CIFAR 数据集
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            './data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            './data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')

    elif args.dataset == 'noaa_ais':
        # 加载 NOAA AIS 数据集
        dataset_train, dataset_test = load_noaa_ais_data()
        if args.iid:
            dict_users = split_data_iid(dataset_train, args.num_users)
        else:
            dict_users = split_data_noniid(dataset_train, args.num_users)

    elif args.dataset == 'nasa_sar':
        # 加载 NASA SAR 图像数据集
        dataset_train, dataset_test = load_nasa_sar_data()
        if args.iid:
            dict_users = split_data_iid(dataset_train, args.num_users)
        else:
            dict_users = split_data_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    '''我们通过定义不同的数据划分方式将数据分为 iid 和 non-iid 两种，用来模拟测试 FedAvg 在不同场景下的性能。返回的是一个字典类型 dict_users，key值是用户 id，values是用户拥有的图片id。(具体实现方式可以自行研究代码)
    '''
    # 获取输入数据的形状
    if args.dataset in ['mnist', 'cifar', 'nasa_sar']:
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'noaa_ais':
        # 对于 AIS 数据，输入是特征向量
        input_size = len(dataset_train[0][0])  # 假设每个样本是 (features, label)
    else:
        exit('Error: unrecognized dataset')

    # build model
if args.model == 'cnn':
    if args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'nasa_sar':
        net_glob = CNNSAR(args=args).to(args.device)
    else:
        exit('Error: CNN model is not applicable for this dataset')
elif args.model == 'mlp':
    if args.dataset in ['mnist', 'cifar', 'nasa_sar']:
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200,
                       dim_out=args.num_classes).to(args.device)
    elif args.dataset == 'noaa_ais':
        net_glob = MLPAIS(dim_in=input_size, dim_hidden=200,
                          dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: MLP model is not applicable for this dataset')
else:
    exit('Error: unrecognized model')

    print(net_glob)
    net_glob.train()

    # copy weights
    # 全局model权重
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # if args.all_clients:
    #     print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # 随机选取一部分clients进行aggregate
        for idx in idxs_users:
            # 每个迭代轮次本地更新
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # 复制参与本轮更新的users的所有权重 w_locals
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        # 通过定义的fedavg函数求参数的平均值
        # w_global
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        # 复制到总的net，每个用户进行更新
        #
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset,
                args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))





    

