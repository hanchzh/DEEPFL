from utils import cfed_options
from models.Update import LocalUpdate
from utils.options import args_parser
import copy
import numpy as np
import pandas as pd

if __name__ == "__main__":
    args = args_parser()
    loss_train = []
    acc_test = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    # ________model_preparing________
    # dict_users, 每个用户所持有的数据集，这里实际上是做了一个数据划分的list
    fed = cfed_options.Fed(args)
    dataset_trains = []
    dataset_tests = []
    for i in range(args.num_users):
        dataset_train, dataset_test, dict_users = fed.load_data_fl(i*0)
        dataset_trains.append(dataset_train)
        dataset_tests.append(dataset_test)
    net_glob = fed.build_model()
    net_local = net_glob
    w_glob = net_glob.state_dict()
    # __________training____________
    m = max(int(args.frac * args.num_users), 1)
    # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        # 随机选取一部分clients进行aggregate
        for idx in range(args.num_users):
            net_glob.load_state_dict(w_locals[idx])
            # 每个迭代轮次本地更新
            local = LocalUpdate(
                args=args, dataset=dataset_trains[idx], idxs=dict_users[idx])

            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # 复制参与本轮更新的users的所有权重 w_locals

            w_locals[i] = (copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # ___________Weight update__________

        w_glob = cfed_options.sfedAvg(w_locals)
        # 把权重更新到global_model
        for i in range(args.num_users):
            w_locals = cfed_options.sfed(w_glob, w_locals)


        # ___________print loss_____________
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)


    print("Training finished")
    fed.plot(loss_train)
    loss_locals = pd.DataFrame(loss_train)
    loss_locals.to_csv('myfile2.csv')
    loss_locals = pd.read_csv('myfile2.csv')
    print(loss_locals)
    fed.testing(net_glob, dataset_trains[1], dataset_tests[1])