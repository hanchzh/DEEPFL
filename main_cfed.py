from utils import cfed_options
from models.Update import LocalUpdate
from utils.options import args_parser
import copy
from utils.Fed import Fed
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
    fed = Fed(args)
    dataset_trains = []
    dataset_tests = []
    for i in range(args.num_users):
        dataset_train, dataset_test, dict_users = fed.load_data_fl_NOAA_NASA(i*0.55)
        dataset_trains.append(dataset_train)
        dataset_tests.append(dataset_test)

    net_glob = fed.build_model()

    w_glob = net_glob.state_dict()
    Locals = []
    for idx in range(args.num_users):
        net_local = fed.build_model()
        # 每个迭代轮次本地更新
        local = LocalUpdate(
            args=args, dataset=dataset_trains[idx], idxs=dict_users[idx], net=net_local)
        Locals.append(local)
    fed.Locals = Locals
    fed.dataset_trains = dataset_trains
    fed.dataset_tests = dataset_tests
    fed_int = copy.deepcopy(fed)
    w_locals = [w_glob for i in range(args.num_users)]
    fed.train(copy.deepcopy(w_locals),0,4)
    fed = copy.deepcopy(fed_int)
    fed.train(copy.deepcopy(w_locals),0,3)
    fed = copy.deepcopy(fed_int)
    fed.train(copy.deepcopy(w_locals),0,2)
    fed = copy.deepcopy(fed_int)
    fed.train(copy.deepcopy(w_locals), 0, 1)
    fed = copy.deepcopy(fed_int)
    fed.train(copy.deepcopy(w_locals),0,0)



