#导入库
import copy
import torch
import argparse
# import os.path as osp
# print(osp.dirname(osp.abspath(__file__)))
# import sys
# print(sys.path)
# sys.path.append('/home/zqm/RT4GOOD/data/data_process')
# print(sys.path)

# 将项目根目录添加到 sys.path 中
from torch_geometric.data import DataLoader

import numpy as np
import os.path as osp
from torch.autograd import grad
from datetime import datetime
import os
from data.data_process.spmotif_dataset import SPMotif
from data.dataloader.dataloader import dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for TR4GOOD')
    # parser.add_argument('--root_dir', type=str, default=os.path.abspath(os.path), help="The root directory of the project")
    parser.add_argument('--device', default=0, type=int, help='cuda device')
    parser.add_argument('--datadir', default='/home/zqm/RT4GOOD/data', type=str, help='directory for datasets.')
    parser.add_argument('--epoch', default=2, type=int, help='training iterations')
    parser.add_argument('--reg', default=True, type=bool)
    parser.add_argument('--seed',  nargs='?', default='[1,2,3]', help='random seed')
    parser.add_argument('--channels', default=32, type=int, help='width of network')
    parser.add_argument('--exp_name', default='', type=str, help='experiment name')
    parser.add_argument('--bias', default='0.333', type=str, help='select bias extend')#根据不同数据要调整
    # hyper 
    parser.add_argument('--pretrain', default=10, type=int, help='pretrain epoch')
    # parser.add_argument('--alpha', default=1e-2, type=float, help='invariant loss')
    # parser.add_argument('--r', default=0.25, type=float, help='causal_ratio')
    # basic
    parser.add_argument( "--output_path", type=str, default="outputs", help="Path to save outputs")
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--net_lr', default=1e-3, type=float, help='learning rate for the predictor')
    '''VQgraph的参数
    python train_teacher.py --exp_setting tran --teacher GCN 
    --dataset citeseer --output_path outputs --seed 0 
    --max_epoch 100 --patience 50 --device 0
    '''
    #python train_teacher.py --exp_setting tran --teacher GCN --dataset citeseer --output_path outputs --seed 0 --max_epoch 100 --patience 50 --device 0

    args = parser.parse_args()
    args.seed = eval(args.seed)
   
    train_loader, val_loader, test_loader, n_train_data, n_val_data, n_test_data = dataloader(args)
    # 输出 train_loader 中的第一个批次数据
    for data in train_loader:
        print(f"Data: {data}")
        print(f"Nodes: {data.x}")  # 假设数据中有节点特征 x
        print(f"Edges: {data.edge_index}")  # 假设数据中有边连接信息 edge_index
        print(f"Labels: {data.y}")  # 假设数据中有标签 y
        break  # 只打印第一个批次

    # # 输出数据集大小
    # n_train_data, n_val_data = len(train_dataset), len(val_dataset)
    # n_test_data = float(len(test_dataset))
    
    # 输出 train_loader 的基本统计信息
    print(f"Total number of batches: {len(train_loader)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of training data points: {n_train_data}")
    print(f"Number of validation data points: {n_val_data}")
    print(f"Number of test data points: {n_test_data}")
    print(f"Number of training data points: {n_train_data}")
    # dataset
    # print(osp.join(args.datadir, f'SPMotif-{args.bias}/'))
    # train_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='train')
    # val_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='val')
    # test_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='test')
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # n_train_data, n_val_data = len(train_dataset), len(val_dataset)
    # n_test_data = float(len(test_dataset))


    # log
    # datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    # all_info = { 'causal_acc':[], 'conf_acc':[], 'train_acc':[], 'val_acc':[], 'test_prec':[], 'train_prec':[], 'test_mrr':[], 'train_mrr':[]}
    # experiment_name = f'spmotif-{args.bias}.{args.reg}.{args.exp_name}.netlr_{args.net_lr}.batch_{args.batch_size}'\
    #                   f'.channels_{args.channels}.pretrain_{args.pretrain}.seed_{args.seed}.{datetime_now}'
    # exp_dir = osp.join('local/', experiment_name)
    # os.makedirs(exp_dir, exist_ok=True)
    # logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
    # args_print(args, logger)



