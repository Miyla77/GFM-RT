from data.data_process.spmotif_dataset import SPMotif
from torch_geometric.data import DataLoader
import os.path as osp
# print(osp)
def dataloader(args):
    train_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='train')
    val_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='val')
    test_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    n_train_data, n_val_data = len(train_dataset), len(val_dataset)
    n_test_data = float(len(test_dataset))
    return train_loader, val_loader, test_loader, n_train_data, n_val_data, n_test_data

