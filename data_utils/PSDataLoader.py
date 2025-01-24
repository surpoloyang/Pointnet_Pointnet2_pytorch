'''
@author: Chen Yang
@file: PSDataLoader.py
@time: 2024-12-11 15:39:21
'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import h5py

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    feature = f['rgb'][:]
    label = f['pid'][:]
    seg = f['seglabel'][:]
    return (data, feature, label, seg)


class PeedlingsDataset(Dataset):
    def __init__(self, root='./data', split='train', transform=None):
        if split == 'train':
            
            self.xyz, self.feature, self.ins, self.sem = load_h5_data_label_seg(os.path.join(root, 'train_aug.h5'))
            # f = h5py.File(os.path.join(root, 'train.h5'), 'r')
        else:    
            self.xyz, self.feature, self.ins, self.sem = load_h5_data_label_seg(os.path.join(root, 'test_aug.h5'))
            # f = h5py.File(os.path.join(root, 'test.h5'), 'r')
    def __getitem__(self, index):
        xyz1 = self.xyz[index]
        feature1 = self.feature[index]
        data1 = np.concatenate((xyz1, feature1), axis=-1)
        # label1 = self.label[index]
        sem1 = self.sem[index]
        # obj1 = self.objlabel[index]
        # sample = {'data': data1, 'feature':feature1,'label': label1, 'seg': seg1}
        return data1, sem1

    def __len__(self):
        return len(self.sem)

        


if __name__ == '__main__':
    import torch, time, random
    # data_root = r'D:\3Dpointclouds\Pointnet_Pointnet2_pytorch\data\PepperSeedlings\semseg_test.txt'
    # num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01
    data_root = r'data/PepperSeedlings/semseg_test.txt'
    point_data = PSDataset(data_root=data_root, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    print('point label 0:', point_data.__getitem__(0)[1])

    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    # for idx in range(4):
    #     end = time.time()
    #     for i, (input, target) in enumerate(train_loader):
    #         print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
    #         end = time.time()