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


class PSDataset(Dataset):
    def __init__(self, data_root='data/PepperSeedlings/train.txt', transform=None):
        super().__init__()
        data = pd.read_csv(data_root, delimiter=' ')   # xyzrgbl, N*7
        data = data.values
        points, labels = data[:, 0:6], data[:, 6] # xyzrgb, N*6; l, N
        xyz_min = np.amin(points, axis=0)[0:3]
        points[:, :3] -= xyz_min
        labels = labels.astype(np.int32)
        tmp, _ = np.histogram(labels, range(3))
        labelweights = tmp
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        self.transform = transform
        self.points = points
        self.labels = labels

        print('read ' + str(len(self.points)) + ' examples')
    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point = self.points[idx]
        label = self.labels[idx]
        if self.transform:
            point = self.transform(point)
        # print(point)
        point = np.expand_dims(point, 0)
        # print(point)
        return point, label
        


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