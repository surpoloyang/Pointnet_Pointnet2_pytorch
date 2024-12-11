'''
@author: Chen Yang
@file: PSDataLoader.py
@time: 2024-12-11 15:39:21
'''

import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class PSDataset(Dataset):
    def __init__(self, data_root='data/PepperSeedlings/train.txt', transform=None):
        super().__init__()
        data = np.load(data_root)   # xyzrgbl, N*7
        points, labels = data[:, 0:6], data[:, 6] # xyzrgb, N*6; l, N
        tmp, _ = np.histogram(labels, range(3))
        labelweights = tmp
        
        self.num_point = num_point
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
        return point, label``
        


if __name__ == '__main__':
    import torch, time, random
    data_root = '/data/PepperSeedlings/train.txt'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = PSDataset(data_root=data_root, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)

    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()