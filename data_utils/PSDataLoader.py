'''
@author: Chen Yang
@file: PSDataLoader.py
@time: 2024-12-11 15:39:21
'''

import os
import numpy as np
from torch.utils.data import Dataset
import h5py

def load_h5_data_label_xyz_semseg(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['seglabel'][:]
    f.close()
    return (data, label)


class PeedlingsDatasetXYZ(Dataset):
    def __init__(self, root='./data/Sy2', split='train', transform=None):
        if split == 'train':
            h5_path = os.path.join(root, 'train_aug.h5')
        else:    
            h5_path = os.path.join(root, 'test_aug.h5')
        
        self.xyz, self.sem = load_h5_data_label_xyz_semseg(h5_path)
        self.transform = transform # 保存 transform

    def __getitem__(self, index):
        # 加载原始数据
        points = self.xyz[index]
        labels = self.sem[index]

        # 创建一个 sample 字典
        sample = {'points': points, 'labels': labels}

        # 应用变换
        if self.transform:
            sample = self.transform(sample)
        
        return sample['points'], sample['labels']

    def __len__(self):
        return len(self.sem)


if __name__ == '__main__':
    import torch
    import time
    import random
    from torchvision import transforms
    from ..provider_transformstyle import Normalize, RandomRotate, Jitter, ToTensor # 假设 provider.py 放在同级目录下或可访问的路径

    # --- 演示如何使用 ---
    # 1. 定义变换
    train_transform = transforms.Compose([
        Normalize(),
        # ShufflePoints(), # -- 已移除 --
        RandomRotate(),
        Jitter(),
        ToTensor()
    ])

    # 2. 创建数据集
    data_root = '../data/546' # 请替换为您的数据路径
    point_data = PeedlingsDatasetXYZ(root=data_root, split='train', transform=train_transform)
    
    # 3. 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=4, shuffle=True)

    # 4. 检查输出
    print('point data size:', point_data.__len__())
    points, labels = next(iter(train_loader))
    print('Batch points shape:', points.shape)
    print('Batch labels shape:', labels.shape)
    print('Data type after transform:', points.dtype)
