import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# 因为原文中特征提取网络中的MLP是shared共享参数的，所以使用一维卷积而不是Linear

# 第一个joint alignment/transformation network, transformation network is a mini-PointNet that takes raw point cloud as input and regresses to a 3 × 3 matrix
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        # Batchnorm is used for all layers with ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        # global feature
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        # output score
        x = self.fc3(x)
        # The output matrix is initialized as an identity matrix
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        # The output adds the identity matrix, which is equivalent to initializing the output as an parameter matrix
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        # 第一个变形层
        self.stn = STN3d(channel)
        # 第一个MLP
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        # 以下两个conv是第二个MLP
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # Batchnorm is used for all layers with ReLU
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        # 第二个变形层
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x) # B x 3 x 3
        # 第一次转换
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]   # B x N1 x D - 3
            x = x[:, :, :3]        # B x N1 x 3
        x = torch.bmm(x, trans) # B x N1 x 3
        if D > 3:
            x = torch.cat([x, feature], dim=2)  # B x N1 x D
        x = x.transpose(2, 1)   # B x (3 or D) x N1
        # 第一个MLP
        x = F.relu(self.bn1(self.conv1(x)))
        # 第二次转换
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        # 局部特征
        pointfeat = x
        # 第二个MLP
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        # 全局特征
        x = x.view(-1, 1024)
        if self.global_feat:    # 只输出全局特征用于分类
            return x, trans, trans_feat # 全局特征:B × 1024便于后续Linear层，第一个转换矩阵，第二个转换矩阵
        else:   # 输出局部特征cat全局特征用于分割
            x = x.view(-1, 1024, 1).repeat(1, 1, N) # B x 1024 x 3
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # B x 1088 x 3, 第一个转换矩阵，第二个转换矩阵

# A regularization loss (with weight 0.001) is added to the softmax classification loss to make the matrix close to orthogonal.
def feature_transform_reguliarzer(trans): # trans:B × k × k
    
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]    # 1 × k × k
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss # (1,)