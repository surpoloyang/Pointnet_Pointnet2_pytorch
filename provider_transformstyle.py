import numpy as np
import torch

# ================================================================
# torchvision.transforms 风格的点云数据增强类
# 每个类都接收一个包含 'points' 和 'labels' 的字典
# 并返回一个经过变换的字典
# ================================================================

class Normalize:
    """
    对单个点云数据进行归一化，使其以原点为中心，并缩放到单位球内。
    只对 'points' 操作。
    """
    def __call__(self, sample):
        points, labels = sample['points'], sample['labels']
        
        assert len(points.shape) == 2
        centroid = np.mean(points, axis=0)
        pc = points - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        # 避免除以零
        if m < 1e-6:
            m = 1
        pc = pc / m
        
        sample['points'] = pc
        return sample

class RandomRotate_Z:
    """
    随机绕Z轴旋转点云。
    只对 'points' 操作。
    """
    def __call__(self, sample):
        points, labels = sample['points'], sample['labels']
        
        assert len(points.shape) == 2 and points.shape[1] == 3
        
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        
        sample['points'] = np.dot(points, rotation_matrix).astype(np.float32)
        return sample

class RandomRotate_Y:
    """
    随机绕Y轴旋转点云。
    只对 'points' 操作。
    """
    def __call__(self, sample):
        points, labels = sample['points'], sample['labels']
        
        assert len(points.shape) == 2 and points.shape[1] == 3
        
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        
        sample['points'] = np.dot(points, rotation_matrix).astype(np.float32)
        return sample

class RandomScale:
    """
    随机缩放点云。
    只对 'points' 操作。
    """
    def __init__(self, scale_low=0.8, scale_high=1.25):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, sample):
        points, labels = sample['points'], sample['labels']
        
        assert len(points.shape) == 2
        scale = np.random.uniform(self.scale_low, self.scale_high)
        points = points * scale
        
        sample['points'] = points.astype(np.float32)
        return sample

class Jitter:
    """
    对点云中的每个点应用随机抖动。
    只对 'points' 操作。
    """
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, sample):
        points, labels = sample['points'], sample['labels']

        assert len(points.shape) == 2
        N, C = points.shape
        jitter = np.clip(self.sigma * np.random.randn(N, C), -self.clip, self.clip)
        
        sample['points'] = (points + jitter).astype(np.float32)
        return sample

class RotatePerturbation:
    """
    对点云进行小幅度的随机旋转扰动。
    结合三个轴向的小角度旋转。
    只对 'points' 操作。
    """
    def __init__(self, angle_sigma=0.1, angle_clip=0.3):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, sample):
        points, _ = sample['points'], sample['labels']
        
        assert len(points.shape) == 2 and points.shape[1] == 3
        
        # 生成三个轴向的小角度扰动
        angles = np.clip(self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip)
        
        # X轴旋转矩阵
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        # Y轴旋转矩阵
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        # Z轴旋转矩阵
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        
        # 组合旋转矩阵
        R = np.dot(Rz, np.dot(Ry, Rx))
        
        sample['points'] = np.dot(points, R).astype(np.float32)
        return sample

class ToTensor:
    """
    将Numpy数组转换为PyTorch Tensor。
    同时对 'points' 和 'labels' 操作。
    """
    def __call__(self, sample):
        points, labels = sample['points'], sample['labels']
        
        sample['points'] = torch.from_numpy(points).float()
        sample['labels'] = torch.from_numpy(labels).long()
        return sample
