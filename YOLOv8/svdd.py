# svdd.py
import torch
import torch.nn as nn

#
# class DeepSVDD(nn.Module):
#     def __init__(self, feat_dim):
#         super().__init__()
#         self.center = torch.zeros(feat_dim)
#
#     def init_center(self, features):
#         self.center = features.mean(0)
#
#     def forward(self, x):
#         return torch.sum((x - self.center) ** 2, dim=1)

import torch
import torch.nn as nn

class DeepSVDD(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.center = None

    def init_center(self, features):
        self.center = torch.mean(features, dim=0)

    def forward(self, x):
        # 假设x是特征
        dist = torch.sum((x - self.center) ** 2, dim=1)
        return dist