# models/ImprovedModels.py
import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedCNN(nn.Module):
    """使用预训练 ResNet50 作为特征提取器"""
    def __init__(self, emb_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层
        self.fc = nn.Linear(2048, emb_dim)

    def forward(self, x):
        x = self.features(x)          # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 2048)
        x = self.fc(x)                 # (B, emb_dim)
        return x


class OCGTL_Image(nn.Module):
    """简化版一类分类头：学习一个中心 c，输出嵌入与中心的距离"""
    def __init__(self, dim_features, config):
        super().__init__()
        self.device = config['device']
        # 初始化中心 c (可学习)
        self.c = nn.Parameter(torch.randn(1, dim_features).to(self.device))

    def forward(self, z):
        # z: (B, dim)
        # 返回 z 和中心 c（扩展至 batch 大小）
        c_batch = self.c.expand(z.size(0), -1)
        return z, c_batch