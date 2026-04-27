import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

import random
class CNN(nn.Module):
    
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1)  
        )

    def forward(self, x):
        feat = self.conv(x).flatten(1)  # (B, output_dim)
        return feat

