import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    """
    Image → Graph-level embedding
    """
    def __init__(self, out_dim=32, pretrained=True):
        super().__init__()

        backbone = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # 去掉 fc
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.backbone(x)          # [B, 512, 1, 1]
        feat = feat.view(x.size(0), -1)  # [B, 512]
        emb = self.proj(feat)            # [B, D]
        return emb
