
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence


from losses import OCC_loss
from layers import CNN



# -------------------------- 辅助模块：Feature2Dist (保留) --------------------------

class Feature2Dist(nn.Module):
    """将特征向量转换为概率分布（通过Softmax）。"""

    def __init__(self, feat_dim, temp=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.temp = temp  # 温度系数，控制分布平滑度

    def forward(self, feat):
        """
        feat: 特征向量，shape=(B, feat_dim) 或 (1, feat_dim)
        返回：概率分布
        """
        # 使用 Softmax 将特征转化为分布
        # + 1e-8 防止 log(0)
        dist = F.softmax(feat / self.temp, dim=-1) + 1e-8
        return dist


# -------------------------- 辅助模块：PatchSampler (保留) --------------------------

class PatchSampler(nn.Module):
    """可学习权重的子图（Patch）采样器"""
    def __init__(self, patch_size=32, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_weights = nn.Parameter(torch.ones(self.num_patches) / self.num_patches)

    def forward(self, x):
        """
        x: 输入图像，shape=(B, C, H, W)
        返回：采样的子图 (B, C, patch_size, patch_size), 采样索引 (B,)
        """
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_patches, C, self.patch_size, self.patch_size)

        sampler = torch.distributions.Categorical(probs=F.softmax(self.patch_weights, dim=0))
        patch_indices = sampler.sample((B,))

        sampled_patches = patches[torch.arange(B), patch_indices]
        return sampled_patches, patch_indices


# -------------------------- Model 主体 (修改 L2 归一化) --------------------------

class Model(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args
        self.device = args.device
        self.input_dim = input_dim
        self.feat_dim = args.feat_dim

        self.patch_sampler = PatchSampler(
            patch_size=args.patch_size,
            img_size=args.img_size
        )
        self.encoder = CNN(
            input_dim=input_dim,
            output_dim=self.feat_dim
        )
        self.occ_loss_fn = OCC_loss()

        # ⚠️ 修正 1: OCC 中心初始化为单位范数
        center_init = torch.randn(1, self.feat_dim, device=self.device)
        self.center = nn.Parameter(F.normalize(center_init, dim=1))

        self.feat2dist = Feature2Dist(feat_dim=self.feat_dim)

    def forward(self, imgs, is_normal=True):
        """
        imgs: 输入图像 (B, C, H, W)
        """
        # 1. 采样子图
        sampled_patches, _ = self.patch_sampler(imgs)

        # 2. 编码原图和子图特征
        feat_orig = self.encoder(imgs)
        feat_patch = self.encoder(sampled_patches)

        # 修正 2: 对特征进行 L2 归一化，防止中心坍塌
        feat_orig = F.normalize(feat_orig, dim=1)
        feat_patch = F.normalize(feat_patch, dim=1)

        # 确保中心点在每次前向传播后也保持单位范数 (如果中心是可学习参数)
        with torch.no_grad():
            self.center.copy_(F.normalize(self.center, dim=1))

        # 3. OCC 中心对齐损失
        if is_normal:
            occ_loss = self.occ_loss_fn(feat_orig, self.center)
        else:
            occ_loss = torch.tensor(0.0, device=self.device)

        # 4. 特征转概率分布
        dist_patch = self.feat2dist(feat_patch)
        dist_center = self.feat2dist(self.center)

        # 5. 信息瓶颈损失： IB = 压缩项 + 相关项

        # 5.1 IB 压缩项：最小化 KL(P_patch || P_prior)
        prior = torch.ones_like(dist_patch) / self.feat_dim

        kl_compress_loss = kl_divergence(
            Categorical(probs=dist_patch),
            Categorical(probs=prior)
        ).mean()

        # 5.2 IB 相关项：最大化 I(Patch; Center) -> 最小化 Cross-Entropy(P_patch, P_center)
        center_dist_expanded = dist_center.expand_as(dist_patch)
        ce_relevance_loss = self.cross_entropy_max_loss(dist_patch, center_dist_expanded)

        # 5.3 坍塌惩罚项：最大化特征方差
        # 仅对正常样本的原图特征计算
        # 计算特征的平均值 E[Z]
        mean_feat = torch.mean(feat_patch, dim=0, keepdim=True)  # (1, feat_dim)
        # 计算特征方差 V[Z] = E[(Z - E[Z])^2]
        variance_loss = -torch.mean(torch.sum((feat_patch - mean_feat) ** 2, dim=0))

        # 另一种简化方式（推荐，因为我们想鼓励分散）：
        # variance_loss = -torch.mean(torch.var(feat_orig, dim=0))
        # 我们使用前者，因为它更接近于惩罚特征协方差的迹。
        collapse_loss = variance_loss

        # 6. 总损失
        total_loss = (
                occ_loss +
                self.args.kl * kl_compress_loss +
                self.args.mi * ce_relevance_loss +
                self.args.collapse * collapse_loss
        )

        # 修正 3: 确保 total_loss 是标量用于 backward()
        total_loss = total_loss.mean()

        loss_dict = {
            "total_loss": total_loss,
            "occ_loss": occ_loss.mean() if occ_loss.dim() > 0 else occ_loss,
            "kl_compress_loss": kl_compress_loss,
            "ce_relevance_loss": ce_relevance_loss
        }
        return loss_dict

    @torch.no_grad()
    def predict_anomaly(self, imgs):
        """
        推理：计算异常分数（异常样本分数更高）
        """
        # 编码并归一化特征
        feat_orig = self.encoder(imgs)
        feat_orig = F.normalize(feat_orig, dim=1)

        # 确保中心点是单位范数 (在 __init__ 和 forward 中已处理)
        center = self.center.expand_as(feat_orig)

        # 异常分数：原图特征到OCC中心的距离 (在单位球上)
        anomaly_score = F.mse_loss(feat_orig, center, reduction='none').sum(dim=1)
        return anomaly_score

    def cross_entropy_max_loss(self, patch_dist, center_dist):
        """
        计算交叉熵 H(P_patch, P_center)，最小化此损失以最大化互信息 I(Patch; Center)。
        """
        # H(P, Q) = - sum( P(x) * log(Q(x)) )
        ce_loss = -torch.sum(patch_dist * torch.log(center_dist), dim=1).mean()
        return ce_loss