
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence


from losses import OCC_loss
from layers import CNN




class Feature2Dist(nn.Module):


    def __init__(self, feat_dim, temp=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.temp = temp  

    def forward(self, feat):


        dist = F.softmax(feat / self.temp, dim=-1) + 1e-8
        return dist


class PatchSampler(nn.Module):

    def __init__(self, patch_size=32, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_weights = nn.Parameter(torch.ones(self.num_patches) / self.num_patches)

    def forward(self, x):

        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_patches, C, self.patch_size, self.patch_size)

        sampler = torch.distributions.Categorical(probs=F.softmax(self.patch_weights, dim=0))
        patch_indices = sampler.sample((B,))

        sampled_patches = patches[torch.arange(B), patch_indices]
        return sampled_patches, patch_indices




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


        center_init = torch.randn(1, self.feat_dim, device=self.device)
        self.center = nn.Parameter(F.normalize(center_init, dim=1))

        self.feat2dist = Feature2Dist(feat_dim=self.feat_dim)

    def forward(self, imgs, is_normal=True):

        sampled_patches, _ = self.patch_sampler(imgs)


        feat_orig = self.encoder(imgs)
        feat_patch = self.encoder(sampled_patches)


        feat_orig = F.normalize(feat_orig, dim=1)
        feat_patch = F.normalize(feat_patch, dim=1)

        with torch.no_grad():
            self.center.copy_(F.normalize(self.center, dim=1))

        if is_normal:
            occ_loss = self.occ_loss_fn(feat_orig, self.center)
        else:
            occ_loss = torch.tensor(0.0, device=self.device)


        dist_patch = self.feat2dist(feat_patch)
        dist_center = self.feat2dist(self.center)



        prior = torch.ones_like(dist_patch) / self.feat_dim

        kl_compress_loss = kl_divergence(
            Categorical(probs=dist_patch),
            Categorical(probs=prior)
        ).mean()


        center_dist_expanded = dist_center.expand_as(dist_patch)
        ce_relevance_loss = self.cross_entropy_max_loss(dist_patch, center_dist_expanded)

        mean_feat = torch.mean(feat_patch, dim=0, keepdim=True)  # (1, feat_dim)

        variance_loss = -torch.mean(torch.sum((feat_patch - mean_feat) ** 2, dim=0))

        # 另一种简化方式（推荐，因为我们想鼓励分散）：
        # variance_loss = -torch.mean(torch.var(feat_orig, dim=0))
        # 我们使用前者，因为它更接近于惩罚特征协方差的迹。
        collapse_loss = variance_loss


        total_loss = (
                occ_loss +
                self.args.kl * kl_compress_loss +
                self.args.mi * ce_relevance_loss +
                self.args.collapse * collapse_loss
        )


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

        feat_orig = self.encoder(imgs)
        feat_orig = F.normalize(feat_orig, dim=1)


        center = self.center.expand_as(feat_orig)

        anomaly_score = F.mse_loss(feat_orig, center, reduction='none').sum(dim=1)
        return anomaly_score

    def cross_entropy_max_loss(self, patch_dist, center_dist):

        # H(P, Q) = - sum( P(x) * log(Q(x)) )
        ce_loss = -torch.sum(patch_dist * torch.log(center_dist), dim=1).mean()
        return ce_loss
