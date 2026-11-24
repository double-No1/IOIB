import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm

# ======================
# Config
# ======================
DATA_ROOT = "E:\Pycharm\OCC\data\images"      # data/zhan , data/zuo
BATCH_SIZE = 8
EPOCHS = 5                # SVDD 轻量训练
LR = 1e-4
RUNS = 5                  # 多次运行算均值方差
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMAL_LABEL = "zhan"
ANOMALY_LABEL = "zuo"

# ======================
# Dataset
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)

labels = np.array([y for _, y in dataset.samples])
class_to_idx = dataset.class_to_idx

zhan_idx = class_to_idx[NORMAL_LABEL]
zuo_idx = class_to_idx[ANOMALY_LABEL]

zhan_indices = np.where(labels == zhan_idx)[0]
zuo_indices = np.where(labels == zuo_idx)[0]

print(f"Normal (zhan): {len(zhan_indices)}")
print(f"Anomaly (zuo): {len(zuo_indices)}")

assert len(zhan_indices) > 0, "❌ No normal samples found!"

# ======================
# Model: ImageNet CNN
# ======================
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

# ======================
# Deep SVDD
# ======================
class DeepSVDD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def init_center(self, features):
        self.center.data = features.mean(dim=0)

# ======================
# Train + Test
# ======================
def run_once(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # backbone
    model = CNNBackbone().to(DEVICE)

    # optimizer（训练 CNN 使特征收缩）
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    svdd = DeepSVDD(model.out_dim).to(DEVICE)

    # -------- Train set (only normal)
    train_set = Subset(dataset, zhan_indices)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # -------- Init center
    feats = []
    model.eval()
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(DEVICE)
            f = model(x)
            feats.append(f)
    feats = torch.cat(feats)
    svdd.init_center(feats)

    # -------- Train SVDD loss
    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for x, _ in train_loader:
            x = x.to(DEVICE)
            f = model(x)
            loss = torch.mean((f - svdd.center) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {np.mean(losses):.4f}")

    # -------- Test
    test_indices = np.concatenate([zhan_indices, zuo_indices])
    y_true = np.array([0]*len(zhan_indices) + [1]*len(zuo_indices))

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    scores = []
    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(test_loader, desc="Testing"):
            x = x.to(DEVICE)
            f = model(x)
            s = torch.sum((f - svdd.center) ** 2, dim=1)
            scores.extend(s.cpu().numpy())

    scores = np.array(scores)
    auc = roc_auc_score(y_true, scores)

    threshold = np.percentile(scores[:len(zhan_indices)], 95)
    y_pred = (scores > threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )

    return auc, p, r, f1

# ======================
# Multiple Runs
# ======================
results = []

for run in range(RUNS):
    print(f"\n===== Run {run+1}/{RUNS} =====")
    metrics = run_once(seed=42+run)
    print(f"AUC={metrics[0]:.4f}, P={metrics[1]:.4f}, R={metrics[2]:.4f}, F1={metrics[3]:.4f}")
    results.append(metrics)

results = np.array(results)

print("\n====== Final Results ======")
print(f"AUC:       {results[:,0].mean():.4f} ± {results[:,0].std():.4f}")
print(f"Precision: {results[:,1].mean():.4f} ± {results[:,1].std():.4f}")
print(f"Recall:    {results[:,2].mean():.4f} ± {results[:,2].std():.4f}")
print(f"F1 Score:  {results[:,3].mean():.4f} ± {results[:,3].std():.4f}")
