# test_svdd_kfold.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

from dataset import ImageFolderDataset
from svdd import DeepSVDD

# =========================
# Config
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_root = 'E:\Pycharm\OCC\data\images'          # 数据集目录
batch_size = 8
YOLO_WEIGHTS = './yolov8n.pt'
NUM_FOLDS = 5

# =========================
# 1. Load dataset
# =========================
full_dataset = ImageFolderDataset(data_root, is_train=True)

# 分离正常和异常索引
zhan_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 1]
zuo_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]

print(f"Total normal (zhan) samples: {len(zhan_indices)}")
print(f"Total anomaly (zuo) samples: {len(zuo_indices)}")

# =========================
# Check number of zhan samples
# =========================
if len(zhan_indices) < 2:
    print(f"⚠ Too few normal samples ({len(zhan_indices)}). Cannot perform KFold. Exiting.")
    exit(1)

# 自动调整折数
if len(zhan_indices) < NUM_FOLDS:
    NUM_FOLDS = len(zhan_indices)
    print(f"⚠ Warning: zhan samples < requested folds. Auto-adjust NUM_FOLDS={NUM_FOLDS}")

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# =========================
# 2. Load YOLOv8 backbone + neck
# =========================
yolo_wrapper = YOLO(YOLO_WEIGHTS)
yolo = yolo_wrapper.model.to(device)
yolo.eval()
for p in yolo.parameters():
    p.requires_grad = False

backbone_neck = yolo.model[:10]
feature_extractor = nn.Sequential(*backbone_neck).to(device)
feature_extractor.eval()

# =========================
# 3. KFold cross-validation
# =========================
fold_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(zhan_indices), 1):
    print(f"\n===== Fold {fold_idx} =====")

    if len(train_idx) == 0 or len(val_idx) == 0:
        print(f"⚠ Skip fold {fold_idx} due to insufficient zhan samples")
        continue

    # 五折训练集：仅用 zhan 的 train_idx
    train_subset = Subset(full_dataset, [zhan_indices[i] for i in train_idx])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 五折测试集：剩余 zhan + 全部 zuo
    val_subset_indices = [zhan_indices[i] for i in val_idx] + zuo_indices
    val_subset = Subset(full_dataset, val_subset_indices)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Fold {fold_idx} - train normal samples: {len(train_subset)}")
    print(f"Fold {fold_idx} - test samples: {len(val_subset)}")

    # =========================
    # Compute feature dimension
    # =========================
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x.to(device)
        feat = feature_extractor(x)
        feat = torch.mean(feat, dim=[2, 3])
        feat_dim = feat.shape[1]

    # =========================
    # Initialize SVDD and compute center
    # =========================
    svdd = DeepSVDD(feat_dim).to(device)
    features = []
    with torch.no_grad():
        for x, _ in tqdm(train_loader, desc='Init center'):
            x = x.to(device)
            f = feature_extractor(x)
            f = torch.mean(f, dim=[2,3])
            features.append(f)
    features = torch.cat(features, dim=0)
    svdd.init_center(features)
    svdd.center = svdd.center.to(device)
    svdd.eval()

    # =========================
    # Evaluate on validation set
    # =========================
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for x, labels in tqdm(val_loader, desc='Testing'):
            x = x.to(device)
            f = feature_extractor(x)
            f = torch.mean(f, dim=[2,3])
            scores = torch.sum((f - svdd.center) ** 2, dim=1)
            all_scores.append(scores.cpu())
            all_labels.append(labels)

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # =========================
    # Metrics
    # =========================
    auc = roc_auc_score(all_labels, all_scores)
    threshold = np.mean(all_scores)
    pred_labels = (all_scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, pred_labels, average='binary')

    print(f"Fold {fold_idx} - AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    fold_metrics.append([auc, precision, recall, f1])

# =========================
# 4. Compute mean and std
# =========================
if len(fold_metrics) == 0:
    print("⚠ No folds were run. Check normal sample count.")
else:
    fold_metrics = np.array(fold_metrics)
    mean_metrics = np.mean(fold_metrics, axis=0)
    std_metrics = np.std(fold_metrics, axis=0)

    print("\n===== Cross Validation Results =====")
    print(f"AUC:       {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"Precision: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"Recall:    {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"F1 Score:  {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
