# test_svdd_single.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

from dataset import ImageFolderDataset
from svdd import DeepSVDD

# =========================
# Config
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_root = 'E:\Pycharm\OCC\data\images'          # 测试集目录
batch_size = 8
YOLO_WEIGHTS = './yolov8n.pt'
SVDD_MODEL_PATH = './svdd.pth'
NUM_RUNS = 5  # 重复次数，用于均值和标准差

# =========================
# 1. Load dataset (全部数据)
# =========================
test_set = ImageFolderDataset(data_root, is_train=False)
assert len(test_set) > 0, "❌ No test images found."

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

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
# 3. Determine feature dimension
# =========================
with torch.no_grad():
    x, _ = next(iter(test_loader))
    x = x.to(device)
    feat = feature_extractor(x)
    feat = torch.mean(feat, dim=[2,3])
    feat_dim = feat.shape[1]

print(f"Feature dimension: {feat_dim}")

# =========================
# 4. Load SVDD model
# =========================
svdd = DeepSVDD(feat_dim).to(device)
svdd.load_state_dict(torch.load(SVDD_MODEL_PATH, map_location=device))
svdd.center = svdd.center.to(device)
svdd.eval()

# =========================
# 5. Repeat testing NUM_RUNS times
# =========================
all_auc, all_precision, all_recall, all_f1 = [], [], [], []

for run in range(NUM_RUNS):
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x, labels in tqdm(test_loader, desc=f'Testing run {run+1}'):
            x = x.to(device)
            f = feature_extractor(x)
            f = torch.mean(f, dim=[2,3])
            scores = torch.sum((f - svdd.center) ** 2, dim=1)
            all_scores.append(scores.cpu())
            all_labels.append(labels)

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_scores)
    threshold = np.mean(all_scores)
    pred_labels = (all_scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, pred_labels, average='binary')

    print(f"Run {run+1} - AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    all_auc.append(auc)
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)

# =========================
# 6. Compute mean and std
# =========================
print("\n===== Test Results (mean ± std) =====")
print(f"AUC:       {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
print(f"Precision: {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
print(f"Recall:    {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
print(f"F1 Score:  {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
