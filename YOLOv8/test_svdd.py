# test_svdd.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from ultralytics import YOLO
from tqdm import tqdm

from dataset import ImageFolderDataset
from svdd import DeepSVDD

# =========================
# Config
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_root = 'E:\Pycharm\OCC\data\images'           # 测试集目录
batch_size = 8
MODEL_PATH = './svdd.pth'
YOLO_WEIGHTS = './yolov8n.pt'


# =========================
# 1. Dataset (全部数据: zhan + zuo)
# =========================
test_set = ImageFolderDataset(data_root, is_train=False)
assert len(test_set) > 0, "❌ No test images found in data/"

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0   # Windows 必须
)


# =========================
# 2. Load YOLOv8 (frozen)
# =========================
yolo_wrapper = YOLO(YOLO_WEIGHTS)
yolo = yolo_wrapper.model.to(device)   # DetectionModel
yolo.eval()
for p in yolo.parameters():
    p.requires_grad = False

# backbone + neck
backbone_neck = yolo.model[:10]
feature_extractor = nn.Sequential(*backbone_neck).to(device)
feature_extractor.eval()


# =========================
# 3. Estimate feature dimension
# =========================
with torch.no_grad():
    x, _ = next(iter(test_loader))
    x = x.to(device)
    feat = feature_extractor(x)
    feat = torch.mean(feat, dim=[2, 3])
    feat_dim = feat.shape[1]

print(f'📐 Feature dimension: {feat_dim}')


# =========================
# 4. Load SVDD center
# =========================
svdd = DeepSVDD(feat_dim).to(device)
svdd.load_state_dict(torch.load(MODEL_PATH, map_location=device))
svdd.center = svdd.center.to(device)   # ⭐ 修复 CPU/GPU 不一致
svdd.eval()


# =========================
# 5. Compute anomaly scores
# =========================
all_scores = []
all_labels = []

with torch.no_grad():
    for x, labels in tqdm(test_loader, desc='Testing'):
        x = x.to(device)
        f = feature_extractor(x)
        f = torch.mean(f, dim=[2, 3])
        scores = torch.sum((f - svdd.center) ** 2, dim=1)  # L2 distance
        all_scores.append(scores.cpu())
        all_labels.append(labels)

all_scores = torch.cat(all_scores).numpy()
all_labels = torch.cat(all_labels).numpy()

# =========================
# 6. Metrics
# =========================
auc = roc_auc_score(all_labels, all_scores)

# 简单阈值分类
threshold = all_scores.mean()
pred_labels = (all_scores > threshold).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, pred_labels, average='binary'
)

print(f"\n📊 Test Results:")
print(f"  AUC:       {auc:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
