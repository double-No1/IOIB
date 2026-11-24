# train_svdd.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO
from tqdm import tqdm

from dataset import ImageFolderDataset
from svdd import DeepSVDD


# =========================
# Config
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_root = 'E:\Pycharm\OCC\data\images'
batch_size = 8
MODEL_PATH = './svdd.pth'
YOLO_WEIGHTS = './yolov8n.pt'


# =========================
# 1. Dataset (ONLY normal samples)
# =========================
train_set = ImageFolderDataset(data_root, is_train=True)
assert len(train_set) > 0, "❌ No training images found in data/zhan/"

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# =========================
# 2. Load YOLOv8 (frozen)
# =========================
yolo_wrapper = YOLO(YOLO_WEIGHTS)
yolo = yolo_wrapper.model.to(device)   # DetectionModel
yolo.eval()

for p in yolo.parameters():
    p.requires_grad = False


# =========================
# 3. Build feature extractor (BACKBONE + NECK)
# =========================
# yolo.model is nn.ModuleList
# [:10] removes Detect head
backbone_neck = yolo.model[:10]
feature_extractor = nn.Sequential(*backbone_neck).to(device)
feature_extractor.eval()


# =========================
# 4. Estimate feature dimension
# =========================
with torch.no_grad():
    x, _ = next(iter(train_loader))
    x = x.to(device)
    feat = feature_extractor(x)
    feat = torch.mean(feat, dim=[2, 3])  # GAP
    feat_dim = feat.shape[1]

print(f'📐 Feature dimension: {feat_dim}')


# =========================
# 5. Initialize Deep SVDD
# =========================
svdd = DeepSVDD(feat_dim).to(device)


# =========================
# 6. Compute center c
# =========================
features = []

with torch.no_grad():
    for x, _ in tqdm(train_loader, desc='Init center'):
        x = x.to(device)
        f = feature_extractor(x)
        f = torch.mean(f, dim=[2, 3])  # GAP
        features.append(f)

features = torch.cat(features, dim=0)
svdd.init_center(features)


# =========================
# 7. Save SVDD (only center)
# =========================
torch.save(svdd.state_dict(), MODEL_PATH)
print(f'✅ SVDD center saved to {MODEL_PATH}')
