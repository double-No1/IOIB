# test_svdd_cnn.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np

from dataset import ImageFolderDataset
from svdd import DeepSVDD

# =========================
# Config
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_root = 'E:\Pycharm\OCC\data\images'
batch_size = 16
NUM_RUNS = 5
SVDD_MODEL_PATH = './svdd_cnn.pth'

# =========================
# 1. Dataset
# =========================
test_set = ImageFolderDataset(data_root, is_train=False)
assert len(test_set) > 0, "❌ No test images found."
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# =========================
# 2. CNN backbone
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, feat_dim=256):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.feat_dim = feat_dim
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

feat_dim = 256
cnn = SimpleCNN(feat_dim=feat_dim).to(device)
cnn.eval()

# =========================
# 3. Load SVDD
# =========================
svdd = DeepSVDD(feat_dim).to(device)
svdd.load_state_dict(torch.load(SVDD_MODEL_PATH, map_location=device))
svdd.center = svdd.center.to(device)
svdd.eval()

# =========================
# 4. Repeat testing
# =========================
all_auc, all_precision, all_recall, all_f1 = [], [], [], []

for run in range(NUM_RUNS):
    all_scores, all_labels = [], []

    with torch.no_grad():
        for x, labels in tqdm(test_loader, desc=f"Run {run+1}"):
            x = x.to(device)
            f = cnn(x)  # CNN backbone
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
# 5. Mean and std
# =========================
print("\n===== Test Results (mean ± std) =====")
print(f"AUC:       {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
print(f"Precision: {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
print(f"Recall:    {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
print(f"F1 Score:  {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
