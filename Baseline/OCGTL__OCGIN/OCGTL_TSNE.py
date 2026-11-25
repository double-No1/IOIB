import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 数据 & 模型
from dataloder_self import get_dataloaders
from models.ImprovedModels import PretrainedCNN, OCGTL_Image

# ==============================
# 参数
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EMB_DIM = 128
DATA_DIR = r"E:\Pycharm\OCC\data\self-dataset"
PATIENCE = 15

# ==============================
# 数据
# ==============================
class Args:
    data_root = DATA_DIR
    img_size = 224
    batch_size = BATCH_SIZE
    num_workers = 0
    seed = 42

args = Args()
train_loader, val_loader, test_loader = get_dataloaders(args)

# ==============================
# 模型
# ==============================
cnn = PretrainedCNN(emb_dim=EMB_DIM).to(DEVICE)
model = OCGTL_Image(dim_features=EMB_DIM, config={'device': DEVICE}).to(DEVICE)

# ==============================
# 损失
# ==============================
def occ_loss(z, c):
    return torch.norm(z - c, dim=1).mean()

optimizer = optim.Adam(list(cnn.parameters()) + list(model.parameters()),
                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ==============================
# 训练
# ==============================
def train_epoch():
    cnn.train()
    model.train()
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(DEVICE)

        optimizer.zero_grad()
        emb = cnn(images)
        z, c = model(emb)

        loss = occ_loss(z, c)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(train_loader.dataset)

# ==============================
# 评估
# ==============================
@torch.no_grad()
def evaluate(loader):
    cnn.eval()
    model.eval()

    scores, targets = [], []

    for images, labels in loader:
        images = images.to(DEVICE)

        emb = cnn(images)
        z, c = model(emb)

        dist = torch.norm(z - c, dim=1)

        scores.append(dist.cpu().numpy())
        targets.append(labels.numpy())

    return np.concatenate(scores), np.concatenate(targets)

# ==============================
# 找阈值
# ==============================
def find_best_threshold(val_scores, val_targets):
    thresholds = np.linspace(val_scores.min(), val_scores.max(), 100)
    best_f1, best_th = 0, 0

    for th in thresholds:
        preds = (val_scores > th).astype(int)
        f1 = f1_score(val_targets, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th, best_f1

# ==============================
# TSNE函数（核心）
# ==============================
def run_tsne(cnn, model, loader, save_path):
    cnn.eval()
    model.eval()

    feats = []
    labels = []

    safe_cnt = 0
    unsafe_cnt = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)

            emb = cnn(images)
            z, _ = model(emb)

            z = torch.nn.functional.normalize(z, dim=1)

            z = z.cpu().numpy()
            targets = targets.numpy()

            for f, t in zip(z, targets):
                if t == 1 and safe_cnt < 200:
                    feats.append(f)
                    labels.append(1)
                    safe_cnt += 1

                elif t == 0 and unsafe_cnt < 200:
                    feats.append(f)
                    labels.append(0)
                    unsafe_cnt += 1

                if safe_cnt >= 200 and unsafe_cnt >= 200:
                    break

            if safe_cnt >= 200 and unsafe_cnt >= 200:
                break

    feats = np.array(feats)
    labels = np.array(labels)

    print(f"[TSNE] safe={safe_cnt}, unsafe={unsafe_cnt}")

    tsne = TSNE(n_components=2, perplexity=20, n_iter=2000, init='pca')
    emb_2d = tsne.fit_transform(feats)

    plt.figure(figsize=(6,6))
    plt.scatter(emb_2d[labels==1,0], emb_2d[labels==1,1],
                c='blue', label='safe', s=15, alpha=0.7)

    plt.scatter(emb_2d[labels==0,0], emb_2d[labels==0,1],
                c='red', label='unsafe', s=15, alpha=0.7)

    plt.legend()
    plt.title("OCGTL", fontsize=14)
    plt.xticks([])
    plt.yticks([])

    os.makedirs("tsne_results", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[TSNE] saved to {save_path}")

# ==============================
# 训练循环
# ==============================
best_val_auc = 0
patience_counter = 0

for epoch in range(1, MAX_EPOCHS+1):
    loss = train_epoch()

    val_scores, val_targets = evaluate(val_loader)
    val_auc = roc_auc_score(val_targets, val_scores)

    scheduler.step(loss)

    print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val AUC {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0

        torch.save({
            'cnn': cnn.state_dict(),
            'model': model.state_dict()
        }, 'best_model.pth')

    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

# ==============================
# 测试
# ==============================
ckpt = torch.load('best_model.pth')
cnn.load_state_dict(ckpt['cnn'])
model.load_state_dict(ckpt['model'])

test_scores, test_targets = evaluate(test_loader)

best_th, _ = find_best_threshold(val_scores, val_targets)

test_preds = (test_scores > best_th).astype(int)

print("\n=== TEST ===")
print("AUC:", roc_auc_score(test_targets, test_scores))
print("F1 :", f1_score(test_targets, test_preds))

# ==============================
# TSNE
# ==============================
run_tsne(
    cnn,
    model,
    test_loader,
    "tsne_results/ocgtl_tsne1.png"
)