import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

# 导入自定义模块
from dataloder_self import get_dataloaders  # 使用上面修改后的版本
from models.ImprovedModels import PretrainedCNN, OCGTL_Image

# ==============================
# 参数配置
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
MAX_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EMB_DIM = 128
DATA_DIR = r"E:\Pycharm\OCC\data\self-dataset"
PATIENCE = 15          # 早停耐心值

# ==============================
# 数据加载
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
# 模型初始化
# ==============================
cnn = PretrainedCNN(emb_dim=EMB_DIM).to(DEVICE)
model = OCGTL_Image(dim_features=EMB_DIM, config={'device': DEVICE}).to(DEVICE)

# ==============================
# 损失函数与优化器
# ==============================
def occ_loss(z, c):
    """Deep SVDD 损失：最小化到中心的距离"""
    dist = torch.norm(z - c, dim=1)
    return dist.mean()

optimizer = optim.Adam(list(cnn.parameters()) + list(model.parameters()),
                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ==============================
# 训练与验证函数
# ==============================
def train_epoch():
    cnn.train()
    model.train()
    total_loss = 0
    for images, _ in train_loader:   # 训练只有正常样本
        images = images.to(DEVICE)
        optimizer.zero_grad()
        embeddings = cnn(images)
        z, c = model(embeddings)
        loss = occ_loss(z, c)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def evaluate(loader):
    cnn.eval()
    model.eval()
    all_scores = []
    all_targets = []
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        embeddings = cnn(images)
        z, c = model(embeddings)
        dist = torch.norm(z - c, dim=1)   # 异常分数（越大越可能异常）
        all_scores.append(dist.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    return all_scores, all_targets

def find_best_threshold(val_scores, val_targets):
    """在验证集上寻找最佳阈值（最大化 F1）"""
    thresholds = np.linspace(val_scores.min(), val_scores.max(), 100)
    best_f1 = 0
    best_th = 0
    for th in thresholds:
        preds = (val_scores > th).astype(int)
        f1 = f1_score(val_targets, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1

# ==============================
# 训练循环
# ==============================
best_val_auc = 0
best_epoch = 0
patience_counter = 0

for epoch in range(1, MAX_EPOCHS+1):
    train_loss = train_epoch()

    # 验证
    val_scores, val_targets = evaluate(val_loader)
    val_auc = roc_auc_score(val_targets, val_scores)
    val_ap = average_precision_score(val_targets, val_scores)

    # 学习率调整
    scheduler.step(train_loss)

    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

    # 早停与模型保存
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        patience_counter = 0
        # 保存最佳模型
        torch.save({
            'cnn': cnn.state_dict(),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, 'best_model.pth')
        print(f"  -> 保存新最佳模型 (AUC={val_auc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"早停于 epoch {epoch}")
            break

# ==============================
# 加载最佳模型并在测试集上评估
# ==============================
checkpoint = torch.load('best_model.pth')
cnn.load_state_dict(checkpoint['cnn'])
model.load_state_dict(checkpoint['model'])
print(f"\n加载第 {checkpoint['epoch']} 轮最佳模型，验证集 AUC = {best_val_auc:.4f}")

# 测试集评估
test_scores, test_targets = evaluate(test_loader)

# 在验证集上找到最佳阈值
best_th, _ = find_best_threshold(val_scores, val_targets)
print(f"验证集最佳阈值 = {best_th:.4f}")

# 在测试集上使用该阈值
test_preds = (test_scores > best_th).astype(int)
test_auc = roc_auc_score(test_targets, test_scores)
test_ap = average_precision_score(test_targets, test_scores)
test_f1 = f1_score(test_targets, test_preds)
test_prec = precision_score(test_targets, test_preds)
test_rec = recall_score(test_targets, test_preds)

print("\n=== 最终测试结果 ===")
print(f"ROC‑AUC : {test_auc:.4f}")
print(f"AP      : {test_ap:.4f}")
print(f"F1      : {test_f1:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall   : {test_rec:.4f}")