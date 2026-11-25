import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch

# 导入自定义模块
from dataloder_self import get_dataloaders
from models.ImprovedModels import PretrainedCNN  # 保持预训练 CNN

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
PATIENCE = 15
K = 5  # kNN 图的邻居数

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
# OCGIN 模型定义
# ==============================
class OCGIN(nn.Module):
    def __init__(self, dim_features, hidden_dim=128, num_layers=3):
        super().__init__()
        self.dim_features = dim_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GIN 卷积层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                nn_in = dim_features
            else:
                nn_in = hidden_dim
            mlp = nn.Sequential(
                nn.Linear(nn_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        # 可学习中心 c
        self.c = nn.Parameter(torch.randn(1, hidden_dim))

    def build_knn_graph(self, x):
        """
        根据节点特征 x (N, dim) 构建 kNN 图，返回边索引 (2, E)
        使用余弦相似度作为距离度量
        """
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(x_norm, x_norm.t())  # 余弦相似度矩阵
        # 排除自身，取 top-k 最大相似度作为邻居
        knn_vals, knn_idx = torch.topk(sim, k=K+1, dim=1)  # 包括自身
        knn_idx = knn_idx[:, 1:]  # 去掉自身
        # 生成边索引 (2, 2*N*K)
        row = torch.arange(x.size(0), device=x.device).repeat_interleave(K)
        col = knn_idx.reshape(-1)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def forward(self, x, batch=None):
        """
        x: 节点特征矩阵 (N, dim_features)
        batch: 每个节点所属的图索引（此处每个样本为一个节点，属于同一图？）
               实际上我们将整个 batch 视为一个图，因此 batch 全为零，但为了兼容 global_mean_pool，仍需传入
        """
        # 动态构建图
        edge_index = self.build_knn_graph(x)

        # GIN 卷积
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)

        # 全局池化得到每个节点的最终表示（这里节点即样本，所以可以直接返回 x）
        # 但为了统一，我们仍使用 global_mean_pool（因为 batch 全 0，每个节点输出就是本身）
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        out = global_mean_pool(x, batch)  # (num_graphs, hidden_dim)，这里 num_graphs=1
        # 扩展中心至节点数
        c_expanded = self.c.expand(x.size(0), -1)
        return out, c_expanded

# ==============================
# 模型初始化
# ==============================
cnn = PretrainedCNN(emb_dim=EMB_DIM).to(DEVICE)
model = OCGIN(dim_features=EMB_DIM, hidden_dim=128, num_layers=3).to(DEVICE)

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
    for images, _ in train_loader:  # 训练只有正常样本
        images = images.to(DEVICE)
        optimizer.zero_grad()
        embeddings = cnn(images)           # (batch_size, EMB_DIM)
        # 将整个 batch 作为一个图，节点特征为 embeddings
        # 构造 batch 向量（全 0，表示所有节点属于同一个图）
        batch = torch.zeros(embeddings.size(0), dtype=torch.long, device=DEVICE)
        z, c = model(embeddings, batch)    # z: (batch_size, hidden_dim)
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
        batch = torch.zeros(embeddings.size(0), dtype=torch.long, device=DEVICE)
        z, c = model(embeddings, batch)
        dist = torch.norm(z - c, dim=1)     # 异常分数
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
# 加载最佳模型并测试
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