# import torch
# import torch.nn as nn
# from torch import optim
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import random
# import os
# import traceback
#
# from models.Trainer2 import GLAD_trainer
# from models.Losses import OCC_loss
# from torch_geometric.nn import GINConv
#
# # ==============================
# # 配置参数
# # ==============================
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# BATCH_SIZE = 32
# MAX_EPOCHS = 100
# LEARNING_RATE = 1e-3
# ONE_CLASS_LABEL = 1
# DATA_DIR = r"E:\Pycharm\data\self_sk"   # 请确认路径正确
# NUM_RUNS = 5
# K = 3
# EMB_DIM = 128
#
# # ==============================
# # 骨架数据集
# # ==============================
# class SkeletonDataset(Dataset):
#     def __init__(self, data_root, indices=None):
#         self.file_paths = []
#         self.labels = []
#
#         for label, folder in enumerate(['class0', 'class1']):
#             folder_path = os.path.join(data_root, folder)
#             if not os.path.isdir(folder_path):
#                 continue
#             for fname in os.listdir(folder_path):
#                 if fname.lower().endswith('.npy'):
#                     self.file_paths.append(os.path.join(folder_path, fname))
#                     self.labels.append(label)
#
#         if indices is not None:
#             self.file_paths = [self.file_paths[i] for i in indices]
#             self.labels = [self.labels[i] for i in indices]
#
#     def __len__(self):
#         return len(self.file_paths)
#
#     def __getitem__(self, idx):
#         data = np.load(self.file_paths[idx])
#         if data.ndim == 3:
#             data = data.mean(axis=0)
#         feat = torch.tensor(data.reshape(-1), dtype=torch.float32)
#         label = self.labels[idx]
#         return feat, label
#
# # ==============================
# # 划分函数
# # ==============================
# def split_skeleton_dataset(data_root, train_ratio=0.4, seed=42):
#     random.seed(seed)
#     all_labels = []
#     for label, folder in enumerate(['class0', 'class1']):
#         folder_path = os.path.join(data_root, folder)
#         if not os.path.isdir(folder_path):
#             continue
#         for fname in os.listdir(folder_path):
#             if fname.lower().endswith('.npy'):
#                 all_labels.append(label)
#
#     normal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]
#     abnormal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]
#
#     random.shuffle(normal_indices)
#     n_train = int(len(normal_indices) * train_ratio)
#     train_indices = normal_indices[:n_train]
#     test_indices = normal_indices[n_train:] + abnormal_indices
#     random.shuffle(test_indices)
#     return train_indices, test_indices
#
# # ==============================
# # OCGIN 模型（修正版）
# # ==============================
# class OCGIN(nn.Module):
#     def __init__(self, dim_features, hidden_dim=128, num_layers=3, K=3):
#         super().__init__()
#         self.dim_features = dim_features
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.K = K
#
#         self.convs = nn.ModuleList()
#         for i in range(num_layers):
#             if i == 0:
#                 nn_in = dim_features
#             else:
#                 nn_in = hidden_dim
#             mlp = nn.Sequential(
#                 nn.Linear(nn_in, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
#             self.convs.append(GINConv(mlp, train_eps=True))
#
#         self.c = nn.Parameter(torch.randn(1, hidden_dim))
#
#     def build_knn_graph(self, x):
#         num_nodes = x.size(0)
#         k_adj = min(self.K, num_nodes - 1)
#         if k_adj < 1:
#             return torch.empty((2, 0), dtype=torch.long, device=x.device)
#         x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
#         sim = torch.mm(x_norm, x_norm.t())
#         knn_vals, knn_idx = torch.topk(sim, k=k_adj+1, dim=1)
#         knn_idx = knn_idx[:, 1:]
#         row = torch.arange(num_nodes, device=x.device).repeat_interleave(k_adj)
#         col = knn_idx.reshape(-1)
#         edge_index = torch.stack([row, col], dim=0)
#         return edge_index
#
#     def forward(self, x):
#         # x: (batch_size, dim_features)
#         edge_index = self.build_knn_graph(x)
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = torch.relu(x)
#         # x 现在是每个节点的嵌入 (batch_size, hidden_dim)
#         c_expanded = self.c.expand(x.size(0), -1)
#         return x, c_expanded
#
# # ==============================
# # 多次运行
# # ==============================
# all_test_auc = []
# all_test_ap = []
# all_test_f1 = []
#
# for run in range(NUM_RUNS):
#     print(f"\n========== Run {run+1}/{NUM_RUNS} ==========")
#     seed = 42 + run
#     try:
#         train_idx, test_idx = split_skeleton_dataset(DATA_DIR, train_ratio=0.4, seed=seed)
#     except Exception as e:
#         print(f"划分失败: {e}")
#         continue
#
#     if len(train_idx) == 0:
#         print("训练集为空")
#         continue
#
#     train_dataset = SkeletonDataset(DATA_DIR, indices=train_idx)
#     test_dataset = SkeletonDataset(DATA_DIR, indices=test_idx)
#
#     if len(train_dataset) == 0 or len(test_dataset) == 0:
#         print("数据集为空")
#         continue
#
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#
#     print(f"训练集大小: {len(train_dataset)}")
#     print(f"测试集大小: {len(test_dataset)}")
#
#     sample_feat, _ = train_dataset[0]
#     feat_dim = sample_feat.shape[0]
#     print(f"特征维度: {feat_dim}")
#
#     model = OCGIN(dim_features=feat_dim, hidden_dim=EMB_DIM, num_layers=3, K=K).to(DEVICE)
#     loss_function = OCC_loss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
#     trainer = GLAD_trainer(model, loss_function, device=DEVICE)
#
#     try:
#         val_loss, val_auc, test_auc, test_ap, test_f1, score, target = trainer.train(
#             train_loader=train_loader,
#             cls=ONE_CLASS_LABEL,
#             max_epochs=MAX_EPOCHS,
#             optimizer=optimizer,
#             test_loader=test_loader,
#             log_every=10
#         )
#     except Exception as e:
#         print(f"训练过程出错: {e}")
#         traceback.print_exc()   # 打印详细堆栈
#         continue
#
#     all_test_auc.append(test_auc)
#     all_test_ap.append(test_ap)
#     all_test_f1.append(test_f1)
#
# if len(all_test_auc) == 0:
#     print("没有成功运行")
# else:
#     print("\n========== 最终结果 ==========")
#     print(f"ROC‑AUC : {np.mean(all_test_auc):.4f} ± {np.std(all_test_auc):.4f}")
#     print(f"AP      : {np.mean(all_test_ap):.4f} ± {np.std(all_test_ap):.4f}")
#     print(f"F1      : {np.mean(all_test_f1):.4f} ± {np.std(all_test_f1):.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from torch_geometric.nn import GINConv

# ==============================
# 配置参数
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
MAX_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
ONE_CLASS_LABEL = 1
DATA_DIR = r"E:\Pycharm\data\self_sk"   # 确认路径
NUM_RUNS = 5
K = 5                         # kNN 邻居数（适当增大）
EMB_DIM = 256                  # 隐藏维度（增大）
NUM_LAYERS = 4                 # GIN 层数
DROPOUT = 0.3                  # Dropout 比例
PATIENCE = 30                  # 早停耐心值

# ==============================
# 骨架数据集（与之前相同）
# ==============================
class SkeletonDataset(Dataset):
    def __init__(self, data_root, indices=None):
        self.file_paths = []
        self.labels = []

        for label, folder in enumerate(['class0', 'class1']):
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith('.npy'):
                    self.file_paths.append(os.path.join(folder_path, fname))
                    self.labels.append(label)

        if indices is not None:
            self.file_paths = [self.file_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        if data.ndim == 3:
            data = data.mean(axis=0)
        feat = torch.tensor(data.reshape(-1), dtype=torch.float32)
        label = self.labels[idx]
        return feat, label

# ==============================
# 划分数据集（训练集：仅正常；验证集：部分正常+部分异常；测试集：剩余正常+剩余异常）
# ==============================
def split_dataset(data_root, train_ratio=0.4, val_ratio=0.1, seed=42):
    random.seed(seed)
    all_labels = []
    for label, folder in enumerate(['class0', 'class1']):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith('.npy'):
                all_labels.append(label)

    normal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]
    abnormal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]

    random.shuffle(normal_indices)
    n_normal = len(normal_indices)
    n_train_norm = int(n_normal * train_ratio)
    n_val_norm = int(n_normal * val_ratio)
    train_norm = normal_indices[:n_train_norm]
    val_norm = normal_indices[n_train_norm:n_train_norm+n_val_norm]
    test_norm = normal_indices[n_train_norm+n_val_norm:]

    random.shuffle(abnormal_indices)
    n_abnormal_val = int(len(abnormal_indices) * val_ratio)
    val_abnormal = abnormal_indices[:n_abnormal_val]
    test_abnormal = abnormal_indices[n_abnormal_val:]

    train_idx = train_norm
    val_idx = val_norm + val_abnormal
    test_idx = test_norm + test_abnormal

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    return train_idx, val_idx, test_idx

# ==============================
# OCGIN 模型（增强版：加入 Dropout）
# ==============================
class OCGIN(nn.Module):
    def __init__(self, dim_features, hidden_dim=128, num_layers=3, K=3, dropout=0.0):
        super().__init__()
        self.dim_features = dim_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.K = K
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                nn_in = dim_features
            else:
                nn_in = hidden_dim
            mlp = nn.Sequential(
                nn.Linear(nn_in, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.c = nn.Parameter(torch.randn(1, hidden_dim))

    def build_knn_graph(self, x):
        num_nodes = x.size(0)
        k_adj = min(self.K, num_nodes - 1)
        if k_adj < 1:
            return torch.empty((2, 0), dtype=torch.long, device=x.device)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(x_norm, x_norm.t())
        knn_vals, knn_idx = torch.topk(sim, k=k_adj+1, dim=1)
        knn_idx = knn_idx[:, 1:]
        row = torch.arange(num_nodes, device=x.device).repeat_interleave(k_adj)
        col = knn_idx.reshape(-1)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def forward(self, x):
        edge_index = self.build_knn_graph(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        c_expanded = self.c.expand(x.size(0), -1)
        return x, c_expanded

# ==============================
# 训练与评估函数
# ==============================
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for features, _ in loader:   # 训练集只有正常样本
        features = features.to(DEVICE)
        optimizer.zero_grad()
        z, c = model(features)
        loss = torch.mean(torch.norm(z - c, dim=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_scores = []
    all_targets = []
    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        z, c = model(features)
        dist = torch.norm(z - c, dim=1).cpu().numpy()
        all_scores.append(dist)
        all_targets.append(labels.cpu().numpy())
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    return all_scores, all_targets

def find_best_threshold(val_scores, val_targets):
    thresholds = np.linspace(val_scores.min(), val_scores.max(), 200)
    best_f1 = 0
    best_th = thresholds[0]
    for th in thresholds:
        preds = (val_scores > th).astype(int)
        f1 = f1_score(val_targets, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1

# ==============================
# 多次运行
# ==============================
all_results = {
    'auc': [], 'ap': [], 'f1': [], 'precision': [], 'recall': []
}

for run in range(NUM_RUNS):
    print(f"\n========== Run {run+1}/{NUM_RUNS} ==========")
    seed = 42 + run
    try:
        train_idx, val_idx, test_idx = split_dataset(DATA_DIR, train_ratio=0.4, val_ratio=0.1, seed=seed)
    except Exception as e:
        print(f"划分失败: {e}")
        continue

    if len(train_idx) == 0:
        print("训练集为空")
        continue

    train_dataset = SkeletonDataset(DATA_DIR, indices=train_idx)
    val_dataset = SkeletonDataset(DATA_DIR, indices=val_idx)
    test_dataset = SkeletonDataset(DATA_DIR, indices=test_idx)

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("数据集为空")
        continue

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"训练集: {len(train_dataset)} (仅正常)")
    print(f"验证集: {len(val_dataset)} (正常+异常)")
    print(f"测试集: {len(test_dataset)} (正常+异常)")

    sample_feat, _ = train_dataset[0]
    feat_dim = sample_feat.shape[0]
    print(f"特征维度: {feat_dim}")

    model = OCGIN(dim_features=feat_dim, hidden_dim=EMB_DIM, num_layers=NUM_LAYERS, K=K, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_auc = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS+1):
        train_loss = train_epoch(model, train_loader, optimizer)

        # 验证
        val_scores, val_targets = evaluate(model, val_loader)
        val_auc = roc_auc_score(val_targets, val_scores)
        val_ap = average_precision_score(val_targets, val_scores)

        scheduler.step(train_loss)

        # 早停与模型保存（基于验证集 AUC）
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停于 epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 在验证集上选择最佳阈值
    val_scores, val_targets = evaluate(model, val_loader)
    best_th, best_val_f1 = find_best_threshold(val_scores, val_targets)
    print(f"验证集最佳阈值 = {best_th:.4f}, 对应 F1 = {best_val_f1:.4f}")

    # 测试集评估
    test_scores, test_targets = evaluate(model, test_loader)
    test_auc = roc_auc_score(test_targets, test_scores)
    test_ap = average_precision_score(test_targets, test_scores)
    test_preds = (test_scores > best_th).astype(int)
    test_f1 = f1_score(test_targets, test_preds)
    test_prec = precision_score(test_targets, test_preds)
    test_rec = recall_score(test_targets, test_preds)

    print(f"测试集 AUC: {test_auc:.4f}, AP: {test_ap:.4f}, F1: {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

    all_results['auc'].append(test_auc)
    all_results['ap'].append(test_ap)
    all_results['f1'].append(test_f1)
    all_results['precision'].append(test_prec)
    all_results['recall'].append(test_rec)

# ==============================
# 输出最终统计结果
# ==============================
if len(all_results['auc']) == 0:
    print("没有成功运行")
else:
    print("\n========== 最终结果 (均值 ± 标准差) ==========")
    for metric in ['auc', 'ap', 'f1', 'precision', 'recall']:
        values = all_results[metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper():10s}: {mean_val:.4f} ± {std_val:.4f}")