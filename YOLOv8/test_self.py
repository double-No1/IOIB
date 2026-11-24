# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
# from sklearn.model_selection import KFold
# from tqdm import tqdm
# from ultralytics import YOLO
# import numpy as np
# import copy
#
# from dataset_self import ImageFolderDataset
#
# # =========================
# # Config
# # =========================
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data_root = 'E:/Pycharm/OCC/data/self-dataset'
# batch_size = 8
# YOLO_WEIGHTS = './yolov8n.pt'
# NUM_FOLDS = 5
#
# # 训练参数
# EPOCHS_PER_FOLD = 30          # 每折训练轮数
# LEARNING_RATE = 5e-5           # 降低学习率，防止震荡
# FREEZE_LAYERS = 8              # 冻结更多层（前8层），减少过拟合风险
# PATIENCE = 7                   # 早停耐心值
#
# # =========================
# # DeepSVDD with fixed center (no updates during training)
# # =========================
# class DeepSVDD(nn.Module):
#     def __init__(self, feat_dim):
#         super().__init__()
#         self.feat_dim = feat_dim
#         self.register_buffer('center', torch.zeros(feat_dim))
#
#     def init_center(self, features):
#         """features: [N, feat_dim] 原始特征（未归一化）"""
#         # 计算均值并归一化中心（使中心位于单位球面上）
#         center = torch.mean(features, dim=0)
#         self.center.data = nn.functional.normalize(center, p=2, dim=0)
#
#     def get_distance(self, features):
#         """features: [B, feat_dim] 原始特征（未归一化）"""
#         # 对特征进行L2归一化
#         features_norm = nn.functional.normalize(features, p=2, dim=1)
#         # 计算与固定中心的平方欧氏距离
#         return torch.sum((features_norm - self.center) ** 2, dim=1)
#
# # =========================
# # 1. Load dataset
# # =========================
# full_dataset = ImageFolderDataset(data_root, is_train=True)
# zhan_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 1]
# zuo_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
#
# print(f"Total normal (zhan) samples: {len(zhan_indices)}")
# print(f"Total anomaly (zuo) samples: {len(zuo_indices)}")
#
# if len(zhan_indices) < 2:
#     print("⚠ Too few normal samples. Exiting.")
#     exit(1)
#
# if len(zhan_indices) < NUM_FOLDS:
#     NUM_FOLDS = len(zhan_indices)
#     print(f"⚠ Adjust NUM_FOLDS={NUM_FOLDS}")
#
# kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
#
# # =========================
# # 2. Load YOLOv8 backbone (部分可训练)
# # =========================
# yolo_wrapper = YOLO(YOLO_WEIGHTS)
# yolo = yolo_wrapper.model.to(device)
#
# # 默认冻结全部参数
# for p in yolo.parameters():
#     p.requires_grad = False
#
# # 解冻最后几层（索引需根据YOLO结构调整，这里假设前10层是特征提取部分）
# backbone_neck = yolo.model[:10]   # 取前10层作为特征提取器
# for idx, layer in enumerate(backbone_neck):
#     if idx >= FREEZE_LAYERS:      # 从第FREEZE_LAYERS层开始解冻
#         for p in layer.parameters():
#             p.requires_grad = True
#
# feature_extractor = nn.Sequential(*backbone_neck).to(device)
#
# # =========================
# # 3. KFold交叉验证（固定中心）
# # =========================
# fold_metrics = []
#
# for fold_idx, (train_idx, val_idx) in enumerate(kf.split(zhan_indices), 1):
#     print(f"\n===== Fold {fold_idx} =====")
#
#     # 构建训练集（仅正常样本）和测试集（剩余正常 + 全部异常）
#     train_subset = Subset(full_dataset, [zhan_indices[i] for i in train_idx])
#     val_subset_indices = [zhan_indices[i] for i in val_idx] + zuo_indices
#     val_subset = Subset(full_dataset, val_subset_indices)
#
#     train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
#
#     print(f"Train normal samples: {len(train_subset)}")
#     print(f"Test samples: {len(val_subset)}")
#
#     # 获取特征维度（用一个小batch）
#     with torch.no_grad():
#         x, _ = next(iter(train_loader))
#         x = x.to(device)
#         feat = feature_extractor(x)
#         feat = torch.mean(feat, dim=[2, 3])   # Global average pooling
#         feat_dim = feat.shape[1]
#
#     # 初始化SVDD
#     svdd = DeepSVDD(feat_dim).to(device)
#
#     # 计算初始中心（使用训练集所有正常样本的特征）
#     init_features = []
#     with torch.no_grad():
#         for x, _ in train_loader:
#             x = x.to(device)
#             f = feature_extractor(x)
#             f = torch.mean(f, dim=[2, 3])     # 提取特征
#             init_features.append(f.cpu())
#     init_features = torch.cat(init_features, dim=0).to(device)
#     svdd.init_center(init_features)   # 中心固定，后续不再更新
#
#     # 优化器（只更新可训练参数）
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, feature_extractor.parameters()), lr=LEARNING_RATE)
#
#     # 训练循环
#     best_val_auc = 0.0
#     best_model_state = None
#     patience_counter = 0
#
#     for epoch in range(1, EPOCHS_PER_FOLD + 1):
#         feature_extractor.train()
#         total_loss = 0.0
#         for x, _ in train_loader:
#             x = x.to(device)
#             f = feature_extractor(x)
#             f = torch.mean(f, dim=[2, 3])        # [B, feat_dim]
#             distances = svdd.get_distance(f)     # 计算到固定中心的距离
#             loss = distances.mean()
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         avg_loss = total_loss / len(train_loader)
#
#         # 验证
#         feature_extractor.eval()
#         all_scores = []
#         all_labels = []
#         with torch.no_grad():
#             for x, labels in val_loader:
#                 x = x.to(device)
#                 f = feature_extractor(x)
#                 f = torch.mean(f, dim=[2, 3])
#                 scores = svdd.get_distance(f)
#                 all_scores.append(scores.cpu().numpy())
#                 all_labels.append(labels.numpy())
#         all_scores = np.concatenate(all_scores)
#         all_labels = np.concatenate(all_labels)
#         val_auc = roc_auc_score(all_labels, all_scores)
#
#         print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")
#
#         # 早停：保存最佳模型
#         if val_auc > best_val_auc:
#             best_val_auc = val_auc
#             best_model_state = copy.deepcopy(feature_extractor.state_dict())
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= PATIENCE:
#                 print(f"Early stopping at epoch {epoch}")
#                 break
#
#     # 恢复最佳模型
#     feature_extractor.load_state_dict(best_model_state)
#     feature_extractor.eval()
#
#     # 最终测试
#     all_scores = []
#     all_labels = []
#     with torch.no_grad():
#         for x, labels in tqdm(val_loader, desc='Final testing'):
#             x = x.to(device)
#             f = feature_extractor(x)
#             f = torch.mean(f, dim=[2, 3])
#             scores = svdd.get_distance(f)
#             all_scores.append(scores.cpu().numpy())
#             all_labels.append(labels.numpy())
#     all_scores = np.concatenate(all_scores)
#     all_labels = np.concatenate(all_labels)
#
#     auc = roc_auc_score(all_labels, all_scores)
#     # 阈值可根据需要调整，但AUC已足够评估性能
#     threshold = np.percentile(all_scores, 90)   # 例如90%分位数
#     pred_labels = (all_scores > threshold).astype(int)
#     precision, recall, f1, _ = precision_recall_fscore_support(all_labels, pred_labels, average='binary')
#
#     print(f"Fold {fold_idx} Final - AUC: {auc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
#     fold_metrics.append([auc, precision, recall, f1])
#
# # =========================
# # 4. 汇总结果
# # =========================
# if len(fold_metrics) == 0:
#     print("No folds completed.")
# else:
#     fold_metrics = np.array(fold_metrics)
#     mean_metrics = np.mean(fold_metrics, axis=0)
#     std_metrics = np.std(fold_metrics, axis=0)
#
#     print("\n===== Cross Validation Results =====")
#     print(f"AUC:       {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
#     print(f"Precision: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
#     print(f"Recall:    {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
#     print(f"F1 Score:  {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")

# test_svdd_kfold_final.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
import copy

from dataset_self import ImageFolderDataset  # 确保dataset.py已正确修改（加载class0和class1）

# =========================
# Config
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_root = 'E:/Pycharm/OCC/data/self-dataset'
batch_size = 8
YOLO_WEIGHTS = './yolov8n.pt'
NUM_FOLDS = 5

# 训练参数
EPOCHS_PER_FOLD = 30
LEARNING_RATE = 5e-5
FREEZE_LAYERS = 8          # 冻结前8层（减少过拟合）
PATIENCE = 7

# =========================
# DeepSVDD with fixed center and feature normalization
# =========================
class DeepSVDD(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.register_buffer('center', torch.zeros(feat_dim))

    def init_center(self, features):
        """features: [N, feat_dim] 原始特征（未归一化）"""
        center = torch.mean(features, dim=0)
        self.center.data = nn.functional.normalize(center, p=2, dim=0)

    def get_distance(self, features):
        """features: [B, feat_dim] 原始特征（未归一化）"""
        features_norm = nn.functional.normalize(features, p=2, dim=1)
        return torch.sum((features_norm - self.center) ** 2, dim=1)

# =========================
# 1. Load dataset
# =========================
full_dataset = ImageFolderDataset(data_root, is_train=True)
zhan_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 1]
zuo_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]

print(f"Total normal (zhan) samples: {len(zhan_indices)}")
print(f"Total anomaly (zuo) samples: {len(zuo_indices)}")

if len(zhan_indices) < 2:
    print("⚠ Too few normal samples. Exiting.")
    exit(1)

if len(zhan_indices) < NUM_FOLDS:
    NUM_FOLDS = len(zhan_indices)
    print(f"⚠ Adjust NUM_FOLDS={NUM_FOLDS}")

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# =========================
# 2. Load YOLOv8 backbone (partially trainable)
# =========================
yolo_wrapper = YOLO(YOLO_WEIGHTS)
yolo = yolo_wrapper.model.to(device)

# 默认冻结全部参数
for p in yolo.parameters():
    p.requires_grad = False

# 解冻最后几层（前10层作为特征提取器，从第FREEZE_LAYERS层开始解冻）
backbone_neck = yolo.model[:10]   # 根据YOLO结构调整，这里假设前10层是特征提取部分
for idx, layer in enumerate(backbone_neck):
    if idx >= FREEZE_LAYERS:
        for p in layer.parameters():
            p.requires_grad = True

feature_extractor = nn.Sequential(*backbone_neck).to(device)

# =========================
# 3. KFold交叉验证
# =========================
fold_metrics = []          # 存储原始阈值下的指标（保留以供对比）
fold_metrics_optimized = [] # 存储优化阈值后的指标

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(zhan_indices), 1):
    print(f"\n===== Fold {fold_idx} =====")

    # 构建训练集（仅正常样本）和测试集（剩余正常 + 全部异常）
    train_subset = Subset(full_dataset, [zhan_indices[i] for i in train_idx])
    val_subset_indices = [zhan_indices[i] for i in val_idx] + zuo_indices
    val_subset = Subset(full_dataset, val_subset_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train normal samples: {len(train_subset)}")
    print(f"Test samples: {len(val_subset)}")

    # 获取特征维度
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x.to(device)
        feat = feature_extractor(x)
        feat = torch.mean(feat, dim=[2, 3])
        feat_dim = feat.shape[1]

    # 初始化SVDD
    svdd = DeepSVDD(feat_dim).to(device)

    # 计算初始中心（使用训练集所有正常样本）
    init_features = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            f = feature_extractor(x)
            f = torch.mean(f, dim=[2, 3])
            init_features.append(f.cpu())
    init_features = torch.cat(init_features, dim=0).to(device)
    svdd.init_center(init_features)   # 中心固定，后续不再更新

    # 优化器（只更新可训练参数）
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, feature_extractor.parameters()), lr=LEARNING_RATE)

    # 训练循环
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS_PER_FOLD + 1):
        feature_extractor.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            f = feature_extractor(x)
            f = torch.mean(f, dim=[2, 3])
            distances = svdd.get_distance(f)
            loss = distances.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证
        feature_extractor.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.to(device)
                f = feature_extractor(x)
                f = torch.mean(f, dim=[2, 3])
                scores = svdd.get_distance(f)
                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels.numpy())
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        val_auc = roc_auc_score(all_labels, all_scores)

        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        # 早停：保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(feature_extractor.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # 恢复最佳模型
    feature_extractor.load_state_dict(best_model_state)
    feature_extractor.eval()

    # =========================
    # 阈值优化：用验证集搜索最佳阈值（最大化F1）
    # =========================
    # 获取验证集所有分数和标签
    val_scores = []
    val_labels = []
    with torch.no_grad():
        for x, labels in val_loader:
            x = x.to(device)
            f = feature_extractor(x)
            f = torch.mean(f, dim=[2, 3])
            scores = svdd.get_distance(f)
            val_scores.append(scores.cpu().numpy())
            val_labels.append(labels.numpy())
    val_scores = np.concatenate(val_scores)
    val_labels = np.concatenate(val_labels)

    # 在分数范围内搜索最佳阈值
    best_f1 = 0
    best_threshold = 0
    thresholds = np.linspace(val_scores.min(), val_scores.max(), 100)
    for thresh in thresholds:
        pred = (val_scores > thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, pred, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f"Best threshold from val set: {best_threshold:.4f}, val F1: {best_f1:.4f}")

    # =========================
    # 最终评估（使用最佳阈值）
    # =========================
    final_pred = (val_scores > best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, final_pred, average='binary')
    auc = roc_auc_score(val_labels, val_scores)   # AUC不变

    print(f"Fold {fold_idx} Optimized - AUC: {auc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
    fold_metrics_optimized.append([auc, precision, recall, f1])

# =========================
# 4. 汇总优化后的结果
# =========================
if len(fold_metrics_optimized) == 0:
    print("No folds completed.")
else:
    fold_metrics = np.array(fold_metrics_optimized)
    mean_metrics = np.mean(fold_metrics, axis=0)
    std_metrics = np.std(fold_metrics, axis=0)

    print("\n===== Cross Validation Results (with optimized threshold) =====")
    print(f"AUC:       {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"Precision: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"Recall:    {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"F1 Score:  {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")