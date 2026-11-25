# # run_NTU_OCGTL.py
#
# import torch
# import yaml
# from torch import optim
#
# from models.Trainer import GLAD_trainer
#
# from ntu_dataloder import NTUSkeletonGraphDataset, split_dataset, collate_fn
# from torch.utils.data import Dataset, DataLoader, Subset
# # ==============================
# # 1) 配置参数
# # ==============================
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# BATCH_SIZE = 32
# MAX_EPOCHS = 100
# LEARNING_RATE = 1e-3
#
# # 你的 NTU 正常图标签 (one-class) 比如 "A009" 用 1，异常用 0
# ONE_CLASS_LABEL = 1
#
# # ==============================
# # 2) 配置参数
# # ==============================
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# BATCH_SIZE = 32
# MAX_EPOCHS = 100
# LEARNING_RATE = 1e-3
# ONE_CLASS_LABEL = 1  # 正常类标签，例如 A008=1
#
# DATA_DIR = r"E:\Pycharm\Baseline\GraphLevel-AnomalyDetection-master\data\graph_data"
#
# # ==============================
# # 3) 加载数据
# # ==============================
# dataset = NTUSkeletonGraphDataset(DATA_DIR)
# train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.4)
#
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
#
# print(f"✅ 数据集大小: {len(dataset)}")
# print(f"训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")
#
# # ==============================
# # 3) 定义模型 & loss
# # ==============================
# print("📌 加载 OCGTL 模型与配置……")
# # 从 config 文件读取 OCGTL 超参数
# with open("config_files/config_OCGTL.yml") as f:
#     cfg = yaml.safe_load(f)
#
# # 动态导入官方模型定义
# from models.Models import OCGTL
# from easydict import EasyDict
# from models.Losses import OCGTL_loss
# from models.GraphNets import GraphNorm
#
# # ==============================
# # 4) 定义模型
# # ==============================
# # 获取节点特征维度
# sample_batch = next(iter(train_loader))
# feat_dim = sample_batch[0].x.shape[1]
#
# # 配置字典
# cfg = {
#     'device': DEVICE,
#     'batch_size': 128,
#     'learning_rate': 0.001,
#     'training_epochs': 100,
#     'hidden_dim': 32,
#     'num_trans': 6,
#     'num_layers': 4,
#     'norm_layer': 'gn',
#     'bias': False,
#     'loss': 'OCGTL',
#     'l2': 0.,
#     'aggregation': 'add',
#     'optimizer': 'Adam',
#     'scheduler': {'class': 'StepLR', 'args': {'step_size': 100, 'gamma': 0.5}},
#     'early_stopper': {'class': 'Patience', 'args': {'patience': 100, 'use_train_loss': False}},
#     'shuffle': True,
#     'num_repeat': 1,
#     'save_scores': False,
#     'result_folder': 'RESULTS/RESULTS_'
# }
#
# # 初始化 OCGTL
# model = OCGTL(dim_features=feat_dim, config=cfg).to(DEVICE)
#
# # ==============================
# # 5) 定义 loss
#
#
# loss_function = OCGTL_loss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
# # ==============================
# # 6) 初始化 trainer
# # ==============================
# trainer = GLAD_trainer(model, loss_function, device=DEVICE)
#
# # ==============================
# # 7) 开始训练
# # ==============================
# val_loss, val_auc, test_auc, test_ap, test_f1, score, target = trainer.train(
#     train_loader=train_loader,
#     cls=ONE_CLASS_LABEL,
#     max_epochs=MAX_EPOCHS,
#     optimizer=optimizer,
#     test_loader=test_loader,
#     log_every=1
# )
#
# # ==============================
# # 8) 输出结果
# # ==============================
# print("\n=== 最终测试结果 ===")
# print(f"ROC‑AUC: {test_auc:.4f}")
# print(f"AP     : {test_ap:.4f}")
# print(f"F1     : {test_f1:.4f}")

# run_NTU_OCGTL.py

# import torch
# import torch.nn.functional as F
# from torch import optim
# from torch.utils.data import DataLoader
#
# from models.Trainer2 import GLAD_trainer
# from models.Models import OCGTL, OCGIN
# from models.Losses import OCGTL_loss, OCC_loss
# from ntu_dataloder import NTUSkeletonGraphDataset, split_dataset, collate_fn
#
# # ==============================
# # 配置参数
# # ==============================
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# BATCH_SIZE = 32
# MAX_EPOCHS = 100
# LEARNING_RATE = 1e-3
# ONE_CLASS_LABEL = 1  # 正常类标签，例如 A008=1
# DATA_DIR = r"E:\Pycharm\Baseline\GraphLevel-AnomalyDetection-master\data\graph_data"
#
# # ==============================
# # 加载数据
# # ==============================
# dataset = NTUSkeletonGraphDataset(DATA_DIR)
# train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.4)
#
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
#
# print(f"✅ 数据集大小: {len(dataset)}")
# print(f"训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")
#
# # ==============================
# # 获取节点特征维度
# # ==============================
# sample_batch = next(iter(train_loader))
# if isinstance(sample_batch, tuple):
#     sample_batch = sample_batch[0]
# feat_dim = sample_batch.x.shape[1]
#
# # ==============================
# # 初始化 OCGTL 模型
# # ==============================
# cfg = {
#     'device': DEVICE,
#     'batch_size': BATCH_SIZE,
#     'learning_rate': LEARNING_RATE,
#     'training_epochs': MAX_EPOCHS,
#     'hidden_dim': 32,
#     'num_trans': 6,
#     'num_layers': 4,
#     'norm_layer': 'gn',
#     'bias': False,
#     'loss': 'OCGTL',
#     'l2': 0.,
#     'aggregation': 'add',
#     'optimizer': 'Adam',
#     'scheduler': {'class': 'StepLR', 'args': {'step_size': 100, 'gamma': 0.5}},
#     'early_stopper': {'class': 'Patience', 'args': {'patience': 100, 'use_train_loss': False}},
#     'shuffle': True,
#     'num_repeat': 1,
#     'save_scores': False,
#     'result_folder': 'RESULTS/RESULTS_'
# }
#
# model = OCGIN(dim_features=feat_dim, config=cfg).to(DEVICE)
#
# # ==============================
# # 定义 loss 和 optimizer
# # ==============================
# loss_function = OCC_loss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
# # ==============================
# # 初始化 trainer
# # ==============================
# trainer = GLAD_trainer(model, loss_function, device=DEVICE)
#
# # ==============================
# # 开始训练
# # ==============================
# val_loss, val_auc, test_auc, test_ap, test_f1, score, target = trainer.train(
#     train_loader=train_loader,
#     cls=ONE_CLASS_LABEL,
#     max_epochs=MAX_EPOCHS,
#     optimizer=optimizer,
#     test_loader=test_loader,
#     log_every=1
# )
#
# # ==============================
# # 输出最终结果
# # ==============================
# print("\n=== 最终测试结果 ===")
# print(f"ROC‑AUC : {test_auc:.4f}")
# print(f"AP      : {test_ap:.4f}")
# print(f"F1      : {test_f1:.4f}")

# test.py
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataloder_self import NTUDataset, get_dataloaders
from models.Models import CNN_Embedder, OCGTL_Image
from models.Trainer2 import GLAD_trainer, OCC_loss

# ==============================
# 参数
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
ONE_CLASS_LABEL = 1
DATA_DIR = r"E:\Pycharm\OCC\data\self-dataset"
EMB_DIM = 128

# ==============================
# 数据
# ==============================
class Args:
    data_root = DATA_DIR
    img_size = 224
    batch_size = BATCH_SIZE
    num_workers = 0

args = Args()
train_loader, test_loader = get_dataloaders(args)

print(f"✅ 数据集大小: {len(train_loader.dataset) + len(test_loader.dataset)}")
print(f"训练集(zhan): {len(train_loader.dataset)} | 测试集(all): {len(test_loader.dataset)}")

# ==============================
# 模型
# ==============================
cnn = CNN_Embedder(emb_dim=EMB_DIM).to(DEVICE)
model = OCGTL_Image(dim_features=EMB_DIM, config={'device': DEVICE, 'hidden_dim': EMB_DIM}).to(DEVICE)

# ==============================
# loss & optimizer
# ==============================
loss_function = OCC_loss()
optimizer = optim.Adam(list(cnn.parameters()) + list(model.parameters()), lr=LEARNING_RATE)

# ==============================
# trainer
# ==============================
class ImageOCGTrainer(GLAD_trainer):
    def _train_epoch(self, train_loader, optimizer, cls):
        self.model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(self.device)
            optimizer.zero_grad()
            embeddings = cnn(images)
            z, c = self.model(embeddings)
            loss = self.loss_fun(z, c)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader, cls=1):
        self.model.eval()
        all_scores = []
        all_targets = []
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            embeddings = cnn(images)
            z, c = self.model(embeddings)
            dist = torch.norm(z - c, dim=1)
            all_scores.append(dist.cpu())
            all_targets.append(labels.cpu())
        all_scores = torch.cat(all_scores).numpy()
        all_targets = torch.cat(all_targets).numpy()
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
        test_auc = roc_auc_score(all_targets, all_scores)
        test_ap = average_precision_score(all_targets, all_scores)
        preds = (all_scores > all_scores.mean()).astype(int)
        test_f1 = f1_score(all_targets, preds)
        test_prec = precision_score(all_targets, preds)
        test_rec = recall_score(all_targets, preds)
        return test_auc, test_ap, test_f1, test_prec, test_rec, all_scores, all_targets

trainer = ImageOCGTrainer(model, loss_function, device=DEVICE)

# ==============================
# 训练
# ==============================
val_loss, val_auc, test_auc, test_ap, test_f1, scores, targets = trainer.train(
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    max_epochs=MAX_EPOCHS,
    cls=ONE_CLASS_LABEL
)

# ==============================
# 输出结果
# ==============================
print("\n=== 最终测试结果 ===")
print(f"ROC‑AUC : {test_auc:.4f}")
print(f"AP      : {test_ap:.4f}")
print(f"F1      : {test_f1:.4f}")
