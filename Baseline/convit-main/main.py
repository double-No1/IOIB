# # Copyright (c) 2015-present, Facebook, Inc.
# # All rights reserved.
# #
# # This source code is licensed under the CC-by-NC license found in the
# # LICENSE file in the root directory of this source tree.
# #
#
# import argparse
# import datetime
# import numpy as np
# import time
# import torch
# import torch.backends.cudnn as cudnn
# import json
# import gc
# from pathlib import Path
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
#
# from timm.data import Mixup
# from timm.models import create_model
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer
# from timm.utils import NativeScaler, get_state_dict, ModelEma
#
# # 注意：如果你的datasets.py、engine.py等文件不在当前目录，需保证导入路径正确
# from datasets import build_dataset  # 若build_dataset不兼容，后面会替换为自定义加载
# from engine import train_one_epoch, evaluate
# from samplers import RASampler
# import models
# import utils
#
#
# def get_args_parser():
#     parser = argparse.ArgumentParser('ConViT training and evaluation script', add_help=False)
#     parser.add_argument('--batch-size', default=8, type=int)  # 本地运行默认小批量
#     parser.add_argument('--epochs', default=30, type=int)  # 二分类50轮足够
#
#     # Model parameters
#     parser.add_argument('--model', default='convit_small', type=str, metavar='MODEL',
#                         help='Name of model to train')
#     parser.add_argument('--pretrained', action='store_true')
#
#     parser.add_argument('--input-size', default=224, type=int, help='images input size')
#     parser.add_argument('--embed_dim', default=48, type=int, help='embedding dimension per head')
#
#     parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
#                         help='Dropout rate (default: 0.)')
#     parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
#                         help='Drop path rate (default: 0.1)')
#     parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
#                         help='Drop block rate (default: None)')
#
#     parser.add_argument('--model-ema', action='store_true')
#     parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
#     parser.set_defaults(model_ema=False)
#     parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
#     parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
#
#     # Optimizer parameters
#     parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
#                         help='Optimizer (default: "adamw"')
#     parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
#                         help='Optimizer Epsilon (default: 1e-8)')
#     parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
#                         help='Optimizer Betas (default: None, use opt default)')
#     parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
#                         help='Clip gradient norm (default: None, no clipping)')
#     parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                         help='SGD momentum (default: 0.9)')
#     parser.add_argument('--weight-decay', type=float, default=0.05,
#                         help='weight decay (default: 0.05)')
#     # Learning rate schedule parameters
#     parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
#                         help='LR scheduler (default: "cosine"')
#     parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',  # 二分类调低学习率
#                         help='learning rate (default: 1e-4)')
#     parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
#                         help='learning rate noise on/off epoch percentages')
#     parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
#                         help='learning rate noise limit percent (default: 0.67)')
#     parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
#                         help='learning rate noise std-dev (default: 1.0)')
#     parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
#                         help='warmup learning rate (default: 1e-6)')
#     parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
#                         help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
#
#     parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
#                         help='epoch interval to decay LR')
#     parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
#                         help='epochs to warmup LR, if scheduler supports')
#     parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
#                         help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
#     parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
#                         help='patience epochs for Plateau LR scheduler (default: 10')
#     parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
#                         help='LR decay rate (default: 0.1)')
#
#     # Augmentation parameters
#     parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
#                         help='Color jitter factor (default: 0.4)')
#     parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
#                         help='Use AutoAugment policy. "v0" or "original". " + \
#                              "(default: rand-m9-mstd0.5-inc1)'),
#     parser.add_argument('--smoothing', type=float, default=0.05, help='Label smoothing (default: 0.05)')
#     parser.add_argument('--train-interpolation', type=str, default='bicubic',
#                         help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
#
#     parser.add_argument('--repeated-aug', action='store_true')
#     parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
#     parser.set_defaults(repeated_aug=True)
#
#     # * Random Erase params
#     parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
#                         help='Random erase prob (default: 0.25)')
#     parser.add_argument('--remode', type=str, default='pixel',
#                         help='Random erase mode (default: "pixel")')
#     parser.add_argument('--recount', type=int, default=1,
#                         help='Random erase count (default: 1)')
#     parser.add_argument('--resplit', action='store_true', default=False,
#                         help='Do not random erase first (clean) augmentation split')
#
#     # * Mixup params（二分类关闭mixup）
#     parser.add_argument('--mixup', type=float, default=0.0,
#                         help='mixup alpha, mixup enabled if > 0. (default: 0.0)')
#     parser.add_argument('--cutmix', type=float, default=0.0,
#                         help='cutmix alpha, cutmix enabled if > 0. (default: 0.0)')
#     parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
#                         help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
#     parser.add_argument('--mixup-prob', type=float, default=1.0,
#                         help='Probability of performing mixup or cutmix when either/both is enabled')
#     parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
#                         help='Probability of switching to cutmix when both mixup and cutmix enabled')
#     parser.add_argument('--mixup-mode', type=str, default='batch',
#                         help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
#
#     # Dataset parameters
#     parser.add_argument('--data-path', default=r"E:\Pycharm\Baseline\convit-main", type=str,  # 替换为你的本地路径
#                         help='dataset path')
#     parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19'],
#                         type=str, help='Image Net dataset path')
#     parser.add_argument('--sampling_ratio', default=1.,
#                         type=float, help='fraction of samples to keep in the training set of imagenet')
#     parser.add_argument('--nb_classes', default=2,
#                         type=int, help='number of classes (zhan/zuo)')
#     parser.add_argument('--inat-category', default='name',
#                         choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
#                         type=str, help='semantic granularity')
#
#     parser.add_argument('--output_dir', default='./5fold_output',
#                         help='path where to save, empty for no saving')
#     parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
#                         help='device to use for training / testing')
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--resume', default='', help='resume from checkpoint')
#     parser.add_argument('--save_every', default=None, type=int, help='save model every epochs')
#     parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
#                         help='start epoch')
#     parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
#     parser.add_argument('--num_workers', default=0, type=int)  # Windows本地运行设0
#     parser.add_argument('--pin-mem', action='store_true',
#                         help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
#     parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
#                         help='')
#     parser.set_defaults(pin_mem=False)  # 本地运行关闭pin_mem
#
#     # distributed training parameters（本地单卡关闭分布式）
#     parser.add_argument('--world_size', default=1, type=int,
#                         help='number of distributed processes')
#     parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
#
#     # locality parameters
#     parser.add_argument('--local_up_to_layer', default=10, type=int,
#                         help='number of GPSA layers')
#     parser.add_argument('--locality_strength', default=1., type=float,
#                         help='Determines how focused each head is around its attention center')
#
#     return parser
#
#
# # 自定义评估函数：计算AUC、Precision、Recall、F1
# def evaluate_metrics(model, data_loader, device):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     all_probs = []
#
#     with torch.no_grad():
#         for batch in data_loader:
#             images, labels = batch
#             images = images.to(device)
#             labels = labels.to(device)
#
#             # 前向传播获取预测概率
#             outputs = model(images)
#             probs = torch.softmax(outputs, dim=1)[:, 1]  # 取第二类（zuo）的概率（可根据需求调整）
#             preds = torch.argmax(outputs, dim=1)
#
#             # 收集结果
#             all_probs.extend(probs.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#     # 计算指标
#     auc = roc_auc_score(all_labels, all_probs)
#     precision = precision_score(all_labels, all_preds, average='binary')  # 二分类
#     recall = recall_score(all_labels, all_preds, average='binary')
#     f1 = f1_score(all_labels, all_preds, average='binary')
#
#     return {
#         'auc': auc,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'acc1': 100 * np.mean(np.array(all_preds) == np.array(all_labels))  # 保留原准确率
#     }
#
#
# # 加载完整数据集（用于5折拆分）
# def load_full_dataset(args):
#     # 定义数据变换（和原代码build_dataset保持一致）
#     train_transform = transforms.Compose([
#         transforms.Resize((args.input_size, args.input_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     val_transform = transforms.Compose([
#         transforms.Resize((args.input_size, args.input_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     # 加载完整数据集（train文件夹，包含zhan/zuo）
#     full_dataset = datasets.ImageFolder(
#         root=Path(args.data_path) / 'train',
#         transform=train_transform
#     )
#     # 提取标签（用于分层拆分）
#     labels = [label for _, label in full_dataset.samples]
#     return full_dataset, labels, val_transform
#
#
# def train_fold(args, fold_idx, train_indices, val_indices, full_dataset, val_transform):
#     # 拆分当前折的训练/验证集
#     train_subset = torch.utils.data.Subset(full_dataset, train_indices)
#     val_subset = torch.utils.data.Subset(full_dataset, [idx for idx in val_indices])
#
#     # 替换验证集的变换（避免训练增强）
#     val_subset.dataset = full_dataset
#     val_subset.transform = val_transform
#
#     # 构建DataLoader
#     train_loader = torch.utils.data.DataLoader(
#         train_subset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=True
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_subset,
#         batch_size=args.batch_size * 2,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem
#     )
#
#     # 创建模型（每折重新初始化）
#     model = create_model(
#         args.model,
#         pretrained=args.pretrained,
#         num_classes=args.nb_classes,
#         drop_rate=args.drop,
#         drop_path_rate=args.drop_path,
#         drop_block_rate=args.drop_block,
#         local_up_to_layer=args.local_up_to_layer,
#         locality_strength=args.locality_strength,
#         embed_dim=args.embed_dim,
#     )
#     model.to(args.device)
#
#     # 优化器/调度器/损失函数
#     linear_scaled_lr = args.lr * args.batch_size / 512.0
#     args.lr = linear_scaled_lr
#     optimizer = create_optimizer(args, model)
#     loss_scaler = NativeScaler()
#     lr_scheduler, _ = create_scheduler(args, optimizer)
#
#     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing) if args.smoothing else torch.nn.CrossEntropyLoss()
#
#     # 训练当前折
#     print(f"\n===== Training Fold {fold_idx + 1}/5 =====")
#     best_f1 = 0.0
#     for epoch in range(args.epochs):
#         gc.collect()
#         # 训练一轮
#         train_stats = train_one_epoch(
#             model, criterion, train_loader,
#             optimizer, args.device, epoch, loss_scaler,
#             args.clip_grad, None, None  # 关闭model_ema和mixup
#         )
#         # 更新学习率
#         lr_scheduler.step(epoch)
#         # 评估指标
#         val_stats = evaluate_metrics(model, val_loader, args.device)
#         print(
#             f"Fold {fold_idx + 1} Epoch {epoch}: AUC={val_stats['auc']:.4f}, P={val_stats['precision']:.4f}, Rec={val_stats['recall']:.4f}, F1={val_stats['f1']:.4f}")
#
#         # 保存当前折最优模型
#         if val_stats['f1'] > best_f1:
#             best_f1 = val_stats['f1']
#             fold_output_dir = Path(args.output_dir) / f"fold_{fold_idx + 1}"
#             fold_output_dir.mkdir(parents=True, exist_ok=True)
#             torch.save({
#                 'model': model.state_dict(),
#                 'epoch': epoch,
#                 'best_f1': best_f1,
#                 'metrics': val_stats
#             }, fold_output_dir / 'best_model.pth')
#
#     # 返回当前折的最优指标
#     return {
#         'auc': val_stats['auc'],
#         'precision': val_stats['precision'],
#         'recall': val_stats['recall'],
#         'f1': val_stats['f1']
#     }
#
#
# def main(args):
#     # 关闭分布式（本地单卡运行）
#     args.distributed = False
#     print("Args:", args)
#
#     # 固定随机种子
#     seed = args.seed
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     cudnn.benchmark = True
#
#     # 创建输出目录
#     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#
#     # 加载完整数据集并拆分5折
#     full_dataset, labels, val_transform = load_full_dataset(args)
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#
#     # 存储5折的指标
#     fold_metrics = []
#     start_time = time.time()
#
#     # 遍历5折
#     for fold_idx, (train_indices, val_indices) in enumerate(skf.split(full_dataset, labels)):
#         fold_result = train_fold(args, fold_idx, train_indices, val_indices, full_dataset, val_transform)
#         fold_metrics.append(fold_result)
#         print(f"Fold {fold_idx + 1} Complete: {fold_result}")
#
#     # 计算5折的平均指标+标准差
#     metrics_avg = {
#         'auc': np.mean([f['auc'] for f in fold_metrics]),
#         'precision': np.mean([f['precision'] for f in fold_metrics]),
#         'recall': np.mean([f['recall'] for f in fold_metrics]),
#         'f1': np.mean([f['f1'] for f in fold_metrics])
#     }
#     metrics_std = {
#         'auc': np.std([f['auc'] for f in fold_metrics]),
#         'precision': np.std([f['precision'] for f in fold_metrics]),
#         'recall': np.std([f['recall'] for f in fold_metrics]),
#         'f1': np.std([f['f1'] for f in fold_metrics])
#     }
#
#     # 打印最终结果
#     print("\n===== 5-Fold Cross Validation Results =====")
#     for metric in ['auc', 'precision', 'recall', 'f1']:
#         print(f"{metric.upper()}: {metrics_avg[metric]:.4f} ± {metrics_std[metric]:.4f}")
#
#     # 保存结果到文件
#     with (Path(args.output_dir) / '5fold_results.json').open('w') as f:
#         json.dump({
#             'fold_metrics': fold_metrics,
#             'average': metrics_avg,
#             'std': metrics_std,
#             'total_time': str(datetime.timedelta(seconds=int(time.time() - start_time)))
#         }, f, indent=4)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('ConViT 5折交叉验证', parents=[get_args_parser()])
#     args = parser.parse_args()
#     # 直接运行main（无需命令行传参，参数已固化在get_args_parser中）
#     main(args)
#


# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import gc
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

import models
import utils


# ===================== 1. 无监督核心组件 =====================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        labels = torch.arange(2 * batch_size, device=z.device)
        labels = (labels + batch_size) % (2 * batch_size)
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)
        loss = self.criterion(sim_matrix, labels)
        return loss


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)
        return [view1, view2]


class ConViTWithProjection(nn.Module):
    def __init__(self, base_model, feat_dim=128):
        super().__init__()
        self.base_model = base_model
        backbone_dim = base_model.head.in_features
        self.base_model.head = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, feat_dim)
        )
        # 用于下游评估的分类头（线性评估时使用）
        self.classification_head = nn.Linear(backbone_dim, 2)  # zhan/zuo二分类

    def forward(self, x, mode='unsup'):
        """
        mode: 'unsup' → 输出投影特征；'linear' → 输出分类概率
        """
        feat = self.base_model(x)
        if mode == 'unsup':
            return self.projection_head(feat)
        elif mode == 'linear':
            return self.classification_head(feat)


# ===================== 2. 指标计算函数 =====================
# 2.1 无监督指标：视图一致性得分（越高越好）
def calculate_view_consistency(model, data_loader, device):
    """计算同一图片两个增强视图的特征余弦相似度均值"""
    model.eval()
    consistency_scores = []
    with torch.no_grad():
        for images, _ in data_loader:
            view1, view2 = images[0].to(device), images[1].to(device)
            # 获取投影特征并归一化
            z1 = F.normalize(model(view1, mode='unsup'), dim=1)
            z2 = F.normalize(model(view2, mode='unsup'), dim=1)
            # 计算余弦相似度
            cos_sim = torch.sum(z1 * z2, dim=1)
            consistency_scores.extend(cos_sim.cpu().numpy())
    return np.mean(consistency_scores)


# 2.2 下游线性评估指标（包含Precision/Recall/AUC/F1/Acc）
def linear_evaluation(model, data_loader, device):
    """冻结backbone，仅评估分类头性能"""
    model.eval()
    # 冻结backbone和投影头，只使用分类头
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.projection_head.parameters():
        param.requires_grad = False

    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            # 输出分类概率
            outputs = model(images, mode='linear')
            probs = torch.softmax(outputs, dim=1)[:, 1]  # zuo类概率
            preds = torch.argmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算所有有监督指标（显式计算Precision/Recall）
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=1)
    acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))

    # 额外输出正负样本分布（辅助分析）
    print(f"线性评估样本分布 → 真实标签: 0(zhan)={np.sum(np.array(all_labels) == 0)}, 1(zuo)={np.sum(np.array(all_labels) == 1)}")
    print(f"线性评估预测分布 → 预测标签: 0(zhan)={np.sum(np.array(all_preds) == 0)}, 1(zuo)={np.sum(np.array(all_preds) == 1)}")

    return {
        'linear_auc': auc,
        'linear_precision': precision,
        'linear_recall': recall,
        'linear_f1': f1,
        'linear_acc': acc
    }


# ===================== 3. 数据加载函数 =====================
# 3.1 无监督训练数据（仅zhan类，双视图增强）
def load_unsup_train_data(args):
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size + 32, args.input_size + 32)),
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    two_crop_transform = TwoCropsTransform(train_transform)

    zhan_dir = Path(args.data_path) / 'train' / 'zhan'
    img_paths = list(zhan_dir.glob('**/*.png'))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"zhan文件夹下未找到PNG图片：{zhan_dir}")
    print(f"无监督训练数据：{len(img_paths)}张zhan类图片")

    class ZhanDataset(torch.utils.data.Dataset):
        def __init__(self, img_paths, transform):
            self.img_paths = img_paths
            self.transform = transform

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img = Image.open(self.img_paths[idx]).convert('RGB')
            return self.transform(img), 0

    dataset = ZhanDataset(img_paths, two_crop_transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )


# 3.2 下游评估数据（完整zhan+zuo，带标签）
def load_linear_eval_data(args):
    eval_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 加载完整训练集作为评估集（也可以单独划分验证集）
    full_dataset = datasets.ImageFolder(
        root=Path(args.data_path) / 'train',
        transform=eval_transform
    )
    print(f"线性评估数据：{len(full_dataset)}张图片，包含{full_dataset.classes}类")
    return torch.utils.data.DataLoader(
        full_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem
    )


# ===================== 4. 训练函数 =====================
def train_one_epoch_unsup(model, criterion, data_loader, optimizer, device, epoch):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data_iter_step, (images, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        view1, view2 = images[0].to(device), images[1].to(device)
        z1 = model(view1, mode='unsup')
        z2 = model(view2, mode='unsup')
        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ===================== 5. 参数配置 & 主函数 =====================
def get_args_parser():
    parser = argparse.ArgumentParser('ConViT 无监督训练（含完整指标）', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--eval_interval', default=5, type=int, help='每N轮评估一次指标')

    parser.add_argument('--model', default='convit_small', type=str)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--embed_dim', default=48, type=int)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--drop-path', type=float, default=0.1)
    parser.add_argument('--drop-block', type=float, default=None)

    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--feat_dim', type=int, default=128)

    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt-eps', default=1e-8, type=float)
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+')
    parser.add_argument('--clip-grad', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    # Learning rate schedule parameters 部分，补全cooldown_epochs的默认值
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        dest='cooldown_epochs',  # 显式指定dest，避免解析错误
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')

    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-lr', type=float, default=1e-6)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--decay-epochs', type=float, default=30)
    parser.add_argument('--warmup-epochs', type=int, default=5)

    parser.add_argument('--data-path', default=r"E:\Pycharm\Baseline\convit-main", type=str)
    parser.add_argument('--output_dir', default='./unsup_with_full_metrics')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true', default=False)

    parser.add_argument('--local_up_to_layer', default=10, type=int)
    parser.add_argument('--locality_strength', default=1., type=float)
    return parser


def main(args):
    args.distributed = False
    print("Args:", args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据
    unsup_loader = load_unsup_train_data(args)
    linear_eval_loader = load_linear_eval_data(args)

    # 创建模型
    base_model = create_model(
        args.model, pretrained=args.pretrained, num_classes=1000,
        drop_rate=args.drop, drop_path_rate=args.drop_path,
        local_up_to_layer=args.local_up_to_layer, locality_strength=args.locality_strength,
        embed_dim=args.embed_dim
    )
    model = ConViTWithProjection(base_model, feat_dim=args.feat_dim)
    model.to(args.device)

    # 优化器 & 损失
    linear_scaled_lr = args.lr * args.batch_size / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = NTXentLoss(temperature=args.temperature)

    # 训练日志
    log_history = []
    best_consistency = 0.0
    start_time = time.time()

    # 主训练循环
    for epoch in range(args.epochs):
        # 无监督训练一轮
        train_stats = train_one_epoch_unsup(model, criterion, unsup_loader, optimizer, args.device, epoch)
        lr_scheduler.step(epoch)

        # 每eval_interval轮计算一次指标
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 1. 无监督指标：视图一致性
            consistency = calculate_view_consistency(model, unsup_loader, args.device)
            # 2. 下游指标：线性评估（含Precision/Recall）
            linear_metrics = linear_evaluation(model, linear_eval_loader, args.device)

            # 格式化打印所有指标
            print(f"\n===== Epoch {epoch} 完整指标汇总 =====")
            print(f"【无监督训练指标】")
            print(f"  - 对比损失: {train_stats['loss']:.4f}")
            print(f"  - 视图一致性得分: {consistency:.4f}")
            print(f"【下游线性评估指标】")
            print(f"  - AUC: {linear_metrics['linear_auc']:.4f}")
            print(f"  - Precision（精确率）: {linear_metrics['linear_precision']:.4f}")
            print(f"  - Recall（召回率）: {linear_metrics['linear_recall']:.4f}")
            print(f"  - F1-Score: {linear_metrics['linear_f1']:.4f}")
            print(f"  - 准确率(Acc): {linear_metrics['linear_acc']:.2f}%")
            print("=" * 60)

            # 保存最优模型（按视图一致性）
            if consistency > best_consistency:
                best_consistency = consistency
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_consistency': best_consistency,
                    'best_linear_metrics': linear_metrics
                }
                torch.save(save_dict, Path(args.output_dir) / 'best_model.pth')
                print(f"✅ 保存最优模型（一致性得分：{best_consistency:.4f}）")

            # 记录日志（包含所有指标）
            log_entry = {
                'epoch': epoch,
                'contrastive_loss': train_stats['loss'],
                'view_consistency': consistency,
                'auc': linear_metrics['linear_auc'],
                'precision': linear_metrics['linear_precision'],
                'recall': linear_metrics['linear_recall'],
                'f1': linear_metrics['linear_f1'],
                'accuracy': linear_metrics['linear_acc']
            }
            log_history.append(log_entry)

    # 保存训练日志（JSON格式，方便后续分析）
    with (Path(args.output_dir) / 'full_metrics_log.json').open('w') as f:
        json.dump(log_history, f, indent=4)

    # 训练完成最终汇总
    total_time = time.time() - start_time
    best_epoch = np.argmax([x['view_consistency'] for x in log_history])
    best_metrics = log_history[best_epoch]
    print(f"\n===== 训练完成 - 最终汇总 =====")
    print(f"总耗时：{datetime.timedelta(seconds=int(total_time))}")
    print(f"最优无监督指标 → 视图一致性得分: {best_consistency:.4f} (Epoch {best_epoch})")
    print(
        f"最优下游指标 → AUC: {best_metrics['auc']:.4f}, Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}, F1: {best_metrics['f1']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConViT 无监督训练', parents=[get_args_parser()])
    args = parser.parse_args()
    args.data_path = r"E:\Pycharm\Baseline\convit-main"
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)