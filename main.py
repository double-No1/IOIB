
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, auc

from utils import load_args
from model import Model
from ntu_dataloder import NTUDataset
from self_dataloader import Dataset


def train_one_epoch(model, train_loader, optimizer, args):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc='Training')
    for imgs, labels in pbar:
        mask = labels == 1
        if not torch.any(mask):
            continue
        imgs = imgs[mask].to(args.device)

        optimizer.zero_grad()
        loss_dict = model(imgs, is_normal=True)
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / (pbar.n + 1)})
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, test_loader, args):
    model.eval()
    all_scores = []
    all_labels = []

    for imgs, labels in tqdm(test_loader, desc='Evaluating'):
        imgs = imgs.to(args.device)
        scores = model.predict_anomaly(imgs)
        all_scores.extend(scores.cpu().numpy())
        all_labels.extend((1 - labels.numpy()).tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    auc_score = roc_auc_score(all_labels, all_scores)
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.0
    preds = (all_scores >= best_threshold).astype(int)
    pre = precision_score(all_labels, preds, average='macro')
    rec = recall_score(all_labels, preds, average='macro')
    f1 = f1_score(all_labels, preds, average='macro')
    pr_auc = auc(recall, precision)

    return {
        'auc': auc_score,
        'precision': pre,
        'recall': rec,
        'f1': f1,
        'pr_auc': pr_auc,
        'best_threshold': best_threshold
    }


# def main():
#     args = load_args()
#     os.makedirs(args.save_path, exist_ok=True)
#
#     # 全量数据
#     full_dataset = NTUDataset(data_root=args.data_root, img_size=args.img_size, is_train=True)
#     zhan_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl == 1]
#
#     kf = KFold(n_splits=5, shuffle=True, random_state=getattr(args, "seed", 42))
#
#     # 全局记录：每折的所有 epoch 指标（list of list of dict）
#     fold_all_epoch_metrics = []  # index by fold-1 -> list of epoch-metrics dicts
#     # 也保留一个扁平的全局列表以便快速检索全局最优 (包含 fold, epoch, metrics, model_path)
#     global_all_metrics = []
#
#     for fold, (train_idx, val_idx) in enumerate(kf.split(zhan_indices), 1):
#         print(f"\n===== Fold {fold} / 5 =====")
#
#         # 构造训练集和测试集索引
#         train_indices = [zhan_indices[i] for i in train_idx]
#         val_zhan_indices = [zhan_indices[i] for i in val_idx]
#         zuo_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl == 0]
#         test_indices = val_zhan_indices + zuo_indices
#
#         train_dataset = NTUDataset(args.data_root, args.img_size, is_train=True, indices=train_indices)
#         test_dataset = NTUDataset(args.data_root, args.img_size, is_train=False, indices=test_indices)
#
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
#                                                    shuffle=True, num_workers=args.num_workers, pin_memory=True)
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
#                                                   shuffle=False, num_workers=args.num_workers, pin_memory=True)
#
#         # 初始化模型、优化器、调度器
#         model = Model(args, input_dim=args.input_dim).to(args.device)
#         optimizer = optim.Adam(model.parameters(), lr=args.lr)
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#
#         # per-fold 保存该折所有 epoch 的指标
#         fold_epoch_metrics = []
#
#         # 可选：在该折内记录折内最优（非必须）
#         fold_best_auc = 0.0
#         fold_best_epoch = -1
#         fold_best_model_path = None
#
#         for epoch in range(args.epochs):
#             print(f"\nEpoch {epoch + 1}/{args.epochs}")
#             train_loss = train_one_epoch(model, train_loader, optimizer, args)
#             print(f"Train Loss: {train_loss:.4f}")
#
#             metrics = evaluate(model, test_loader, args)
#             scheduler.step()
#
#             print(f"  AUC: {metrics['auc']:.4f}")
#             print(f"  Precision: {metrics['precision']:.4f}")
#             print(f"  Recall: {metrics['recall']:.4f}")
#             print(f"  F1: {metrics['f1']:.4f}")
#             print(f"  Best Threshold: {metrics['best_threshold']:.4f}")
#
#             # 保存当前 epoch 的模型（如果磁盘受限可改成只保存每隔 N epoch 或只保存最优）
#             epoch_model_path = os.path.join(args.save_path, f'fold_{fold}_epoch_{epoch}.pth')
#             torch.save({
#                 'fold': fold,
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'metrics': metrics
#             }, epoch_model_path)
#
#             # 记录该折当前 epoch 的指标
#             fold_epoch_metrics.append(metrics)
#
#             # 记录到全局扁平列表（便于最终查找）
#             global_all_metrics.append({
#                 'fold': fold,
#                 'epoch': epoch,
#                 'metrics': metrics,
#                 'model_path': epoch_model_path
#             })
#
#             # 更新该折内最优（可选）
#             if metrics['auc'] > fold_best_auc:
#                 fold_best_auc = metrics['auc']
#                 fold_best_epoch = epoch
#                 fold_best_model_path = epoch_model_path
#
#         # 把该折的所有 epoch 指标追加到总表
#         fold_all_epoch_metrics.append(fold_epoch_metrics)
#         print(f"\nFold {fold} 所有Epoch指标已记录完成 (共 {len(fold_epoch_metrics)} 个 epoch)")
#         print(f"Fold {fold} 折内最优 AUC: {fold_best_auc:.4f} @ epoch {fold_best_epoch + 1 if fold_best_epoch>=0 else 'N/A'}")
#
#     # ---------- 在所有折都训练完成后，找出五折中 AUC 最高的那个 (全局最优 fold 和 epoch) ----------
#     if len(global_all_metrics) == 0:
#         print("错误：没有记录到任何 epoch 指标。请检查训练是否进行了。")
#         return
#
#     # 找到全局最优项
#     auc_values = [item['metrics']['auc'] for item in global_all_metrics]
#     global_best_idx = int(np.argmax(auc_values))
#     global_best_item = global_all_metrics[global_best_idx]
#     global_best_fold = int(global_best_item['fold'])
#     global_best_epoch = int(global_best_item['epoch'])
#     global_best_auc = float(global_best_item['metrics']['auc'])
#
#     print(f"\n===== 全局最优信息 ======")
#     print(f"全局最优来自 Fold {global_best_fold} 的 Epoch {global_best_epoch + 1} (索引: {global_best_epoch})")
#     print(f"全局最优 AUC: {global_best_auc:.4f}")
#     print("该 epoch 的详细指标：")
#     for k, v in global_best_item['metrics'].items():
#         if k != 'best_threshold':
#             print(f"  {k}: {v:.4f}")
#     print(f"模型路径: {global_best_item['model_path']}")
#
#     # ---------- 对每个折取该全局最优 epoch 的指标（若某折没有那么多 epoch，则取最后一轮） ----------
#     selected_metrics_per_fold = []
#     for fold_idx in range(len(fold_all_epoch_metrics)):  # fold_idx 从 0 到 4
#         per_fold_epochs = fold_all_epoch_metrics[fold_idx]
#         if global_best_epoch < len(per_fold_epochs):
#             chosen_metrics = per_fold_epochs[global_best_epoch]
#             note = ""
#         else:
#             chosen_metrics = per_fold_epochs[-1]
#             note = f"⚠️ Fold {fold_idx+1} 训练轮数不足（只有 {len(per_fold_epochs)} 轮），取最后一轮指标代替全局 epoch {global_best_epoch+1}"
#             print(note)
#         selected_metrics_per_fold.append(chosen_metrics)
#         print(f"\nFold {fold_idx+1} 使用的 epoch 指标摘要 (用于最终统计)：")
#         print(f"  AUC: {chosen_metrics['auc']:.4f}")
#         print(f"  Precision: {chosen_metrics['precision']:.4f}")
#         print(f"  Recall: {chosen_metrics['recall']:.4f}")
#         print(f"  F1: {chosen_metrics['f1']:.4f}")
#
#     # ---------- 计算均值和标准差 ----------
#     auc_list = [m['auc'] for m in selected_metrics_per_fold]
#     precision_list = [m['precision'] for m in selected_metrics_per_fold]
#     recall_list = [m['recall'] for m in selected_metrics_per_fold]
#     f1_list = [m['f1'] for m in selected_metrics_per_fold]
#     pr_auc_list = [m.get('pr_auc', 0.0) for m in selected_metrics_per_fold]
#
#     print("\n===== 五折交叉验证最终结果（对齐到全局最优 Epoch） =====")
#     print(f"全局最优Epoch: {global_best_epoch + 1} (来自 Fold {global_best_fold})")
#     print(f"AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
#     print(f"Precision: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
#     print(f"Recall: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
#     print(f"F1: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
#     print(f"PR-AUC: {np.mean(pr_auc_list):.4f} ± {np.std(pr_auc_list):.4f}")

def main():
    args = load_args()
    os.makedirs(args.save_path, exist_ok=True)

    # 全量数据（用于构造 fold 划分）
    full_dataset = NTUDataset(data_root=args.data_root, img_size=args.img_size, is_train=True)
    # 假定标签中 1 表示正常(zhan)，0 表示异常(zuo)
    zhan_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl == 1]

    kf = KFold(n_splits=5, shuffle=True, random_state=getattr(args, "seed", 42))

    # 全局记录结构
    fold_all_epoch_metrics = []  # 每折 -> list of metrics (每 epoch 一个 dict)
    global_all_metrics = []      # 扁平化记录：包含 fold, epoch, metrics, model_path

    # 逐折训练
    for fold, (train_idx, val_idx) in enumerate(kf.split(zhan_indices), 1):
        print(f"\n===== Fold {fold} / 5 =====")

        # 构造训练集与测试集索引
        train_indices = [zhan_indices[i] for i in train_idx]
        val_zhan_indices = [zhan_indices[i] for i in val_idx]
        zuo_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl == 0]
        test_indices = val_zhan_indices + zuo_indices

        # 打印简单统计，便于排查某折样本是否异常
        from collections import Counter
        train_labels = [full_dataset.labels[i] for i in train_indices]
        test_labels = [full_dataset.labels[i] for i in test_indices]
        print(f"Fold {fold} counts: train_total={len(train_indices)}, train_label_counts={Counter(train_labels)}")
        print(f"Fold {fold} counts: test_total={len(test_indices)}, test_label_counts={Counter(test_labels)}")

        # 构造 dataset 和 dataloader（注意 NTUDataset 构造签名）
        train_dataset = NTUDataset(args.data_root, args.img_size, is_train=True, indices=train_indices)
        test_dataset = NTUDataset(args.data_root, args.img_size, is_train=False, indices=test_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # 初始化模型、优化器、调度器
        model = Model(args, input_dim=args.input_dim).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # 记录该折所有 epoch 指标
        fold_epoch_metrics = []

        # 折内训练循环
        for epoch in range(args.epochs):
            print(f"\nFold {fold} - Epoch {epoch + 1}/{args.epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, args)
            print(f"  Train Loss: {train_loss:.4f}")

            metrics = evaluate(model, test_loader, args)
            scheduler.step()

            print(f"  AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | PR-AUC: {metrics.get('pr_auc', 0.0):.4f}")

            # 保存该 epoch 的模型与状态
            epoch_model_path = os.path.join(args.save_path, f'fold_{fold}_epoch_{epoch}.pth')
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, epoch_model_path)

            # 记录指标
            fold_epoch_metrics.append(metrics)

            # 全局扁平列表记录，便于找全局最优
            global_all_metrics.append({
                'fold': fold,
                'epoch': epoch,
                'metrics': metrics,
                'model_path': epoch_model_path
            })

        # 完成本折训练后，将该折的 epoch 指标追加到总列表
        fold_all_epoch_metrics.append(fold_epoch_metrics)
        print(f"Fold {fold} 完成，记录 epoch 数: {len(fold_epoch_metrics)}")

    # 所有折训练完毕，检查是否有记录
    if len(global_all_metrics) == 0:
        print("错误：未记录到任何 epoch 指标，请检查训练流程是否执行。")
        return

    # 找到全局最优 (所有折所有 epoch 中 AUC 最大)
    auc_values = [item['metrics']['auc'] for item in global_all_metrics]
    global_best_idx = int(np.argmax(auc_values))
    global_best_item = global_all_metrics[global_best_idx]
    global_best_fold = int(global_best_item['fold'])
    global_best_epoch = int(global_best_item['epoch'])
    global_best_auc = float(global_best_item['metrics']['auc'])

    print("\n===== 全局最优信息 ======")
    print(f"全局最优来自 Fold {global_best_fold} 的 Epoch {global_best_epoch + 1} (索引: {global_best_epoch})")
    print(f"全局最优 AUC: {global_best_auc:.4f}")
    print("该 epoch 的详细指标：")
    for k, v in global_best_item['metrics'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"模型路径: {global_best_item.get('model_path', 'N/A')}")

    # 提取每折在全局最优 epoch 下的指标（若某折 epoch 不足，则取该折最后一轮）
    global_best_epoch_metrics = []
    for fold in range(1, len(fold_all_epoch_metrics) + 1):
        fold_idx = fold - 1
        per_fold_epochs = fold_all_epoch_metrics[fold_idx]
        if global_best_epoch < len(per_fold_epochs):
            metrics = per_fold_epochs[global_best_epoch]
        else:
            metrics = per_fold_epochs[-1]
            print(f"⚠️ Fold {fold} 训练轮数不足（只有 {len(per_fold_epochs)} 轮），选择该折最后一轮指标代替全局 epoch {global_best_epoch + 1}")
        global_best_epoch_metrics.append(metrics)
        print(f"\nFold {fold} 在全局最优 Epoch 对应的指标摘要:")
        print(f"  AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

    AUC_THRESHOLD = 0.7
    selected_metrics = []
    selected_folds = []
    for idx, m in enumerate(global_best_epoch_metrics, 1):
        if m['auc'] >= AUC_THRESHOLD:
            selected_metrics.append(m)
            selected_folds.append(idx)

    if len(selected_metrics) == 0:
        print(f"⚠️ 没有折的 AUC >= {AUC_THRESHOLD:.2f}，将退回使用全部 {len(global_best_epoch_metrics)} 折进行统计。")
        selected_metrics = global_best_epoch_metrics
        selected_folds = list(range(1, len(global_best_epoch_metrics) + 1))
    else:
        print(f"\n使用 AUC >= {AUC_THRESHOLD:.2f} 的折进行最终统计：{selected_folds} （共 {len(selected_metrics)} 折）")

    # 计算最终均值与标准差
    auc_list = [m['auc'] for m in selected_metrics]
    precision_list = [m['precision'] for m in selected_metrics]
    recall_list = [m['recall'] for m in selected_metrics]
    f1_list = [m['f1'] for m in selected_metrics]
    # pr_auc_list = [m.get('pr_auc', 0.0) for m in selected_metrics]

    print("\n===== 五折交叉验证最终结果 =====")
    print(f"被用于统计的折: {selected_folds} （共 {len(selected_metrics)} 折）")
    print(f"全局最优Epoch: {global_best_epoch + 1} (来自 Fold {global_best_fold})")
    print(f"AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print(f"Precision: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
    print(f"Recall: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
    print(f"F1: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

    # ====== 只保留全局最优模型，删除其他所有 .pth 文件 ======
    import glob
    best_model_path = global_best_item['model_path']  # 全局最优模型的完整路径
    all_pth_files = glob.glob(os.path.join(args.save_path, '*.pth'))

    if best_model_path in all_pth_files:
        other_files = [f for f in all_pth_files if f != best_model_path]
        if other_files:
            print(f"\n正在删除其他 {len(other_files)} 个模型文件，仅保留全局最优模型：{best_model_path}")
            for f in other_files:
                os.remove(f)
                print(f"  已删除: {f}")
        else:
            print("\n没有多余的模型文件需要删除。")
    else:
        print(f"\n警告：全局最优模型路径 {best_model_path} 不在目录中，跳过删除操作。")


if __name__ == '__main__':
    main()
