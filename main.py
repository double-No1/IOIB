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


def main():
    args = load_args()
    os.makedirs(args.save_path, exist_ok=True)

    # Full dataset for fold splitting
    full_dataset = NTUDataset(data_root=args.data_root, img_size=args.img_size, is_train=True)
    # Label 1 = normal, label 0 = anomaly
    zhan_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl == 1]

    kf = KFold(n_splits=5, shuffle=True, random_state=getattr(args, "seed", 42))

    # Global recording structure
    fold_all_epoch_metrics = []  # per fold -> list of metrics dict per epoch
    global_all_metrics = []      # flat list: fold, epoch, metrics, model_path

    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(zhan_indices), 1):
        print(f"\n===== Fold {fold} / 5 =====")

        # Construct train and test indices
        train_indices = [zhan_indices[i] for i in train_idx]
        val_zhan_indices = [zhan_indices[i] for i in val_idx]
        zuo_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl == 0]
        test_indices = val_zhan_indices + zuo_indices

        # Print sample statistics
        from collections import Counter
        train_labels = [full_dataset.labels[i] for i in train_indices]
        test_labels = [full_dataset.labels[i] for i in test_indices]
        print(f"Fold {fold} counts: train_total={len(train_indices)}, train_label_counts={Counter(train_labels)}")
        print(f"Fold {fold} counts: test_total={len(test_indices)}, test_label_counts={Counter(test_labels)}")

        # Create dataset and dataloader
        train_dataset = NTUDataset(args.data_root, args.img_size, is_train=True, indices=train_indices)
        test_dataset = NTUDataset(args.data_root, args.img_size, is_train=False, indices=test_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # Initialize model, optimizer, scheduler
        model = Model(args, input_dim=args.input_dim).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # Record metrics for all epochs in this fold
        fold_epoch_metrics = []

        # Training loop for current fold
        for epoch in range(args.epochs):
            print(f"\nFold {fold} - Epoch {epoch + 1}/{args.epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, args)
            print(f"  Train Loss: {train_loss:.4f}")

            metrics = evaluate(model, test_loader, args)
            scheduler.step()

            print(f"  AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | PR-AUC: {metrics.get('pr_auc', 0.0):.4f}")

            # Save model and state for current epoch
            epoch_model_path = os.path.join(args.save_path, f'fold_{fold}_epoch_{epoch}.pth')
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, epoch_model_path)

            # Record metrics
            fold_epoch_metrics.append(metrics)

            # Record to global list for best selection
            global_all_metrics.append({
                'fold': fold,
                'epoch': epoch,
                'metrics': metrics,
                'model_path': epoch_model_path
            })

        # Append metrics of current fold to total list
        fold_all_epoch_metrics.append(fold_epoch_metrics)
        print(f"Fold {fold} finished, recorded epochs: {len(fold_epoch_metrics)}")

    # All folds completed, check recorded metrics
    if len(global_all_metrics) == 0:
        print("Error: No epoch metrics recorded. Please check training process.")
        return

    # Find global best (max AUC across all folds and epochs)
    auc_values = [item['metrics']['auc'] for item in global_all_metrics]
    global_best_idx = int(np.argmax(auc_values))
    global_best_item = global_all_metrics[global_best_idx]
    global_best_fold = int(global_best_item['fold'])
    global_best_epoch = int(global_best_item['epoch'])
    global_best_auc = float(global_best_item['metrics']['auc'])

    print("\n===== Global Best Information ======")
    print(f"Global best from Fold {global_best_fold}, Epoch {global_best_epoch + 1} (index: {global_best_epoch})")
    print(f"Global best AUC: {global_best_auc:.4f}")
    print("Detailed metrics of this epoch:")
    for k, v in global_best_item['metrics'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"Model path: {global_best_item.get('model_path', 'N/A')}")

    # Get metrics at global best epoch for each fold (use last epoch if insufficient)
    global_best_epoch_metrics = []
    for fold in range(1, len(fold_all_epoch_metrics) + 1):
        fold_idx = fold - 1
        per_fold_epochs = fold_all_epoch_metrics[fold_idx]
        if global_best_epoch < len(per_fold_epochs):
            metrics = per_fold_epochs[global_best_epoch]
        else:
            metrics = per_fold_epochs[-1]
            print(f"⚠️ Fold {fold} has only {len(per_fold_epochs)} epochs, using last epoch instead of global epoch {global_best_epoch + 1}")
        global_best_epoch_metrics.append(metrics)
        print(f"\nFold {fold} metrics at global best epoch:")
        print(f"  AUC: {metrics['auc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

    # Calculate mean and std using ALL 5 folds
    print(f"\nUsing all {len(global_best_epoch_metrics)} folds for final statistics")

    auc_list = [m['auc'] for m in global_best_epoch_metrics]
    precision_list = [m['precision'] for m in global_best_epoch_metrics]
    recall_list = [m['recall'] for m in global_best_epoch_metrics]
    f1_list = [m['f1'] for m in global_best_epoch_metrics]

    print("\n===== 5-Fold Cross Validation Final Results =====")
    print(f"Folds used for evaluation: [1, 2, 3, 4, 5] (total {len(global_best_epoch_metrics)} folds)")
    print(f"Global best epoch: {global_best_epoch + 1} (from Fold {global_best_fold})")
    print(f"AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print(f"Precision: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
    print(f"Recall: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
    print(f"F1: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

    # Keep only the global best model, delete all other .pth files
    import glob
    best_model_path = global_best_item['model_path']
    all_pth_files = glob.glob(os.path.join(args.save_path, '*.pth'))

    if best_model_path in all_pth_files:
        other_files = [f for f in all_pth_files if f != best_model_path]
        if other_files:
            print(f"\nDeleting {len(other_files)} model files, keeping only global best: {best_model_path}")
            for f in other_files:
                os.remove(f)
                print(f"  Deleted: {f}")
        else:
            print("\nNo extra model files to delete.")
    else:
        print(f"\nWarning: Global best model path {best_model_path} not found in directory, skip deletion.")


if __name__ == '__main__':
    main()