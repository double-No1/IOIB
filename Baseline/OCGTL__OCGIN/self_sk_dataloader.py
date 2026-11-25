# dataloder_skeleton.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class SkeletonDataset(Dataset):
    def __init__(self, data_root, is_train=True, indices=None, max_frames=300, num_joints=25, coord_dim=3):
        self.data_root = data_root
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.feature_dim = num_joints * coord_dim  # 平均池化后特征维度

        self.file_paths = []
        self.labels = []

        for label, folder in enumerate(['class0', 'class1']):
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith('.npy'):
                    self.file_paths.append(os.path.join(folder_path, fname))
                    self.labels.append(label)

        if indices is not None:
            self.file_paths = [self.file_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        # 训练时可用简单增强：随机旋转、缩放等，但骨架数据增强较复杂，这里略过

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npy_path = self.file_paths[idx]
        label = self.labels[idx]
        data = np.load(npy_path)  # 假设 shape (T, V, C)
        # 对时间维度平均
        if data.ndim == 3:
            pooled = data.mean(axis=0)  # (V, C)
        else:
            # 如果已经是 (V, C) 则直接使用
            pooled = data
        # 展平
        feature = pooled.reshape(-1)  # (V*C,)
        # 如果特征维度不一致，可能需要填充或截断
        # 这里假设所有样本 V 和 C 固定
        feature = torch.tensor(feature, dtype=torch.float32)
        return feature, label

def get_dataloaders(args):
    data_root = args.data_root
    seed = getattr(args, 'seed', 42)
    random.seed(seed)

    # 构建索引
    all_files, all_labels = [], []
    for label, folder in enumerate(['class0', 'class1']):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith('.npy'):
                all_files.append(os.path.join(folder_path, fname))
                all_labels.append(label)

    normal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]
    abnormal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]

    random.shuffle(normal_indices)
    n_normal = len(normal_indices)
    n_train = int(0.7 * n_normal)
    n_val = int(0.1 * n_normal)
    train_norm = normal_indices[:n_train]
    val_norm = normal_indices[n_train:n_train+n_val]
    test_norm = normal_indices[n_train+n_val:]

    random.shuffle(abnormal_indices)
    n_abnormal_val = int(0.1 * len(abnormal_indices))
    val_abnormal = abnormal_indices[:n_abnormal_val]
    test_abnormal = abnormal_indices[n_abnormal_val:]

    train_indices = train_norm
    val_indices = val_norm + val_abnormal
    test_indices = test_norm + test_abnormal

    train_dataset = SkeletonDataset(data_root, is_train=True, indices=train_indices)
    val_dataset = SkeletonDataset(data_root, is_train=False, indices=val_indices)
    test_dataset = SkeletonDataset(data_root, is_train=False, indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # 打印信息...
    print("="*50)
    print(f"正常样本总数: {len(normal_indices)}，异常样本总数: {len(abnormal_indices)}")
    print(f"训练集 (仅正常): {len(train_dataset)}")
    print(f"验证集 (正常+异常): {len(val_dataset)} (正常:{len(val_norm)}, 异常:{len(val_abnormal)})")
    print(f"测试集 (正常+异常): {len(test_dataset)} (正常:{len(test_norm)}, 异常:{len(test_abnormal)})")
    print("="*50)

    return train_loader, val_loader, test_loader