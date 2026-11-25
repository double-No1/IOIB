import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import random

class NTUDataset(Dataset):
    def __init__(self, data_root, img_size=224, is_train=True, indices=None):
        self.img_size = img_size
        self.image_paths = []
        self.labels = []

        # 遍历 class0（异常）和 class1（正常）
        for label, folder in enumerate(['class0', 'class1']):
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
                continue
            for img_name in sorted(os.listdir(folder_path)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(label)

        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        # 数据增强：训练集使用强增强，测试集仅基础变换
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label


def get_dataloaders(args):
    """返回 train_loader, val_loader, test_loader"""
    data_root = args.data_root
    seed = getattr(args, 'seed', 42)
    random.seed(seed)

    # 构建全局索引
    all_paths, all_labels = [], []
    for label, folder in enumerate(['class0', 'class1']):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for img_name in sorted(os.listdir(folder_path)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_paths.append(os.path.join(folder_path, img_name))
                all_labels.append(label)

    normal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]   # class1 正常
    abnormal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0] # class0 异常

    # 将正常样本划分为 70% 训练 + 10% 验证 + 20% 测试（可根据需要调整）
    random.shuffle(normal_indices)
    n_normal = len(normal_indices)
    n_train = int(0.7 * n_normal)
    n_val = int(0.1 * n_normal)
    train_norm = normal_indices[:n_train]
    val_norm = normal_indices[n_train:n_train+n_val]
    test_norm = normal_indices[n_train+n_val:]

    # 训练集：仅正常样本
    train_indices = train_norm
    # 验证集：正常样本 (val_norm) + 部分异常样本（可选，这里为了阈值选择，可以只用正常或混合，但一般验证集应包含两类）
    # 此处为了选择最佳阈值，验证集包含正常和异常（使用一部分异常）
    random.shuffle(abnormal_indices)
    n_abnormal_val = int(0.1 * len(abnormal_indices))   # 取10%异常作为验证
    val_abnormal = abnormal_indices[:n_abnormal_val]
    val_indices = val_norm + val_abnormal
    # 测试集：剩余正常 + 剩余异常
    test_abnormal = abnormal_indices[n_abnormal_val:]
    test_indices = test_norm + test_abnormal

    train_dataset = NTUDataset(data_root, args.img_size, is_train=True, indices=train_indices)
    val_dataset = NTUDataset(data_root, args.img_size, is_train=False, indices=val_indices)
    test_dataset = NTUDataset(data_root, args.img_size, is_train=False, indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print("="*50)
    print(f"正常样本总数: {len(normal_indices)}，异常样本总数: {len(abnormal_indices)}")
    print(f"训练集 (仅正常): {len(train_dataset)}")
    print(f"验证集 (正常+异常): {len(val_dataset)} (正常:{len(val_norm)}, 异常:{len(val_abnormal)})")
    print(f"测试集 (正常+异常): {len(test_dataset)} (正常:{len(test_norm)}, 异常:{len(test_abnormal)})")
    print("="*50)
    return train_loader, val_loader, test_loader