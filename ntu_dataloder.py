import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import load_args
import matplotlib.pyplot as plt
import random


class NTUDataset(Dataset):
    """动作数据集：zhan(1)为正常，zuo(0)为异常
    支持通过 indices 参数只加载部分样本（传入全局索引列表）。
    """
    def __init__(self, data_root, img_size=256, is_train=True, indices=None):
        """
        Args:
            data_root: 数据集根目录（如'E:\\Pycharm\\OCC\\data\\images'）
            img_size: 图像resize后的尺寸
            is_train: 是否为训练集（训练集用增强，测试集仅resize+归一化）
            indices: 可选的样本索引列表，仅加载这些索引对应的样本（基于按文件遍历的顺序）
        """
        self.img_size = img_size
        self.image_paths = []
        self.labels = []  # 1: zhan（正常）, 0: zuo（异常）

        # 定义数据增强（训练集用，测试集仅resize+归一化）
        self.transform = self._get_transform(is_train)

        # 遍历文件夹，按固定顺序收集图片路径和标签（先zuo再zhan，与你原始实现一致）
        for label, folder in enumerate(['zuo', 'zhan']):  # zuo→0, zhan→1
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
                # 如果文件夹不存在，跳过（使用者应确保路径正确）
                continue
            # 按文件名排序保证顺序可复现
            for img_name in sorted(os.listdir(folder_path)):
                if img_name.endswith('.png') or img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    img_path = os.path.join(folder_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(1 if folder == 'zhan' else 0)

        # 如果提供了 indices，则只保留这些索引的样本
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def _get_transform(self, is_train):
        """定义图像变换（训练集增强，测试集仅标准化）"""
        if is_train:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                                     std=[0.229, 0.224, 0.225])   # ImageNet标准差
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """加载单张图片和标签"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label


def get_dataloaders(args):
    """
    创建 train_loader 与 test_loader：
      - 训练集：zhan 的 70%（仅正样本）
      - 测试集：剩余 30% 的 zhan + 所有 zuo（正+负）
    要求 args 至少包含: data_root, img_size, batch_size, num_workers
    可选: args.seed
    """
    data_root = args.data_root
    seed = getattr(args, "seed", 42)

    # 首先构建完整的索引列表（与 NTUDataset 中构建顺序一致）
    all_image_paths = []
    all_labels = []
    for label, folder in enumerate(['zuo', 'zhan']):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for img_name in sorted(os.listdir(folder_path)):
            if img_name.endswith('.png') or img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                img_path = os.path.join(folder_path, img_name)
                all_image_paths.append(img_path)
                all_labels.append(1 if folder == 'zhan' else 0)

    # 收集所有 zhan 和 zuo 的全局索引
    zhan_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]
    zuo_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]

    # 随机但可复现地划分 zhan 为 70% 训练 / 30% 测试
    random.seed(seed)
    zhan_indices_shuffled = zhan_indices.copy()
    random.shuffle(zhan_indices_shuffled)

    n_zhan = len(zhan_indices_shuffled)
    n_train_zhan = int(round(0.7 * n_zhan))
    train_zhan_indices = zhan_indices_shuffled[:n_train_zhan]
    val_zhan_indices = zhan_indices_shuffled[n_train_zhan:]  # 剩下30%

    # 训练集索引：仅 train_zhan_indices（注意：传入的是全局索引）
    # 测试集索引：val_zhan_indices + 所有 zuo_indices
    test_indices = val_zhan_indices + zuo_indices

    # 创建 Dataset（注意：NTUDataset 接受 indices，用于只加载子集）
    train_dataset = NTUDataset(
        data_root=data_root,
        img_size=args.img_size,
        is_train=True,
        indices=train_zhan_indices
    )
    test_dataset = NTUDataset(
        data_root=data_root,
        img_size=args.img_size,
        is_train=False,
        indices=test_indices
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 打印划分信息，方便调试（可选）
    print("=" * 50)
    print(f"数据根目录: {data_root}")
    print(f"zhan 总数量: {len(zhan_indices)}，zuo 总数量: {len(zuo_indices)}")
    print(f"训练集 zhan 数量(70%): {len(train_zhan_indices)}")
    print(f"测试集 zhan 数量(30%): {len(val_zhan_indices)}, 测试集 zuo 数量: {len(zuo_indices)}")
    print(f"训练样本总数: {len(train_dataset)}, 测试样本总数: {len(test_dataset)}")
    print("=" * 50)

    return train_loader, test_loader


# 反归一化函数（用于可视化）
def denormalize(tensor):
    """将归一化的tensor转回原图范围"""
    # 如果传入的是单张tensor (C,H,W) 或 batch (B,C,H,W)，都尝试处理
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        added = True
    else:
        added = False

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    # 将 mean/std 移动到与输入相同的 device
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)

    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)  # 限制范围在0-1

    if added:
        tensor = tensor.squeeze(0)
    return tensor
