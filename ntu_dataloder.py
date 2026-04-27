import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import load_args
import matplotlib.pyplot as plt
import random


class NTUDataset(Dataset):

    def __init__(self, data_root, img_size=256, is_train=True, indices=None):
   
        self.img_size = img_size
        self.image_paths = []
        self.labels = [] 

        self.transform = self._get_transform(is_train)

        for label, folder in enumerate(['zuo', 'zhan']):  # zuo→0, zhan→1
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
           
                continue

            for img_name in sorted(os.listdir(folder_path)):
                if img_name.endswith('.png') or img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    img_path = os.path.join(folder_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(1 if folder == 'zhan' else 0)


        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def _get_transform(self, is_train):

        if is_train:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                     std=[0.229, 0.224, 0.225]) 
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

        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label


def get_dataloaders(args):

    data_root = args.data_root
    seed = getattr(args, "seed", 42)

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


    zhan_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]
    zuo_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]


    random.seed(seed)
    zhan_indices_shuffled = zhan_indices.copy()
    random.shuffle(zhan_indices_shuffled)

    n_zhan = len(zhan_indices_shuffled)
    n_train_zhan = int(round(0.7 * n_zhan))
    train_zhan_indices = zhan_indices_shuffled[:n_train_zhan]
    val_zhan_indices = zhan_indices_shuffled[n_train_zhan:]  # 剩下30%


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

    print("=" * 50)
    print(f"数据根目录: {data_root}")
    print(f"zhan 总数量: {len(zhan_indices)}，zuo 总数量: {len(zuo_indices)}")
    print(f"训练集 zhan 数量(70%): {len(train_zhan_indices)}")
    print(f"测试集 zhan 数量(30%): {len(val_zhan_indices)}, 测试集 zuo 数量: {len(zuo_indices)}")
    print(f"训练样本总数: {len(train_dataset)}, 测试样本总数: {len(test_dataset)}")
    print("=" * 50)

    return train_loader, test_loader



def denormalize(tensor):


    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        added = True
    else:
        added = False

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    mean = mean.to(tensor.device)
    std = std.to(tensor.device)

    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1) 

    if added:
        tensor = tensor.squeeze(0)
    return tensor
