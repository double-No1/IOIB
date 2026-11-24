# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageFolderDataset(Dataset):
    def __init__(self, root, is_train=True):
        """
        Args:
            root: 数据集根目录，应包含 class0 和 class1 两个子文件夹
            is_train: 保留参数，用于后续扩展（如数据增强），不影响数据加载
        """
        self.is_train = is_train
        self.samples = []
        self.labels = []

        # 修改点：始终加载 class0 和 class1，无论 is_train 为何值
        # class0 → 标签 0（异常/zuo），class1 → 标签 1（正常/zhan）
        for label, name in enumerate(['class0', 'class1']):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                print(f"⚠ Warning: {folder} does not exist, skipping.")
                continue
            for f in os.listdir(folder):
                # 可添加文件扩展名过滤，这里简化处理
                self.samples.append(os.path.join(folder, f))
                self.labels.append(label)

        # 图像预处理：与 YOLO 输入尺寸一致
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]