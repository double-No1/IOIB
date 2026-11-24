# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageFolderDataset(Dataset):
    def __init__(self, root, is_train=True):
        self.is_train = is_train
        self.samples = []
        self.labels = []

        if is_train:
            folder = os.path.join(root, 'class0')
            for f in os.listdir(folder):
                self.samples.append(os.path.join(folder, f))
                self.labels.append(0)  # 正常
        else:
            for label, name in enumerate(['class0', 'class1']):
                folder = os.path.join(root, name)
                for f in os.listdir(folder):
                    self.samples.append(os.path.join(folder, f))
                    self.labels.append(label)

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
