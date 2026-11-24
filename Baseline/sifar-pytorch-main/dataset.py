import torch
import os
import numpy as np

class NTUImageStackDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_segments=8, transform=None):
        self.root_dir = root_dir
        self.num_segments = num_segments
        self.transform = transform

        self.samples = []  # [(sample_path, label)]

        for label_name, label in [('zhan', 1), ('zuo', 0)]:
            cls_dir = os.path.join(root_dir, label_name)
            for sample in os.listdir(cls_dir):
                self.samples.append(
                    (os.path.join(cls_dir, sample), label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir, label = self.samples[idx]

        img_list = sorted(os.listdir(sample_dir))
        img_list = self._sample_frames(img_list)

        imgs = []
        for img_name in img_list:
            img = Image.open(os.path.join(sample_dir, img_name)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        # ★ SIFAR 核心：channel 拼接
        stacked = torch.cat(imgs, dim=0)  # [3K, H, W]

        return stacked, label

    def _sample_frames(self, img_list):
        if len(img_list) <= self.num_segments:
            return img_list
        idxs = np.linspace(0, len(img_list) - 1, self.num_segments).astype(int)
        return [img_list[i] for i in idxs]
