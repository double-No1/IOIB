# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# import os
# from torchvision import datasets, transforms
#
# from timm.data.constants import \
#     IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from timm.data import create_transform
#
# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)
#
#     print("Transform = ")
#     if isinstance(transform, tuple):
#         for trans in transform:
#             print(" - - - - - - - - - - ")
#             for t in trans.transforms:
#                 print(t)
#     else:
#         for t in transform.transforms:
#             print(t)
#     print("---------------------------")
#
#     if args.data_set == 'CIFAR':
#         dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
#         nb_classes = 100
#     elif args.data_set == 'IMNET':
#         print("reading from datapath", args.data_path)
#         root = os.path.join(args.data_path, 'train' if is_train else 'val')
#         dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = 1000
#     elif args.data_set == "image_folder":
#         root = args.data_path if is_train else args.eval_data_path
#         dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = args.nb_classes
#         assert len(dataset.class_to_idx) == nb_classes
#     else:
#         raise NotImplementedError()
#     print("Number of the class = %d" % nb_classes)
#
#     return dataset, nb_classes
#
#
# def build_transform(is_train, args):
#     resize_im = args.input_size > 32
#     imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
#     mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
#
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#             mean=mean,
#             std=std,
#         )
#         if not resize_im:
#             transform.transforms[0] = transforms.RandomCrop(
#                 args.input_size, padding=4)
#         return transform
#
#     t = []
#     if resize_im:
#         # warping (no cropping) when evaluated at 384 or larger
#         if args.input_size >= 384:
#             t.append(
#             transforms.Resize((args.input_size, args.input_size),
#                             interpolation=transforms.InterpolationMode.BICUBIC),
#         )
#             print(f"Warping {args.input_size} size input images...")
#         else:
#             if args.crop_pct is None:
#                 args.crop_pct = 224 / 256
#             size = int(args.input_size / args.crop_pct)
#             t.append(
#                 # to maintain same ratio w.r.t. 224 images
#                 transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
#             )
#             t.append(transforms.CenterCrop(args.input_size))
#
#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(mean, std))
#     return transforms.Compose(t)

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# -------------------------------
# 自定义 Dataset
# -------------------------------
class NTUDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# -------------------------------
# 构建数据集函数
# -------------------------------
def build_dataset(is_train, args):
    """
    构建训练/验证集
    args.data_path: images 文件夹路径
    自动划分 80% train，20% val
    """
    transform = build_transform(is_train, args)

    # 图片路径
    zhan_dir = os.path.join(args.data_path, 'zhan')
    zuo_dir = os.path.join(args.data_path, 'zuo')

    zhan_imgs = [os.path.join(zhan_dir, f) for f in os.listdir(zhan_dir) if f.endswith(('png','jpg','jpeg'))]
    zuo_imgs  = [os.path.join(zuo_dir, f)  for f in os.listdir(zuo_dir) if f.endswith(('png','jpg','jpeg'))]

    all_imgs = zhan_imgs + zuo_imgs
    all_labels = [0]*len(zhan_imgs) + [1]*len(zuo_imgs)

    # 自动划分训练/验证集
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        all_imgs, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    if is_train:
        dataset = NTUDataset(train_imgs, train_labels, transform)
    else:
        dataset = NTUDataset(val_imgs, val_labels, transform)

    nb_classes = 2
    return dataset, nb_classes

# -------------------------------
# 数据增强函数
# -------------------------------
def build_transform(is_train, args):
    """
    数据增强
    """
    resize_im = args.input_size > 32
    if is_train:
        t = [
            transforms.RandomResizedCrop(args.input_size, scale=(0.8,1.0), ratio=(0.75,1.3333)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        # 验证集使用中心裁剪
        size = int(args.input_size / 0.875)
        t = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.input_size)
        ]
    # 公共 transform
    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]
    return transforms.Compose(t)
