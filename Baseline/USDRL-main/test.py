# import os
# import shutil
# import random
#
# # 你原始 pt 文件夹路径
# original_folder = r"E:\Pycharm\X\data\graph_data"  # 修改成你的路径
#
# # 新建 train 和 val 文件夹
# train_folder = r"E:\Pycharm\Baseline\USDRL-main\data\data_train"
# val_folder = r"E:\Pycharm\Baseline\USDRL-main\data\data_val"
#
# os.makedirs(train_folder, exist_ok=True)
# os.makedirs(val_folder, exist_ok=True)
#
# # 获取所有 pt 文件
# all_files = [f for f in os.listdir(original_folder) if f.endswith('.pt')]
#
# # 打乱顺序
# random.shuffle(all_files)
#
# # 80% 训练，20% 验证
# num_train = int(0.8 * len(all_files))
# train_files = all_files[:num_train]
# val_files = all_files[num_train:]
#
# # 拷贝文件
# for f in train_files:
#     shutil.copy(os.path.join(original_folder, f), os.path.join(train_folder, f))
#
# for f in val_files:
#     shutil.copy(os.path.join(original_folder, f), os.path.join(val_folder, f))
#
# print(f"总文件数: {len(all_files)}")
# print(f"训练集: {len(train_files)} 文件 -> {train_folder}")
# print(f"验证集: {len(val_files)} 文件 -> {val_folder}")


import os
import torch
import numpy as np

pt_dir = r"E:\Pycharm\Baseline\USDRL-main\data\train"
out_dir = r"E:\Pycharm\Baseline\USDRL-main\data\usdrldata"

os.makedirs(out_dir, exist_ok=True)

V = 25   # NTU skeleton joints
C = 3

all_data = []
num_frames = []

for fname in os.listdir(pt_dir):
    if not fname.endswith(".pt"):
        continue

    data = torch.load(os.path.join(pt_dir, fname))

    x = data.x  # [T*V, C]
    T = x.shape[0] // V
    #
    # x = x.view(T, V, C)          # (T, V, C)
    T_fake = 64  # 必须 >= 50，和 input_size 对齐

    x = x.view(1, V, C)  # (1, V, C)
    x = x.repeat(T_fake, 1, 1)  # (T, V, C)

    x = x.permute(2, 0, 1)       # (C, T, V)
    x = x.unsqueeze(-1)          # (C, T, V, 1)

    all_data.append(x.numpy())
    num_frames.append(T)

all_data = np.stack(all_data)         # (N, C, T, V, 1)
num_frames = np.array(num_frames)

np.save(os.path.join(out_dir, "train_data_joint.npy"), all_data)
np.save(os.path.join(out_dir, "train_num_frame.npy"), num_frames)

print("Saved:", all_data.shape)
