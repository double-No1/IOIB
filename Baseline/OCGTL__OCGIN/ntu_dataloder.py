import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random
from torch_geometric.data import Batch


class NTUSkeletonGraphDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f)
                      for f in os.listdir(root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_path = self.files[idx]
        graph = torch.load(data_path)

        # 🧩 自动判断标签
        fname = os.path.basename(data_path)
        # label = torch.tensor(0 if 'A008' in fname else 1, dtype=torch.long)
        label = torch.tensor(1 if 'A008' in fname else 0, dtype=torch.long)


        # ✅ 直接在 graph.y 中写入标签
        graph.y = label.clone()
        return graph



# def collate_fn(batch):
#     graphs = Batch.from_data_list(batch)
#     labels = torch.tensor([g.y.item() for g in batch], dtype=torch.long)
#     return graphs, labels
# def collate_fn(batch):
#     graphs = Batch.from_data_list(batch)
#     graphs.__num_graphs__ = graphs.num_graphs  # 临时兼容旧代码
#     labels = torch.tensor([g.y.item() for g in batch], dtype=torch.long)
#     return graphs
def collate_fn(batch):
    graphs = Batch.from_data_list(batch)

    return graphs

# ==============================
# ✅ 划分函数（不变）
# ==============================
def split_dataset(dataset, train_ratio=0.4, seed=42):
    random.seed(seed)

    idx_label1 = [i for i, f in enumerate(dataset.files) if 'A009' in f]
    idx_label0 = [i for i, f in enumerate(dataset.files) if 'A008' in f]

    n_train_label1 = int(len(idx_label1) * train_ratio)
    train_label1 = random.sample(idx_label1, n_train_label1)

    test_indices = list(set(idx_label1 + idx_label0) - set(train_label1))

    train_dataset = Subset(dataset, train_label1)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset



# ==============================
# ✅ 主程序
# ==============================
if __name__ == "__main__":
    data_dir = r"E:\Pycharm\X\data\graph_data"
    dataset = NTUSkeletonGraphDataset(data_dir)
    print(f"✅ 数据集大小: {len(dataset)}")

    train_dataset, test_dataset = split_dataset(dataset)
    print(f"训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 🔍 查看几个样本
    for batch in train_loader:
        print("\n=== 训练批次 ===")
        print("批次大小:", len(batch))
        print("第一个样本:", batch)
        print("第一个样本:", batch[0].x)
        print("标签:", batch[0].y)
        break

    for batch in test_loader:
        print("\n=== 测试批次 ===")
        print("批次大小:", len(batch))
        print("第一个样本:", batch[0])
        print("标签:", batch[0].y)
        break

