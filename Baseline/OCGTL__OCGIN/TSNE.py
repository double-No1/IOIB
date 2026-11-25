import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.nn import GINConv

# ==============================
# 参数
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
MAX_EPOCHS = 200
LEARNING_RATE = 1e-3
DATA_DIR = r"E:\Pycharm\data\self_sk"
NUM_RUNS = 5

K = 5
EMB_DIM = 256
NUM_LAYERS = 4

# ==============================
# 数据集
# ==============================
class SkeletonDataset(Dataset):
    def __init__(self, root, indices=None):
        self.files, self.labels = [], []
        for label, folder in enumerate(['class0', 'class1']):
            folder_path = os.path.join(root, folder)
            for f in os.listdir(folder_path):
                if f.endswith('.npy'):
                    self.files.append(os.path.join(folder_path, f))
                    self.labels.append(label)

        if indices is not None:
            self.files = [self.files[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        if data.ndim == 3:
            data = data.mean(axis=0)
        x = torch.tensor(data.reshape(-1), dtype=torch.float32)
        return x, self.labels[idx]

# ==============================
# 划分
# ==============================
def split_dataset(root, seed=42):
    random.seed(seed)
    labels = []

    for label, folder in enumerate(['class0','class1']):
        for f in os.listdir(os.path.join(root, folder)):
            if f.endswith('.npy'):
                labels.append(label)

    normal = [i for i,l in enumerate(labels) if l==1]
    abnormal = [i for i,l in enumerate(labels) if l==0]

    random.shuffle(normal)
    random.shuffle(abnormal)

    train = normal[:int(0.4*len(normal))]
    test = normal[int(0.4*len(normal)):] + abnormal

    return train, test

# ==============================
# 模型
# ==============================
class OCGIN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.K = K
        self.convs = nn.ModuleList()

        for i in range(NUM_LAYERS):
            in_dim = dim if i==0 else EMB_DIM
            mlp = nn.Sequential(
                nn.Linear(in_dim, EMB_DIM),
                nn.ReLU(),
                nn.Linear(EMB_DIM, EMB_DIM)
            )
            self.convs.append(GINConv(mlp))

        self.c = nn.Parameter(torch.randn(1, EMB_DIM))

    def knn(self,x):
        x = F.normalize(x, dim=1)
        sim = x @ x.t()
        idx = sim.topk(k=min(self.K+1, x.size(0)), dim=1)[1][:,1:]
        row = torch.arange(x.size(0), device=x.device).repeat_interleave(idx.size(1))
        col = idx.reshape(-1)
        return torch.stack([row,col])

    def forward(self,x):
        edge = self.knn(x)
        for conv in self.convs:
            x = F.relu(conv(x,edge))

        x = F.normalize(x, dim=1)  # ⭐关键
        c = self.c.expand(x.size(0), -1)
        return x,c

# ==============================
# TSNE（200/200）
# ==============================
def run_tsne(model, loader, save_path):
    model.eval()
    feats, labels = [], []
    safe_cnt, unsafe_cnt = 0, 0

    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            z,_ = model(x)
            z = z.cpu().numpy()
            y = y.numpy()

            for f,t in zip(z,y):
                if t==1 and safe_cnt<200:
                    feats.append(f); labels.append(1); safe_cnt+=1
                elif t==0 and unsafe_cnt<200:
                    feats.append(f); labels.append(0); unsafe_cnt+=1

                if safe_cnt>=200 and unsafe_cnt>=200:
                    break

            if safe_cnt>=200 and unsafe_cnt>=200:
                break

    feats = np.array(feats)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, perplexity=20, n_iter=2000, init='pca')
    emb = tsne.fit_transform(feats)

    plt.figure(figsize=(6,6))
    plt.scatter(emb[labels==1,0], emb[labels==1,1],
                c='blue', label='safe', s=15, alpha=0.7)
    plt.scatter(emb[labels==0,0], emb[labels==0,1],
                c='red', label='unsafe', s=15, alpha=0.7)

    plt.legend()
    plt.title("OCGIN", fontsize=14)
    plt.xticks([]); plt.yticks([])

    os.makedirs("tsne_results", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[TSNE] saved: {save_path}")

# ==============================
# 训练
# ==============================
def train_epoch(model, loader, opt):
    model.train()
    total = 0
    for x,_ in loader:
        x = x.to(DEVICE)
        opt.zero_grad()
        z,c = model(x)
        loss = torch.mean(torch.norm(z-c, dim=1))
        loss.backward()
        opt.step()
        total += loss.item()
    return total/len(loader)

# ==============================
# 主循环
# ==============================
for run in range(NUM_RUNS):
    print(f"\n===== Run {run+1} =====")

    train_idx, test_idx = split_dataset(DATA_DIR, seed=42+run)

    train_ds = SkeletonDataset(DATA_DIR, train_idx)
    test_ds = SkeletonDataset(DATA_DIR, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    feat_dim = train_ds[0][0].shape[0]
    model = OCGIN(feat_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):
        loss = train_epoch(model, train_loader, optimizer)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss {loss:.4f}")

    # ===== TSNE =====
    run_tsne(model, test_loader, f"tsne_results/run_{run+1}.png")