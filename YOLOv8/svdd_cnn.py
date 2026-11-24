import torch
from torch.utils.data import DataLoader
from dataset import ImageFolderDataset
from svdd import DeepSVDD
from test_svdd_cnn import SimpleCNN, device  # 如果在同一目录

# 配置
data_root = 'E:\Pycharm\OCC\data\images'
batch_size = 16
feat_dim = 256

# 1. 数据集：只用 zhan
full_dataset = ImageFolderDataset(data_root, is_train=True)
zhan_indices = [i for i, (_, label) in enumerate(full_dataset) if label==1]
train_subset = torch.utils.data.Subset(full_dataset, zhan_indices)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=0)

# 2. CNN backbone
cnn = SimpleCNN(feat_dim=feat_dim).to(device)
cnn.eval()

# 3. SVDD
svdd = DeepSVDD(feat_dim).to(device)

# 4. 计算 center
features = []
with torch.no_grad():
    for x, _ in train_loader:
        x = x.to(device)
        f = cnn(x)
        features.append(f)
features = torch.cat(features, dim=0)
svdd.init_center(features)

# 5. 保存权重
torch.save(svdd.state_dict(), './svdd_cnn.pth')
print("✅ SVDD weights saved to svdd_cnn.pth")
