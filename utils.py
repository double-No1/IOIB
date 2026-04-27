import argparse
import torch


def load_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--r', type=float, default=0.8, help='Node retention probability')


    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--feat_dim', type=int, default=64, help='Feature dimension')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--patch_size', type=int, default=32, help='Patch size')
    parser.add_argument('--input_dim', type=int, default=3, help='Input embedding dimension')
    parser.add_argument('--out_dim', type=int, default=128, help='Output embedding dimension')
    parser.add_argument('--conv_layers', type=int, default=2, help='Number of GIN layers')


    parser.add_argument('--kl', type=float, default=0.5, help='KL loss weight')
    parser.add_argument('--mi', type=float, default=2.0, help='Mutual information loss weight')
    parser.add_argument('--collapse', type=float, default=1.0, help='Collapse loss weight')
    parser.add_argument('--occ', type=float, default=1.0, help='OCC loss weight')
    parser.add_argument('--gae', type=float, default=0.8, help='GAE loss weight')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    # parser.add_argument('--data_root', type=str, default=r'E:\Pycharm\OCC\data\images', help='Dataset root directory')
    parser.add_argument('--data_root', type=str, default=r'E:/Pycharm/OCC/data/self-dataset', help='Dataset root directory')
    # parser.add_argument('--data_root', type=str, default=r'E:\images2', help='Visualization')
    parser.add_argument('--save_path', type=str, default=r'E:\Pycharm\OCC\visualize', help='Save path')
    # parser.add_argument('--save_path2', type=str, default=r'E:\Pycharm\OCC\epoch_pth', help='Save path')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')

    parser.add_argument('--grid_search', action='store_true', help='Enable grid search mode')


    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    return args
