# Trainer2.py
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

class GLAD_trainer:
    def __init__(self, model, loss_fun, device='cuda'):
        self.model = model
        self.loss_fun = loss_fun
        self.device = device

    def _train_epoch(self, train_loader, optimizer, cls):
        self.model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(self.device)
            optimizer.zero_grad()
            z, c = self.model(images)
            loss = self.loss_fun(z, c)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        return total_loss / len(train_loader.dataset)

    def train(self, train_loader, test_loader, optimizer, max_epochs=50, cls=1, log_every=1):
        best_test_auc = 0
        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, cls)

            # 测试
            test_auc, test_ap, test_f1, test_prec, test_rec, scores, targets = self.evaluate(test_loader, cls)

            if epoch % log_every == 0:
                print(f"Epoch {epoch}: TR loss: {train_loss:.4f}, "
                      f"AUC: {test_auc:.4f}, AP: {test_ap:.4f}, "
                      f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")

        return train_loss, None, test_auc, test_ap, test_f1, scores, targets

    @torch.no_grad()
    def evaluate(self, loader, cls=1):
        self.model.eval()
        all_scores = []
        all_targets = []
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            z, c = self.model(images)
            dist = torch.norm(z - c, dim=1)
            all_scores.append(dist.cpu())
            all_targets.append(labels.cpu())

        all_scores = torch.cat(all_scores).numpy()
        all_targets = torch.cat(all_targets).numpy()

        test_auc = roc_auc_score(all_targets, all_scores)
        test_ap = average_precision_score(all_targets, all_scores)
        preds = (all_scores > all_scores.mean()).astype(int)
        test_f1 = f1_score(all_targets, preds)
        test_prec = precision_score(all_targets, preds)
        test_rec = recall_score(all_targets, preds)

        return test_auc, test_ap, test_f1, test_prec, test_rec, all_scores, all_targets

# ==============================
# Loss function
# ==============================
class OCC_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, c):
        return torch.mean(torch.norm(z - c, dim=1))
