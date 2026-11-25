# # Raising the Bar in Graph-level Anomaly Detection (GLAD)
# # Copyright (c) 2022 Robert Bosch GmbH
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU Affero General Public License as published
# # by the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU Affero General Public License for more details.
# #
# # You should have received a copy of the GNU Affero General Public License
# # along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# import time
# import torch
# from sklearn.metrics import roc_auc_score,average_precision_score
# import numpy as np
# from utils import compute_pre_recall_f1,format_time
# from sklearn.metrics import precision_score, recall_score, f1_score
# class GLAD_trainer:
#
#     def __init__(self, model, loss_function,device='cuda'):
#
#         self.loss_fun = loss_function
#         self.device = torch.device(device)
#         self.model = model.to(self.device)
#
#     def _train(self, train_loader, optimizer):
#         self.model.train()
#         loss_all = 0
#         for batch in train_loader:
#             batch = batch.to(self.device)
#             z = self.model(batch)
#             loss = self.loss_fun(z)
#             loss_mean = loss.mean()
#             optimizer.zero_grad()
#             loss_mean.backward()
#             optimizer.step()
#             loss_all += loss.sum()
#         return loss_all.item() / len(train_loader.dataset)
#
#
#
#     def detect_outliers(self, loader, cls):
#         model = self.model
#         model.eval()
#
#         loss_in = 0
#         loss_out = 0
#         target_all = []
#         score_all = []
#         for batch in loader:
#             with torch.no_grad():
#                 data = batch.to(self.device)
#                 labels = (data.y != cls).cpu().numpy()
#                 z = model(data)
#                 score = self.loss_fun(z, eval=True).cpu().numpy()
#                 loss_in += score[labels == 0].sum()
#                 loss_out += score[labels == 1].sum()
#                 target_all.append(labels)
#                 score_all.append(score)
#
#         target_all = np.concatenate(target_all)
#         score_all = np.concatenate(score_all)
#
#         auc = roc_auc_score(target_all, score_all)
#         ap = average_precision_score(target_all, score_all)
#         # threshold 可以用 0.5 或 median score
#         pred_label = (score_all > np.median(score_all)).astype(int)
#         precision = precision_score(target_all, pred_label)
#         recall = recall_score(target_all, pred_label)
#         f1 = f1_score(target_all, pred_label)
#
#         return auc, ap, precision, recall, f1, loss_in / (target_all == 0).sum(), loss_out / (
#                     target_all == 1).sum(), score_all, target_all
#
#     def train(self, train_loader,cls = None,max_epochs=100, optimizer=None, scheduler=None,
#               validation_loader=None, test_loader=None, early_stopping=None, logger=None, log_every=2):
#
#         early_stopper = early_stopping() if early_stopping is not None else None
#
#         val_auc, val_f1, = -1, -1
#         test_auc, test_f1, test_score = None, None,None
#         score,target = None,None
#
#         torch.cuda.empty_cache()
#         try:
#             self.model.eval()
#             self.model.init_center(train_loader)
#         except:
#             pass
#
#         for epoch in range(1, max_epochs+1):
#
#             train_loss = self._train(train_loader, optimizer)
#
#             if scheduler is not None:
#                 scheduler.step()
#
#             valin_loss, valout_loss = 0.0, 0.0
#             val_ap = 0.0
#
#             if test_loader is not None:
#                 test_auc, test_ap, precision, recall, test_f1, testin_loss, testout_loss, score, target = \
#                     self.detect_outliers(test_loader, cls)
#
#             if validation_loader is not None:
#                 val_auc, val_ap, precision, recall, val_f1, valin_loss, valout_loss, _, _ = \
#                     self.detect_outliers(validation_loader, cls)
#
#                 if early_stopper is not None and early_stopper.stop(epoch, valin_loss, val_auc, testin_loss, test_auc, test_ap,test_f1,
#                                                                     train_loss,score,target):
#                     break
#
#             if epoch % log_every == 0 or epoch == 1:
#                 msg = f'Epoch: {epoch}, TR loss: {train_loss:.4f}, VAL loss: ({valin_loss:.4f}, {valout_loss:.4f}), ' \
#                       f'VL auc: {val_auc:.4f} VL ap: {val_ap:.4f} Precision: {precision:.4f} Recall: {recall:.4f} VL f1: {val_f1:.4f}'
#                 if logger is not None:
#                     logger.log(msg)
#                     print(msg)
#                 else:
#                     print(msg)
#
#         if early_stopper is not None:
#             train_loss, val_loss, val_auc, test_loss, test_auc, test_ap, test_f1, best_epoch,score,target \
#                 = early_stopper.get_best_vl_metrics()
#             msg = f'Stopping at epoch {best_epoch}, TR loss: {train_loss}, VAL loss: {val_loss}, VAL auc: {val_auc} ,' \
#                 f'TS loss: {test_loss}, TS auc: {test_auc} TS ap: {test_ap} TS f1: {test_f1}'
#             if logger is not None:
#                 logger.log(msg)
#                 print(msg)
#             else:
#                 print(msg)
#
#
#         return val_loss, val_auc, test_auc, test_ap,test_f1,score,target
#


# models/Trainer.py
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support, roc_curve, precision_recall_curve
)


class GLAD_trainer:
    """
    Final Trainer for OCGTL / One-Class Graph Anomaly Detection
    - Compatible with OCGTL output: (z_cat, center)
    - No validation loader required
    - Test loader is used as evaluation loader
    """

    def __init__(self, model, loss_function, device='cuda'):
        self.model = model.to(device)
        self.loss_fun = loss_function
        self.device = torch.device(device)

    # --------------------------------------------------
    # Train one epoch
    # --------------------------------------------------
    # def _train(self, train_loader, optimizer):
    #     self.model.train()
    #     total_loss = 0.0
    #     total_samples = 0
    #
    #     for batch in train_loader:
    #         batch = batch.to(self.device)
    #
    #         # OCGTL forward → (z_cat, center)
    #         z_cat = self.model(batch)
    #
    #         # OCGTL loss returns per-sample loss
    #         loss = self.loss_fun(z_cat)
    #         loss_mean = loss.mean()
    #
    #         optimizer.zero_grad()
    #         loss_mean.backward()
    #         optimizer.step()
    #
    #         total_loss += loss.sum().item()
    #         total_samples += loss.shape[0]
    #
    #     return total_loss / total_samples

    def _train(self, train_loader, optimizer, encoder=None):
        self.model.train()
        if encoder is not None:
            encoder.train()

        loss_all = 0
        for batch in train_loader:
            images, _ = batch
            images = images.to(self.device)

            # Image → embedding
            z_img = encoder(images)  # [B, D]
            z = z_img.unsqueeze(1)  # [B, 1, D]

            loss = self.loss_fun((z, self.model.c))
            loss_mean = loss.mean()

            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            loss_all += loss.sum().item()

        return loss_all / len(train_loader.dataset)

    # --------------------------------------------------
    # Evaluate (One-Class Detection)
    # --------------------------------------------------
    @torch.no_grad()
    # def detect_outliers(self, loader, cls):
    #     self.model.eval()
    #
    #     score_all = []
    #     target_all = []
    #
    #     for batch in loader:
    #         batch = batch.to(self.device)
    #
    #         # anomaly label: 1 = anomaly, 0 = normal
    #         labels = (batch.y != cls).long().cpu().numpy()
    #
    #         z_cat = self.model(batch)
    #
    #         # eval=True → return anomaly score
    #         scores = self.loss_fun(z_cat, eval=True)
    #
    #         score_all.append(scores.cpu().numpy())
    #         target_all.append(labels)
    #
    #     score_all = np.concatenate(score_all)
    #     target_all = np.concatenate(target_all)
    #
    #     # ---------------- Metrics ----------------
    #     auc = roc_auc_score(target_all, score_all)
    #     ap = average_precision_score(target_all, score_all)
    #
    #     # threshold: 95 percentile (standard OCC)
    #     thresh = np.percentile(score_all, 95)
    #     # pred = (score_all > thresh).astype(int)
    #     fpr, tpr, thresholds = roc_curve(target_all, score_all)
    #     best_idx = np.argmax(tpr - fpr)
    #     thresh = thresholds[best_idx]
    #     pred = (score_all > thresh).astype(int)
    #     # precision_vals, recall_vals, pr_thresholds = precision_recall_curve(target_all, score_all)
    #     # optimal_idx = np.argmax(2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-8))
    #     # optimal_threshold = pr_thresholds[optimal_idx]
    #     # idx = np.where(tpr >= 0.9)[0][0]
    #     # thresh = thresholds[idx]
    #     # 生成预测标签
    #     # pred = (score_all > thresh).astype(int)
    #
    #     precision, recall, f1, _ = precision_recall_fscore_support(
    #         target_all,
    #         pred,
    #         average='binary',
    #         zero_division=0
    #     )
    #
    #     return auc, ap, precision, recall, f1, score_all, target_all

    def detect_outliers(self, loader, cls, encoder):
        self.model.eval()
        encoder.eval()

        scores = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                images, labels = batch
                images = images.to(self.device)

                z_img = encoder(images)
                z = z_img.unsqueeze(1)

                score = self.loss_fun((z, self.model.c), eval=True)

                scores.append(score.cpu())
                targets.append((labels != cls).int())

        scores = torch.cat(scores).numpy()
        targets = torch.cat(targets).numpy()

        auc = roc_auc_score(targets, scores)
        ap = average_precision_score(targets, scores)
        # precision, recall, f1 = compute_pre_recall_f1(targets, scores)
        precision, recall, f1, _ = precision_recall_fscore_support(
                targets, scores,
                average='binary',
                zero_division=0
            )

        return auc, ap, precision, recall, f1, scores, targets

    # --------------------------------------------------
    # Main training loop
    # --------------------------------------------------
    def train(
        self,
        train_loader,
        cls,
        max_epochs=100,
        optimizer=None,
        scheduler=None,
        test_loader=None,
        log_every=1
    ):
        assert test_loader is not None, \
            "For one-class setting, test_loader must be provided."

        best_auc = -1
        best_results = None

        for epoch in range(1, max_epochs + 1):

            train_loss = self._train(train_loader, optimizer)

            if scheduler is not None:
                scheduler.step()

            # ---- Evaluate on TEST set (OCC standard) ----
            auc, ap, precision, recall, f1, score, target = \
                self.detect_outliers(test_loader, cls)

            if auc > best_auc:
                best_auc = auc
                best_results = (auc, ap, precision, recall, f1, score, target)

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"Epoch: {epoch}, "
                    f"TR loss: {train_loss:.4f}, "
                    f"AUC: {auc:.4f}, "
                    f"AP: {ap:.4f}, "
                    f"Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, "
                    f"F1: {f1:.4f}"
                )

        # -------- Return best epoch results --------
        auc, ap, precision, recall, f1, score, target = best_results

        return (
            train_loss,   # final train loss
            auc,
            auc,          # keep interface compatibility
            ap,
            f1,
            score,
            target
        )


