import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score
)
import logging
import numpy as np

logger = logging.getLogger(__name__)


def sce_loss(x, y, alpha=2.0):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1.0 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class GINBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.PReLU()
        )
        self.conv = GINConv(mlp, train_eps=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class SessionGraphMAE(pl.LightningModule):
    def __init__(self,
                 in_dim=128,
                 hidden_dim=128,
                 num_attack_families=6,
                 mask_rate=0.5,
                 lr=0.0001,
                 enc_layers=4,
                 dec_layers=4,
                 id_to_label_map=None):
        super().__init__()
        self.save_hyperparameters(ignore=['id_to_label_map'])
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.mask_rate = mask_rate
        self.lr = lr
        self.print_final_report = False
        self.id_to_label_map = id_to_label_map or {}

        self.mask_token = nn.Parameter(torch.zeros(1, self.in_dim))
        nn.init.xavier_uniform_(self.mask_token)
        self.dmask_token = nn.Parameter(torch.zeros(1, self.hidden_dim))
        nn.init.xavier_uniform_(self.dmask_token)

        self.encoder = nn.ModuleList()
        self.encoder.append(GINBlock(self.in_dim, self.hidden_dim))
        for _ in range(enc_layers - 1):
            self.encoder.append(GINBlock(self.hidden_dim, self.hidden_dim))

        self.decoder = nn.ModuleList()
        for _ in range(dec_layers - 1):
            self.decoder.append(GINBlock(self.hidden_dim, self.hidden_dim))
        self.decoder.append(GINBlock(self.hidden_dim, self.in_dim))

        # 🎯 终极核武器：维度变为 hidden_dim * 3 + 1 (最后1维是显式的对数节点数)
        clf_in_dim = self.hidden_dim * 3 + 1

        self.is_malicious_clf = nn.Sequential(
            nn.Linear(clf_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self.attack_family_clf = nn.Sequential(
            nn.Linear(clf_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_attack_families)
        )

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def mask_nodes(self, x):
        num_nodes = x.size(0)
        num_mask = int(self.mask_rate * num_nodes)
        perm = torch.randperm(num_nodes, device=x.device)
        mask_idx = perm[:num_mask]
        x_masked = x.clone()
        x_masked[mask_idx] = self.mask_token
        return x_masked, mask_idx

    def forward(self, x, edge_index, batch):
        z = x
        for conv in self.encoder:
            z = conv(z, edge_index)

        emb_mean = global_mean_pool(z, batch)
        emb_max = global_max_pool(z, batch)
        emb_add = global_add_pool(z, batch)

        # 🎯 显式统计图的节点数，取对数平滑 (防止极大值冲坏 MLP)
        num_nodes = torch.bincount(batch).float().unsqueeze(-1)
        log_num_nodes = torch.log1p(num_nodes)  # log(1 + x)

        graph_emb = torch.cat([emb_mean, emb_max, emb_add, log_num_nodes], dim=-1)

        return self.is_malicious_clf(graph_emb), self.attack_family_clf(graph_emb)

    def training_step(self, batch, batch_idx):
        x_orig, edge_index, graph_batch = batch.x, batch.edge_index, batch.batch

        x_masked, mask_idx = self.mask_nodes(x_orig)
        z = x_masked
        for conv in self.encoder:
            z = conv(z, edge_index)

        z_masked = z.clone()
        z_masked[mask_idx] = self.dmask_token

        h_recon = z_masked
        for conv in self.decoder:
            h_recon = conv(h_recon, edge_index)

        loss_recon = sce_loss(h_recon[mask_idx], x_orig[mask_idx], alpha=2.0)

        emb_mean = global_mean_pool(z, graph_batch)
        emb_max = global_max_pool(z, graph_batch)
        emb_add = global_add_pool(z, graph_batch)

        # 🎯 显式注入对数图大小
        num_nodes = torch.bincount(graph_batch).float().unsqueeze(-1)
        log_num_nodes = torch.log1p(num_nodes)
        graph_emb = torch.cat([emb_mean, emb_max, emb_add, log_num_nodes], dim=-1)

        logits_bin = self.is_malicious_clf(graph_emb)
        logits_multi = self.attack_family_clf(graph_emb)

        y_bin_target = batch.y_bin.squeeze(-1).float()
        num_pos_bin = y_bin_target.sum()
        num_neg_bin = y_bin_target.size(0) - num_pos_bin
        pos_weight_bin = torch.clamp((num_neg_bin + 1.0) / (num_pos_bin + 1.0), min=1.0, max=10.0)

        loss_bin = F.binary_cross_entropy_with_logits(
            logits_bin.squeeze(-1),
            y_bin_target,
            pos_weight=pos_weight_bin
        )

        num_classes = logits_multi.size(-1)
        y_m = batch.y_multi.view(-1)
        valid_mask = (y_m >= 0) & (y_m < num_classes)
        loss_multi = torch.tensor(0.0, device=self.device)

        if valid_mask.sum() > 0:
            valid_targets = y_m[valid_mask].long()
            valid_logits = logits_multi[valid_mask]
            one_hot_targets = F.one_hot(valid_targets, num_classes=num_classes).float()

            pos_counts = one_hot_targets.sum(dim=0)
            neg_counts = valid_targets.size(0) - pos_counts
            pos_weight_multi = torch.clamp((neg_counts + 1.0) / (pos_counts + 1.0), min=1.0, max=50.0)

            loss_multi_matrix = F.binary_cross_entropy_with_logits(
                valid_logits, one_hot_targets, reduction='none', pos_weight=pos_weight_multi
            )
            loss_multi = loss_multi_matrix.mean()

        total_loss = loss_recon + (loss_bin * 5.0) + (loss_multi * 20.0)

        self.training_step_outputs.append(total_loss.detach())
        self.log('train_loss', total_loss, batch_size=batch.num_graphs)
        return total_loss

    def on_train_epoch_end(self):
        if self.training_step_outputs:
            avg_loss = torch.stack(self.training_step_outputs).mean()
            logger.info(f"🔄 [Epoch {self.current_epoch:02d}] 训练结束 | 平均训练损失: {avg_loss:.4f}")
            self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        logits_bin, logits_multi = self(batch.x, batch.edge_index, batch.batch)
        prob_bin = torch.sigmoid(logits_bin).squeeze(-1)
        prob_multi = torch.sigmoid(logits_multi)

        self.validation_step_outputs.append({
            "prob_bin": prob_bin.detach(),
            "targets_bin": batch.y_bin.squeeze(-1).detach(),
            "prob_multi": prob_multi.detach(),
            "targets_multi": batch.y_multi.detach()
        })

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.validation_step_outputs.clear()
            return

        prob_bin = torch.cat([x["prob_bin"] for x in self.validation_step_outputs], dim=0).cpu().numpy()
        targets_bin = torch.cat([x["targets_bin"] for x in self.validation_step_outputs], dim=0).cpu().numpy()
        prob_multi = torch.cat([x["prob_multi"] for x in self.validation_step_outputs], dim=0).cpu().numpy()
        targets_multi = torch.cat([x["targets_multi"] for x in self.validation_step_outputs], dim=0).cpu().numpy()

        preds_bin = (prob_bin > 0.5).astype(int)
        preds_multi = np.argmax(prob_multi, axis=-1)

        val_acc_bin = accuracy_score(targets_bin, preds_bin)
        val_prec_bin = precision_score(targets_bin, preds_bin, zero_division=0)
        val_rec_bin = recall_score(targets_bin, preds_bin, zero_division=0)
        val_f1_bin = f1_score(targets_bin, preds_bin, zero_division=0)

        malicious_mask = (targets_multi > 0)
        val_f1_family, val_micro_f1_family = 0.0, 0.0

        if malicious_mask.sum() > 0:
            targets_mal = targets_multi[malicious_mask]
            preds_mal = preds_multi[malicious_mask]

            def get_family_name(tid):
                if tid == 0: return "Benign (Misclassified)"
                name = self.id_to_label_map.get(tid) or self.id_to_label_map.get(str(tid)) or f"Class_{tid}"
                name_l = name.lower()
                if "ddos" in name_l: return "DDoS"
                if "dos" in name_l: return "DoS"
                if "recon" in name_l: return "Recon"
                return "Other"

            targets_fam_names = [get_family_name(t) for t in targets_mal]
            preds_fam_names = [get_family_name(p) for p in preds_mal]

            actual_fams = sorted(list(set(targets_fam_names)))
            val_f1_family = f1_score(targets_fam_names, preds_fam_names, labels=actual_fams, average="macro",
                                     zero_division=0)
            val_micro_f1_family = f1_score(targets_fam_names, preds_fam_names, labels=actual_fams, average="micro",
                                           zero_division=0)

        if not getattr(self, "print_final_report", False):
            logger.info(
                f"✨ [Epoch {self.current_epoch:02d}] 验证 | "
                f"二分类 -> Acc: {val_acc_bin:.4f}, Pre: {val_prec_bin:.4f}, Rec: {val_rec_bin:.4f}, F1: {val_f1_bin:.4f} | "
                f"【Family】Macro: {val_f1_family:.4f}, Micro: {val_micro_f1_family:.4f}"
            )
            self.validation_step_outputs.clear()
            return

        logger.info("============================================================")
        logger.info("🤖 图级别(GraphMAE) is_malicious 任务测试报告")
        logger.info(
            f"    准确率: {val_acc_bin:.4f} | 精确率: {val_prec_bin:.4f} | 召回率: {val_rec_bin:.4f} | F1分数: {val_f1_bin:.4f}")
        logger.info(f"🎯 混淆矩阵:\n{confusion_matrix(targets_bin, preds_bin, labels=[0, 1])}")

        if malicious_mask.sum() > 0:
            logger.info("============================================================")
            logger.info("🤖 attack_family 任务测试报告 (严格对齐 Flow-BERT，仅评估恶意样本)")
            logger.info(f"macro_f1={val_f1_family:.4f}, micro_f1={val_micro_f1_family:.4f}")
            actual_fams = sorted(list(set(targets_fam_names)))
            report_fam = classification_report(targets_fam_names, preds_fam_names, labels=actual_fams, digits=4,
                                               zero_division=0)
            for line in report_fam.split('\n'):
                if line.strip(): logger.info(line)

            logger.info("============================================================")
            logger.info("🤖 attack_type 任务测试报告 (严格对齐 Flow-BERT，仅评估恶意样本)")

            def get_type_name(tid):
                if tid == 0: return "Benign (Misclassified)"
                return self.id_to_label_map.get(tid) or self.id_to_label_map.get(str(tid)) or f"Class_{tid}"

            targets_type_names = [get_type_name(t) for t in targets_mal]
            preds_type_names = [get_type_name(p) for p in preds_mal]
            actual_types = sorted(list(set(targets_type_names)))
            val_f1_type = f1_score(targets_type_names, preds_type_names, labels=actual_types, average="macro",
                                   zero_division=0)
            val_micro_f1_type = f1_score(targets_type_names, preds_type_names, labels=actual_types, average="micro",
                                         zero_division=0)
            logger.info(f"macro_f1={val_f1_type:.4f}, micro_f1={val_micro_f1_type:.4f}")
            report_type = classification_report(targets_type_names, preds_type_names, labels=actual_types, digits=4,
                                                zero_division=0)
            for line in report_type.split('\n'):
                if line.strip(): logger.info(line)

        logger.info("============================================================")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)