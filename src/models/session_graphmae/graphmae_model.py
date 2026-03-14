import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
import logging
import numpy as np

logger = logging.getLogger(__name__)


# 缩放余弦误差 (Scaled Cosine Error)
def sce_loss(x, y, alpha=2.0):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1.0 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class GINBlock(nn.Module):
    """
    GIN 卷积块：内部使用 MLP + PReLU 进行特征映射
    对应论文表 3.8 的 GIN 和 prelu 要求
    """

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
                 in_dim=768,  # 对齐多视图 Flow BERT 的输出维度
                 hidden_dim=128,  # 编码器输出维度
                 num_attack_families=6,
                 mask_rate=0.5,  # 掩码比例 0.5
                 lr=0.0001,
                 enc_layers=4,  # 编码器层数
                 dec_layers=4):  # 解码器层数
        super().__init__()
        self.save_hyperparameters()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.mask_rate = mask_rate
        self.lr = lr

        self.print_final_report = False

        # [MASK] 标记，用于编码器输入前
        self.mask_token = nn.Parameter(torch.zeros(1, self.in_dim))
        nn.init.xavier_uniform_(self.mask_token)

        # [DMASK] 标记，用于解码器输入前 (重掩码机制)
        self.dmask_token = nn.Parameter(torch.zeros(1, self.hidden_dim))
        nn.init.xavier_uniform_(self.dmask_token)

        # ---------------- 编码器 ----------------
        self.encoder = nn.ModuleList()
        self.encoder.append(GINBlock(self.in_dim, self.hidden_dim))
        for _ in range(enc_layers - 1):
            self.encoder.append(GINBlock(self.hidden_dim, self.hidden_dim))

        # ---------------- 解码器 ----------------
        self.decoder = nn.ModuleList()
        for _ in range(dec_layers - 1):
            self.decoder.append(GINBlock(self.hidden_dim, self.hidden_dim))
        self.decoder.append(GINBlock(self.hidden_dim, self.in_dim))

        # ---------------- 下游分类器 ----------------
        self.is_malicious_clf = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.attack_family_clf = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_attack_families)
        )

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def mask_nodes(self, x):
        """对输入节点特征进行随机 [MASK]"""
        num_nodes = x.size(0)
        num_mask = int(self.mask_rate * num_nodes)
        perm = torch.randperm(num_nodes, device=x.device)
        mask_idx = perm[:num_mask]

        x_masked = x.clone()
        x_masked[mask_idx] = self.mask_token
        return x_masked, mask_idx

    def forward(self, x, edge_index, batch):
        """前向传播：仅提取指纹并分类，不进行掩码"""
        z = x
        for conv in self.encoder:
            z = conv(z, edge_index)

        graph_emb = global_mean_pool(z, batch)
        return self.is_malicious_clf(graph_emb), self.attack_family_clf(graph_emb)

    def training_step(self, batch, batch_idx):
        x_orig, edge_index, graph_batch = batch.x, batch.edge_index, batch.batch

        # 1. 第一阶段掩码 [MASK]
        x_masked, mask_idx = self.mask_nodes(x_orig)

        # 2. 编码器处理
        z = x_masked
        for conv in self.encoder:
            z = conv(z, edge_index)

        # 3. 重掩码机制 [DMASK]：送入解码器前掩蔽潜在特征
        z_masked = z.clone()
        z_masked[mask_idx] = self.dmask_token

        # 4. 解码器处理
        h_recon = z_masked
        for conv in self.decoder:
            h_recon = conv(h_recon, edge_index)

        # 5. 计算损失
        loss_recon = sce_loss(h_recon[mask_idx], x_orig[mask_idx], alpha=2.0)

        # 图级别分类表征 (使用未受 [DMASK] 影响的 z)
        graph_emb = global_mean_pool(z, graph_batch)
        logits_bin = self.is_malicious_clf(graph_emb)
        logits_multi = self.attack_family_clf(graph_emb)

        loss_bin = F.binary_cross_entropy_with_logits(logits_bin.squeeze(-1), batch.y_bin.squeeze(-1).float())

        mask = (batch.y_bin.view(-1) == 1.0)
        loss_multi = 0.0
        if mask.sum() > 0:
            # 确保 batch.y_multi 是标量索引 (long类型)
            loss_multi = F.cross_entropy(logits_multi[mask], batch.y_multi[mask].long())

        total_loss = loss_recon + loss_bin + loss_multi

        self.training_step_outputs.append(total_loss.detach())
        self.log('train_loss', total_loss, batch_size=batch.num_graphs)
        return total_loss

    def on_train_epoch_end(self):
        if self.training_step_outputs:
            avg_loss = torch.stack(self.training_step_outputs).mean()
            logger.info(f"🔄 [Epoch {self.current_epoch:02d}] 训练结束 | 平均训练损失 (Train Loss): {avg_loss:.4f}")
            self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        logits_bin, logits_multi = self(batch.x, batch.edge_index, batch.batch)
        prob_bin = torch.sigmoid(logits_bin).squeeze(-1)
        prob_multi = torch.softmax(logits_multi, dim=-1)

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

        if not getattr(self, "print_final_report", False):
            val_f1_bin = f1_score(targets_bin, preds_bin, zero_division=0)
            val_f1_multi = 0.0
            if len(targets_multi) > 0:
                val_f1_multi = f1_score(targets_multi, preds_multi, average="macro", zero_division=0)

            logger.info(
                f"✨ [Epoch {self.current_epoch:02d}] 验证完成 | 二分类 F1: {val_f1_bin:.4f} | 多分类 Macro-F1: {val_f1_multi:.4f}")
            self.validation_step_outputs.clear()
            return

        # ==========================================================
        # 最终验证报告打印 (Final Report)
        # ==========================================================
        logger.info("============================================================")
        logger.info("🤖 图级别(GraphMAE) is_malicious任务的测试报告")
        logger.info("============================================================")

        acc = accuracy_score(targets_bin, preds_bin)
        prec = precision_score(targets_bin, preds_bin, zero_division=0)
        rec = recall_score(targets_bin, preds_bin, zero_division=0)
        f1 = f1_score(targets_bin, preds_bin, zero_division=0)

        logger.info("📊 is_malicious任务的基础指标:")
        logger.info(f"    准确率: {acc:.4f}")
        logger.info(f"    精确率: {prec:.4f}")
        logger.info(f"    召回率: {rec:.4f}")
        logger.info(f"    F1分数: {f1:.4f}")

        logger.info("📋 is_malicious任务的详细分类报告:")
        report = classification_report(targets_bin, preds_bin, labels=[0, 1], target_names=["正常", "恶意"], digits=4,
                                       zero_division=0)
        for line in report.split('\n'):
            if line.strip(): logger.info(line)

        logger.info("🎯 is_malicious任务的混淆矩阵:")
        cm = confusion_matrix(targets_bin, preds_bin, labels=[0, 1])
        logger.info(f"\n{cm}")

        logger.info("📈 is_malicious任务的高级指标:")
        if len(np.unique(targets_bin)) > 1:
            auc = roc_auc_score(targets_bin, prob_bin)
            ap = average_precision_score(targets_bin, prob_bin)
            logger.info(f"    ROC-AUC: {auc:.4f}")
            logger.info(f"    Average Precision: {ap:.4f}")
        else:
            logger.info("    ROC-AUC: N/A (仅包含单类数据)")
            logger.info("    Average Precision: N/A (仅包含单类数据)")

        total_samples = len(targets_bin)
        pos_samples = np.sum(targets_bin == 1)
        neg_samples = np.sum(targets_bin == 0)
        pos_ratio = (pos_samples / total_samples) * 100 if total_samples > 0 else 0

        logger.info("📊 样本的is_malicious标签数量统计:")
        logger.info(f"    总样本数: {total_samples}")
        logger.info(f"    正样本数: {pos_samples}.0")
        logger.info(f"    负样本数: {neg_samples}.0")
        logger.info(f"    正样本比例: {pos_ratio:.2f}%")
        logger.info("============================================================")

        attack_classes = ['DoS', 'DDoS', 'PortScan', 'BruteForce', 'Bot', 'Web Attack']

        if len(targets_multi) > 0:
            macro_f1_multi = f1_score(targets_multi, preds_multi, average="macro", zero_division=0)
            micro_f1_multi = f1_score(targets_multi, preds_multi, average="micro", zero_division=0)

            logger.info("============================================================")
            logger.info("🤖 图级别(GraphMAE) attack_family 任务测试报告")
            logger.info(f"macro_f1={macro_f1_multi:.4f}, micro_f1={micro_f1_multi:.4f}")

            logger.info("📋 attack_family 任务的分类报告:")
            report_multi = classification_report(targets_multi, preds_multi,
                                                 target_names=attack_classes[:len(np.unique(targets_multi))], digits=4,
                                                 zero_division=0)
            for line in report_multi.split('\n'):
                if line.strip(): logger.info(line)

            logger.info("📊 attack_family per-class F1:")
            per_class_f1 = f1_score(targets_multi, preds_multi, average=None, zero_division=0)
            unique_classes = np.unique(targets_multi).astype(int)
            for idx, cls_idx in enumerate(unique_classes):
                cls_name = attack_classes[cls_idx] if cls_idx < len(attack_classes) else f"Class_{cls_idx}"
                logger.info(f"    {cls_name:<18}: F1={per_class_f1[idx]:.4f}")
            logger.info("============================================================")

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)