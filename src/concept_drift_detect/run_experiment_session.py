import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
import numpy as np

# 路径定位
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'utils'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from logging_config import setup_preset_logging

logger = setup_preset_logging(log_level=logging.INFO)

# 导入底层组件
from models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule
from models.session_graphmae.graphmae_model import SessionGraphMAE
from concept_drift_detect.streaming_graph_buffer import StreamingSessionGraphBuffer
from concept_drift_detect.detectors import BNDMDetector
from concept_drift_detect.adapter import ModelAdapter


def print_gorgeous_report(targets_bin, preds_bin, prob_bin, targets_multi, preds_multi, prob_multi, level_name=""):
    """完美对齐你要求的华丽报表打印函数"""
    logger.info("============================================================")
    logger.info(f"🤖 {level_name} is_malicious任务的测试报告")
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
        logger.info(f"🤖 {level_name} attack_family 任务测试报告（简要）")
        logger.info(f"macro_f1={macro_f1_multi:.4f}, micro_f1={micro_f1_multi:.4f}")

        logger.info("📋 attack_family 任务的分类报告:")
        report_multi = classification_report(targets_multi, preds_multi, target_names=attack_classes, digits=4,
                                             zero_division=0)
        for line in report_multi.split('\n'):
            if line.strip(): logger.info(line)

        logger.info("📊 attack_family per-class F1:")
        per_class_f1 = f1_score(targets_multi, preds_multi, average=None, zero_division=0)
        for idx, cls_name in enumerate(attack_classes):
            logger.info(f"    {cls_name:<18}: F1={per_class_f1[idx]:.4f}")
        logger.info("============================================================")


@hydra.main(config_path="../models/flow_bert_multiview/config", config_name="flow_bert_multiview_config",
            version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = "/tmp/pycharm_project_982/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto"

    # =========================================================
    # 0. 强行锁定数据路径
    # =========================================================
    OmegaConf.set_struct(cfg, False)
    if "dataset" not in cfg: cfg.dataset = {}
    cfg.dataset.name = "CIC-IDS-2017"
    cfg.data.flow_data_path = os.path.join(dataset_dir, "all_embedded_flow.csv")
    if "session_split" not in cfg.data: cfg.data.session_split = {}
    cfg.data.session_split.session_split_path = os.path.join(dataset_dir, "all_split_session.csv")

    # =========================================================
    # 1. 准备数据流 (DataModule)
    # =========================================================
    logger.info("📦 正在初始化 DataModule 加载测试数据流...")
    datamodule = MultiviewFlowDataModule(cfg)
    datamodule.setup("test")
    # 模拟真实流环境：Batch Size = 1, 不打乱顺序
    test_loader = DataLoader(datamodule.test_dataset, batch_size=1, shuffle=False)

    # 预先获取原始 DataFrame 以便读取 session_id 和时间戳
    test_df = datamodule.test_dataset.flow_df

    # =========================================================
    # 2. 加载两个级别的核心模型
    # =========================================================
    flow_ckpt = os.path.join(PROJECT_ROOT, "processed_data", "best_model.ckpt")
    logger.info("⚙️ 加载流级别多视图模型 (FlowBertMultiview) - 动态适应节点...")
    flow_bert = FlowBertMultiview.load_from_checkpoint(flow_ckpt, cfg=cfg, dataset=datamodule.test_dataset)
    flow_bert.to(device)
    flow_bert.eval()

    graph_ckpt = os.path.join(dataset_dir, "best_graphmae.ckpt")
    logger.info("⚙️ 加载图级别网络结构模型 (GraphMAE) - 冻结推断节点...")
    graph_mae = SessionGraphMAE.load_from_checkpoint(graph_ckpt)
    graph_mae.to(device)
    graph_mae.eval()  # 图模型在测试阶段绝对冻结

    # =========================================================
    # 3. 初始化概念漂移检测器、适配器与流式图缓存器
    # =========================================================
    detector = BNDMDetector(cfg)
    adapter = ModelAdapter(flow_bert, cfg)

    # 从配置文件或使用默认值初始化缓存器 (并发 0.1s, 顺序 1.0s)
    graph_buffer = StreamingSessionGraphBuffer(concurrent_threshold=0.1, sequential_threshold=1.0)

    # =========================================================
    # 4. 结果收集器
    # =========================================================
    flow_results = {'targets_bin': [], 'preds_bin': [], 'prob_bin': [], 'targets_multi': [], 'preds_multi': [],
                    'prob_multi': []}
    graph_results = {'targets_bin': [], 'preds_bin': [], 'prob_bin': [], 'targets_multi': [], 'preds_multi': [],
                     'prob_multi': []}

    logger.info("🚀======================================================🚀")
    logger.info("🚀 启动全链路流式增量推断 (Flow Streaming & Graph Inference) 🚀")
    logger.info("🚀======================================================🚀")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="处理网络流")):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # ---------------------------------------------------------
            # 阶段 A：流级别推断 (Flow Inference)
            # ---------------------------------------------------------
            outputs = flow_bert(batch)
            logits_bin = outputs['is_malicious_logits']
            logits_multi = outputs['attack_family_logits']

            prob_bin = torch.sigmoid(logits_bin).squeeze(-1).item()
            prob_multi = torch.sigmoid(logits_multi).squeeze(0).cpu().numpy()

            y_bin = batch['is_malicious_label']
            y_multi = batch['attack_family_label']

            flow_results['prob_bin'].append(prob_bin)
            flow_results['preds_bin'].append(int(prob_bin > 0.5))
            flow_results['targets_bin'].append(y_bin.item())

            flow_results['prob_multi'].append(prob_multi)
            flow_results['preds_multi'].append((prob_multi > 0.5).astype(int))
            flow_results['targets_multi'].append(y_multi.squeeze(0).cpu().numpy())

            # ---------------------------------------------------------
            # 阶段 B：概念漂移检测与在线适应 (BNDM & Adaptation)
            # ---------------------------------------------------------
            is_drift = detector.detect(outputs['multiview_embeddings'].cpu().numpy())
            if is_drift:
                logger.warning(f"⚠️ 在第 {i} 条流检测到概念漂移！启动增量适应...")
                # 开启梯度进行小样本自适应训练
                torch.set_grad_enabled(True)
                flow_bert.train()
                adapter.adapt(batch)
                flow_bert.eval()
                torch.set_grad_enabled(False)
                logger.info("✅ 适应完成，底层模型权重已更新。")

            # ---------------------------------------------------------
            # 阶段 C：提取适应后的流表征，送入图缓存器
            # ---------------------------------------------------------
            # 获取最新表征 x_i
            x_i = flow_bert.extract_fusion_features(batch)  # [1, 768]

            # 从原始 DF 读取 session_id 和 timestamp
            session_id = test_df.iloc[i]['session_id']
            # 如果你的 df 没有 timestamp 字段，可以用索引模拟时间流逝
            timestamp = test_df.iloc[i].get('flowmeter.packet_timestamp_vector', [i])[0]

            # 放入缓存池，检查是否有过期的 Session 吐出图
            completed_graph = graph_buffer.add_flow_and_check_completion(
                session_id, timestamp, x_i, y_bin, y_multi
            )

            # ---------------------------------------------------------
            # 阶段 D：图级别推理 (Graph Inference)
            # ---------------------------------------------------------
            if completed_graph is not None:
                g = completed_graph.to(device)
                g_batch_idx = torch.zeros(g.x.size(0), dtype=torch.long, device=device)

                g_logits_bin, g_logits_multi = graph_mae(g.x, g.edge_index, g_batch_idx)

                g_prob_bin = torch.sigmoid(g_logits_bin).squeeze(-1).item()
                g_prob_multi = torch.sigmoid(g_logits_multi).squeeze(0).cpu().numpy()

                graph_results['prob_bin'].append(g_prob_bin)
                graph_results['preds_bin'].append(int(g_prob_bin > 0.5))
                graph_results['targets_bin'].append(g.y_bin.item())

                graph_results['prob_multi'].append(g_prob_multi)
                graph_results['preds_multi'].append((g_prob_multi > 0.5).astype(int))
                graph_results['targets_multi'].append(g.y_multi.squeeze(0).cpu().numpy())

    # ---------------------------------------------------------
    # 阶段 E：强制清空缓存池中最后剩余的图并推理
    # ---------------------------------------------------------
    remaining_graphs = graph_buffer.force_flush_all()
    for g in remaining_graphs:
        g = g.to(device)
        g_batch_idx = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
        g_logits_bin, g_logits_multi = graph_mae(g.x, g.edge_index, g_batch_idx)

        g_prob_bin = torch.sigmoid(g_logits_bin).squeeze(-1).item()
        g_prob_multi = torch.sigmoid(g_logits_multi).squeeze(0).cpu().numpy()

        graph_results['prob_bin'].append(g_prob_bin)
        graph_results['preds_bin'].append(int(g_prob_bin > 0.5))
        graph_results['targets_bin'].append(g.y_bin.item())

        graph_results['prob_multi'].append(g_prob_multi)
        graph_results['preds_multi'].append((g_prob_multi > 0.5).astype(int))
        graph_results['targets_multi'].append(g.y_multi.squeeze(0).cpu().numpy())

    # =========================================================
    # 5. 打印双料终极成绩单
    # =========================================================
    logger.info("\n\n🎉 整个测试集流式推断结束！生成最终成绩单：\n")

    # 打印流级别成绩
    print_gorgeous_report(
        np.array(flow_results['targets_bin']), np.array(flow_results['preds_bin']), np.array(flow_results['prob_bin']),
        np.array(flow_results['targets_multi']), np.array(flow_results['preds_multi']),
        np.array(flow_results['prob_multi']),
        level_name="流级别(FlowBert)"
    )

    # 打印图级别成绩
    print_gorgeous_report(
        np.array(graph_results['targets_bin']), np.array(graph_results['preds_bin']),
        np.array(graph_results['prob_bin']),
        np.array(graph_results['targets_multi']), np.array(graph_results['preds_multi']),
        np.array(graph_results['prob_multi']),
        level_name="图级别(GraphMAE)"
    )


if __name__ == "__main__":
    main()