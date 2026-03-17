import os
import sys
import glob
import torch
import argparse
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
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(src_dir, ".."))

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logging_config import setup_preset_logging

logger = setup_preset_logging(log_level=logging.INFO)

# 导入底层组件
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_datamodule import MultiviewFlowDataModule
from src.models.session_graphmae.graphmae_model import SessionGraphMAE

# 导入漂移检测组件
try:
    from concept_drift_detect.streaming_graph_buffer import StreamingSessionGraphBuffer
    from concept_drift_detect.detectors import BNDMDetector
    from concept_drift_detect.adapter import IncrementalAdapter
except ImportError as e:
    logger.error(f"❌ 无法导入漂移检测组件，路径未对齐: {e}")
    sys.exit(1)


def print_gorgeous_report(targets_bin, preds_bin, prob_bin, targets_multi, preds_multi, prob_multi, level_name=""):
    """华丽报表打印函数"""
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
        actual_classes = np.unique(targets_multi)
        target_names = [attack_classes[i] for i in actual_classes if i < len(attack_classes)]

        report_multi = classification_report(targets_multi, preds_multi, labels=actual_classes,
                                             target_names=target_names, digits=4, zero_division=0)
        for line in report_multi.split('\n'):
            if line.strip(): logger.info(line)

        logger.info("============================================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="加工后的数据集根目录")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Flow-BERT 模型权重路径")
    parser.add_argument("--dataset_config", type=str, default="cic_ids_2017", help="数据集配置")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"🔧 正在通过 原生 Hydra 解析配置: datasets={args.dataset_config}")
    config_abs_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "src", "models", "flow_bert_multiview", "config"))
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_abs_dir):
        cfg = compose(config_name="flow_bert_multiview_config", overrides=[f"datasets={args.dataset_config}"])

    cfg.data.flow_data_path = os.path.join(args.dataset_dir, "all_embedded_flow.csv")
    if "session_split" not in cfg.data: cfg.data.session_split = {}
    cfg.data.session_split.session_split_path = os.path.join(args.dataset_dir, "all_split_session.csv")

    from src.models.flow_bert_multiview.data.prepare_flow_file_pipeline import prepare_sampled_data_files
    prepare_sampled_data_files(cfg)

    logger.info("📦 正在初始化 DataModule 加载测试数据流...")
    datamodule = MultiviewFlowDataModule(cfg)
    datamodule.setup("fit")
    datamodule.setup("test")

    test_loader = DataLoader(datamodule.test_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_df = datamodule.test_dataset.flow_df

    logger.info("⚙️ 加载流级别多视图模型 (FlowBertMultiview) - 动态适应节点...")
    flow_bert = FlowBertMultiview.load_from_checkpoint(args.model_ckpt, cfg=cfg, dataset=datamodule.test_dataset)
    flow_bert.to(device)
    flow_bert.eval()

    graph_ckpt_dir = os.path.join(args.dataset_dir, "saved_models", "graphmae")
    graph_ckpt_files = glob.glob(os.path.join(graph_ckpt_dir, "*.ckpt"))
    if not graph_ckpt_files:
        logger.error(f"❌ 未找到 GraphMAE 权重: {graph_ckpt_dir}")
        sys.exit(1)
    graph_ckpt = graph_ckpt_files[0]

    logger.info(f"⚙️ 加载图级别网络结构模型 (GraphMAE) - 冻结推断节点... ({os.path.basename(graph_ckpt)})")
    graph_mae = SessionGraphMAE.load_from_checkpoint(graph_ckpt)
    graph_mae.to(device)
    graph_mae.eval()

    # 初始化检测器和适配器
    try:
        det_config = OmegaConf.to_container(cfg.concept_drift.detectors.bndm, resolve=True)
    except:
        det_config = {'seed': 2026}

    try:
        adapt_config = OmegaConf.to_container(cfg.concept_drift.adaptation, resolve=True)
    except:
        adapt_config = {'buffer_size': 2000, 'batch_size': 32, 'lr': 1e-4, 'epochs': 3}

    detector = BNDMDetector(det_config)
    adapter = IncrementalAdapter(flow_bert, adapt_config)
    graph_buffer = StreamingSessionGraphBuffer(concurrent_threshold=0.1, sequential_threshold=1.0)

    flow_results = {'targets_bin': [], 'preds_bin': [], 'prob_bin': [], 'targets_multi': [], 'preds_multi': [],
                    'prob_multi': []}
    graph_results = {'targets_bin': [], 'preds_bin': [], 'prob_bin': [], 'targets_multi': [], 'preds_multi': [],
                     'prob_multi': []}

    logger.info("🚀======================================================🚀")
    logger.info("🚀 启动全链路流式增量推断 (Flow Streaming & Graph Inference) 🚀")
    logger.info("🚀======================================================🚀")

    time_col = next((col for col in ['conn.ts', 'ts', 'flow_start', 'timestamp', 'Time'] if col in test_df.columns),
                    None)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="处理网络流", mininterval=2.0)):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = flow_bert(batch_gpu)

            logits_bin = outputs['is_malicious_cls_logits']
            logits_multi = outputs['attack_family_cls_logits']

            prob_bin = torch.sigmoid(logits_bin).squeeze(-1).item()
            # 🎯 修复: 增加 .detach() 安全剥离张量
            prob_multi_array = torch.sigmoid(logits_multi).squeeze(0).detach().cpu().numpy()
            pred_multi_idx = int(np.argmax(prob_multi_array))

            y_bin_tensor = batch_gpu.get('is_malicious_label', torch.tensor([[0.0]], device=device))
            y_bin = y_bin_tensor.view(-1)[0].item()

            y_multi_tensor = batch_gpu.get('attack_family_label', torch.zeros((1, 6), device=device))
            if y_multi_tensor.dim() > 1:
                y_multi_idx_tensor = torch.argmax(y_multi_tensor, dim=1)
                y_multi_idx = int(y_multi_idx_tensor.item())
            else:
                y_multi_idx_tensor = y_multi_tensor.long()
                y_multi_idx = int(y_multi_idx_tensor.item())

            flow_results['prob_bin'].append(prob_bin)
            flow_results['preds_bin'].append(int(prob_bin > 0.5))
            flow_results['targets_bin'].append(y_bin)

            flow_results['prob_multi'].append(prob_multi_array)
            flow_results['preds_multi'].append(pred_multi_idx)
            flow_results['targets_multi'].append(y_multi_idx)

            x_i = outputs['multiview_embeddings']

            # 🎯 修复: 增加 .detach() 安全剥离张量
            val = detector.preprocess(x_i.detach())
            is_drift = detector.update(val)

            if is_drift:
                logger.warning(f"⚠️ 在第 {i} 条流检测到概念漂移！启动增量适应...")
                torch.set_grad_enabled(True)
                flow_bert.train()

                adapter.adapt(batch_gpu, y_bin_tensor.float().view(1), y_multi_idx_tensor)

                flow_bert.eval()
                torch.set_grad_enabled(False)

                outputs = flow_bert(batch_gpu)
                x_i = outputs['multiview_embeddings']

                detector.reset()
                logger.info("✅ 适应完成，底层模型权重已更新，检测器已重置。")

            session_id = test_df.iloc[i].get('uid', f"uid_{i}")
            timestamp = test_df.iloc[i][time_col] if time_col else i

            completed_graph = graph_buffer.add_flow_and_check_completion(
                session_id, timestamp, x_i.detach().cpu(), y_bin, y_multi_idx
            )

            if completed_graph is not None:
                g = completed_graph.to(device)
                g_batch_idx = torch.zeros(g.x.size(0), dtype=torch.long, device=device)

                g_logits_bin, g_logits_multi = graph_mae(g.x, g.edge_index, g_batch_idx)

                g_prob_bin = torch.sigmoid(g_logits_bin).squeeze(-1).item()
                # 🎯 修复: 增加 .detach() 安全剥离张量
                g_prob_multi = torch.softmax(g_logits_multi, dim=-1).squeeze(0).detach().cpu().numpy()
                g_pred_multi_idx = int(np.argmax(g_prob_multi))

                graph_results['prob_bin'].append(g_prob_bin)
                graph_results['preds_bin'].append(int(g_prob_bin > 0.5))
                graph_results['targets_bin'].append(g.y_bin.item())

                graph_results['prob_multi'].append(g_prob_multi)
                graph_results['preds_multi'].append(g_pred_multi_idx)
                graph_results['targets_multi'].append(int(g.y_multi.item()))

    # 清理缓存池
    remaining_graphs = graph_buffer.force_flush_all()
    for g in remaining_graphs:
        g = g.to(device)
        g_batch_idx = torch.zeros(g.x.size(0), dtype=torch.long, device=device)

        g_logits_bin, g_logits_multi = graph_mae(g.x, g.edge_index, g_batch_idx)

        g_prob_bin = torch.sigmoid(g_logits_bin).squeeze(-1).item()
        # 🎯 修复: 增加 .detach() 安全剥离张量
        g_prob_multi = torch.softmax(g_logits_multi, dim=-1).squeeze(0).detach().cpu().numpy()
        g_pred_multi_idx = int(np.argmax(g_prob_multi))

        graph_results['prob_bin'].append(g_prob_bin)
        graph_results['preds_bin'].append(int(g_prob_bin > 0.5))
        graph_results['targets_bin'].append(g.y_bin.item())

        graph_results['prob_multi'].append(g_prob_multi)
        graph_results['preds_multi'].append(g_pred_multi_idx)
        graph_results['targets_multi'].append(int(g.y_multi.item()))

    logger.info("\n\n🎉 整个测试集流式推断结束！生成最终成绩单：\n")

    print_gorgeous_report(
        np.array(flow_results['targets_bin']), np.array(flow_results['preds_bin']), np.array(flow_results['prob_bin']),
        np.array(flow_results['targets_multi']), np.array(flow_results['preds_multi']),
        np.array(flow_results['prob_multi']),
        level_name="流级别(FlowBert)"
    )

    if len(graph_results['targets_bin']) > 0:
        print_gorgeous_report(
            np.array(graph_results['targets_bin']), np.array(graph_results['preds_bin']),
            np.array(graph_results['prob_bin']),
            np.array(graph_results['targets_multi']), np.array(graph_results['preds_multi']),
            np.array(graph_results['prob_multi']),
            level_name="图级别(GraphMAE)"
        )
    else:
        logger.warning("缓存器没有吐出任何有效图！请检查 Buffer 逻辑。")


if __name__ == "__main__":
    main()