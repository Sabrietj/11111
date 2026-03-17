import os
import sys
import ast
import torch
import logging
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader
from torch_geometric.data import Data

# ============================================================================
# 🚨 路径修复：确保项目根目录和 utils 目录在系统路径中
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
utils_path = os.path.join(project_root, "src", "utils")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
# ============================================================================

from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir
from src.models.flow_bert_multiview.data.flow_bert_multiview_datamodule import MultiviewFlowDataModule
from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataset
import src.utils.config_manager as ConfigManager

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 🏷️ 100% 复刻 SessionParser 聚合逻辑
# ============================================================================
def normalize_label(label: str) -> str:
    if label is None: return ""
    return str(label).strip().lower()


def is_malicious(raw_label: str) -> bool:
    norm_label = normalize_label(raw_label)
    if norm_label.startswith(("benign", "normal", "legitimate")):
        return False
    return True


def match_configured_label(raw_label_norm: str, label_id_map: dict) -> str:
    raw_label_norm = raw_label_norm.strip().lower()
    for configured_label in label_id_map.keys():
        configured_lower = configured_label.lower()
        if raw_label_norm == configured_lower:
            return configured_label
        if configured_lower in raw_label_norm:
            return configured_label
    return None


def aggregate_session_label(uids, flow_meta_dict, label_id_map, dominant_ratio_threshold=0.8):
    benign_counter = Counter()
    attack_counter = Counter()

    for flow_uid in uids:
        raw_label = str(flow_meta_dict[flow_uid]['label'])
        class_type = match_configured_label(raw_label, label_id_map)
        if class_type is None: continue

        if is_malicious(raw_label):
            attack_counter[class_type] += 1
        else:
            benign_counter[class_type] += 1

    if sum(attack_counter.values()) == 0:
        if len(benign_counter) == 1:
            label_name = next(iter(benign_counter))
        else:
            label_name = "benign_unknown" if "benign_unknown" in label_id_map else "benign"
    elif len(attack_counter) == 1:
        label_name = next(iter(attack_counter))
    else:
        total_attack = sum(attack_counter.values())
        dominant_label, dominant_count = attack_counter.most_common(1)[0]
        if dominant_count / total_attack >= dominant_ratio_threshold:
            label_name = dominant_label
        else:
            label_name = "mixed"

    label_name = normalize_label(label_name)
    is_mal = is_malicious(label_name)

    if label_name == "mixed":
        return "mixed", -1, is_mal

    label_id = label_id_map.get(label_name, -1)
    return label_name, label_id, is_mal


# ============================================================================
# 🕸️ 100% 像素级复刻 SessionGraphBuilder 拓扑逻辑
# ============================================================================
class ExactSessionGraphBuilder:
    def __init__(self, flow_meta_dict, label_id_map, global_embeddings, concurrent_thresh=1.0, seq_thresh=10.0,
                 k_nearest=3):
        self.flow_meta = flow_meta_dict
        self.label_id_map = label_id_map
        self.global_embeddings = global_embeddings
        self.concurrent_thresh = concurrent_thresh
        self.seq_thresh = seq_thresh
        self.k_nearest = k_nearest

    def create_concurrent_edges(self, burst, edges):
        n = len(burst)
        if n <= 1: return
        for j in range(n):
            for d in range(1, self.k_nearest + 1):
                if j + d < n:
                    edges.append((burst[j]['id'], burst[j + d]['id'], 0))

    def build_graphs_for_session(self, uids):
        valid_uids = [u for u in uids if u in self.flow_meta and u in self.global_embeddings]
        if not valid_uids: return []

        flows = [{'uid': u, 'ts': self.flow_meta[u]['ts']} for u in valid_uids]
        flows.sort(key=lambda x: x['ts'])

        # 1. Flow Burst 聚类
        flow_bursts = []
        current_burst = [flows[0]]
        for flow in flows[1:]:
            if abs(flow['ts'] - current_burst[-1]['ts']) <= self.concurrent_thresh:
                current_burst.append(flow)
            else:
                flow_bursts.append(current_burst)
                current_burst = [flow]
        if current_burst: flow_bursts.append(current_burst)

        def calc_avg_interval(win):
            if len(win) <= 1: return self.concurrent_thresh
            times = [f['ts'] for f in win]
            return (max(times) - min(times)) / (len(win) - 1)

        # 2. 会话窗口聚合与截断
        session_windows = []
        curr_win = list(flow_bursts[0])
        for next_burst in flow_bursts[1:]:
            avg_before = calc_avg_interval(curr_win)
            avg_after = calc_avg_interval(curr_win + next_burst)
            if avg_after <= self.seq_thresh * avg_before:
                curr_win.extend(next_burst)
            else:
                session_windows.append(curr_win)
                curr_win = list(next_burst)
        if curr_win: session_windows.append(curr_win)

        # 3. 构建 PyG 格式的图
        graphs = []
        for win in session_windows:
            nodes = win
            for idx, n in enumerate(nodes): n['id'] = idx
            edges = []  # (src, dst, etype)

            # 窗口内再次 Node Burst 聚类并生成并发边
            node_bursts = []
            curr_b = [nodes[0]]
            for n in nodes[1:]:
                if abs(n['ts'] - curr_b[-1]['ts']) <= self.concurrent_thresh:
                    curr_b.append(n)
                else:
                    self.create_concurrent_edges(curr_b, edges)
                    node_bursts.append(curr_b)
                    curr_b = [n]
            if curr_b:
                self.create_concurrent_edges(curr_b, edges)
                node_bursts.append(curr_b)

            # 顺序边构建
            for j in range(len(node_bursts) - 1):
                cb, nb = node_bursts[j], node_bursts[j + 1]
                if len(cb) == 1 and len(nb) == 1:
                    edges.append((cb[0]['id'], nb[0]['id'], 1))
                else:
                    edges.append((cb[-1]['id'], nb[-1]['id'], 1))
                    edges.append((cb[0]['id'], nb[0]['id'], 1))

            # 聚合标签
            win_uids = [n['uid'] for n in nodes]
            label_name, label_id, is_malicious = aggregate_session_label(win_uids, self.flow_meta, self.label_id_map)

            num_nodes = len(nodes)
            # 补充自环边 (etype=0)，保证消息传递不崩溃
            for i in range(num_nodes):
                edges.append((i, i, 0))

            src = [e[0] for e in edges]
            dst = [e[1] for e in edges]
            etype = [e[2] for e in edges]

            # 从全局字典获取 128 维特征
            x = torch.stack([self.global_embeddings[u] for u in win_uids])
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(etype, dtype=torch.long)

            pyg_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y_bin=torch.tensor([is_malicious], dtype=torch.float),
                y_multi=torch.tensor([label_id], dtype=torch.long)  # 包含 -1 等越界标签，交给模型屏蔽
            )
            graphs.append(pyg_data)

        return graphs


def get_flow_metadata(flow_csv_path):
    """提取时间戳和标签（完美兼容 conn.ts）"""
    logger.info(f"读取全局 Flow 元数据: {flow_csv_path}")
    df = pd.read_csv(flow_csv_path, low_memory=False).copy()

    time_col = None
    time_candidates = ['conn.ts', 'ts', 'flow_start', 'timestamp', 'Time', 'Flow Start']
    for col in time_candidates:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        logger.warning("⚠️ 未找到时间戳列！时序图将退化为并发图。")
        df['ts'] = 0.0
        time_col = 'ts'

    label_col = next((col for col in ['label', 'Label', 'attack_type'] if col in df.columns), 'label')
    if label_col not in df.columns: df['label'] = 'benign'

    df = df.rename(columns={time_col: 'ts', label_col: 'label'})
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce').fillna(0.0)
    return df.set_index('uid')[['ts', 'label']].to_dict('index')


def extract_embeddings(bert_ckpt_path, raw_data_dir, output_dir, dataset_config_name, device='cuda'):
    logger.info(f"🔧 正在通过 原生 Hydra 解析配置: datasets={dataset_config_name}")

    config_abs_dir = os.path.abspath(os.path.join(current_dir, "..", "flow_bert_multiview", "config"))
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_abs_dir):
        cfg = compose(config_name="flow_bert_multiview_config",
                      overrides=[f"datasets={dataset_config_name}", "concept_drift.enabled=False"])

    cfg.data.flow_data_path = os.path.join(raw_data_dir, "all_embedded_flow.csv")
    cfg.data.session_split.session_split_path = os.path.join(raw_data_dir, "all_split_session.csv")

    from src.models.flow_bert_multiview.data.prepare_flow_file_pipeline import prepare_sampled_data_files
    prepare_sampled_data_files(cfg)

    splits_names = ["train", "val", "test"]
    for split in splits_names:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    logger.info("📦 初始化 DataModule 以获取数据统计和映射...")
    datamodule = MultiviewFlowDataModule(cfg)
    datamodule.setup("fit")
    datamodule.setup("test")

    logger.info(f"🚀 加载训练好的 Flow-BERT: {bert_ckpt_path}")
    model = FlowBertMultiview.load_from_checkpoint(bert_ckpt_path, cfg=cfg, dataset=datamodule.train_dataset)
    model.to(device)
    model.eval()

    logger.info("⏳ 阶段 2: 准备对全量 Flow 数据进行一次性极速特征推理...")
    all_flow_df = datamodule.flow_df
    all_uids = all_flow_df['uid'].tolist()

    all_dataset = MultiviewFlowDataset(
        all_flow_df, cfg, is_training=False,
        train_categorical_mappings=datamodule.train_dataset.categorical_val2idx_mappings,
        train_categorical_columns_effective=datamodule.train_dataset.categorical_columns_effective
    )
    all_dataset.numeric_stats = datamodule.train_dataset.numeric_stats
    all_dataset.apply_numeric_stats()

    batch_size = cfg.data.get('batch_size', 256)
    num_workers = cfg.data.get('num_workers', 4)
    loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    global_flow_embeddings = {}
    with torch.no_grad():
        all_embeds = []
        for batch in tqdm(loader, desc="⚡ 全局 Flow 离线推理"):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch_gpu)
            if isinstance(outputs, dict) and 'multiview_embeddings' in outputs:
                embeds = outputs['multiview_embeddings'].cpu()
            elif isinstance(outputs, tuple):
                embeds = outputs[1].cpu()
            else:
                embeds = outputs.cpu()
            all_embeds.append(embeds)
        all_embeds = torch.cat(all_embeds, dim=0)

    for uid, emb in zip(all_uids, all_embeds):
        global_flow_embeddings[uid] = emb

    dict_path = os.path.join(output_dir, "flow_uid_to_embedding.pt")
    torch.save(global_flow_embeddings, dict_path)
    logger.info(f"✅ 阶段 2 完成！全局特征字典已落盘: {dict_path}")

    logger.info("⏳ 阶段 3: 提取流时间戳并构建 PyG 纯净拓扑图 (为 GraphMAE 准备)...")
    flow_meta_dict = get_flow_metadata(cfg.data.flow_data_path)
    label_id_map = ConfigManager.read_session_label_id_map(cfg.data.dataset)

    builder = ExactSessionGraphBuilder(
        flow_meta_dict=flow_meta_dict,
        label_id_map=label_id_map,
        global_embeddings=global_flow_embeddings,
        concurrent_thresh=cfg.data.get('concurrent_flow_iat_threshold', 1.0),
        seq_thresh=cfg.data.get('sequential_flow_iat_threshold', 10.0),
        k_nearest=3  # 保持与原逻辑一致
    )

    session_df = datamodule.session_df
    split_name_map = {datamodule.train_split: 'train', datamodule.validate_split: 'val', datamodule.test_split: 'test'}

    pyg_graphs_dict = {'train': [], 'val': [], 'test': []}

    for i, row in tqdm(session_df.iterrows(), total=len(session_df), desc="构建 PyG 纯净图"):
        split_val = row[datamodule.split_column]
        if split_val not in split_name_map: continue

        target_dir_name = split_name_map[split_val]
        uid_str = str(row[datamodule.flow_uid_list_column])
        if uid_str.startswith('['):
            uids = ast.literal_eval(uid_str)
        else:
            uids = [u.strip() for u in uid_str.split(',') if u.strip()]

        graphs = builder.build_graphs_for_session(uids)
        pyg_graphs_dict[target_dir_name].extend(graphs)

    logger.info("📦 正在将 PyG 图按集合打包保存为一个大文件 (graphs.pt) ...")
    for sp in splits_names:
        if len(pyg_graphs_dict[sp]) > 0:
            save_path = os.path.join(output_dir, sp, "graphs.pt")
            torch.save(pyg_graphs_dict[sp], save_path)
            logger.info(f"  📂 {sp} 目录: 成功生成极速包 graphs.pt (内含 {len(pyg_graphs_dict[sp])} 张图)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_ckpt", type=str, required=True)
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default="cic_ids_2017")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    extract_embeddings(
        bert_ckpt_path=args.bert_ckpt, raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir, dataset_config_name=args.dataset_config, device=args.device
    )


if __name__ == "__main__":
    main()