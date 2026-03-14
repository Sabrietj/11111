import os
import sys
import glob
import torch
import logging
import argparse
import dgl
from tqdm import tqdm
from torch_geometric.data import Data
from dgl.data.utils import load_graphs, load_info

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def reconstruct_batch_from_ndata(ndata, device):
    """
    将 DGL 图的 ndata 重新组装为 Flow-BERT 期待的层次化字典 batch
    """
    batch = {}
    for k, v in ndata.items():
        if k.endswith("_input_ids"):
            view_name = k.replace("_input_ids", "_features")
            if view_name not in batch:
                batch[view_name] = {}
            batch[view_name]["input_ids"] = v.to(device)
        elif k.endswith("_attention_mask"):
            view_name = k.replace("_attention_mask", "_features")
            if view_name not in batch:
                batch[view_name] = {}
            batch[view_name]["attention_mask"] = v.to(device)
        else:
            batch[k] = v.to(device)

    # =====================================================================
    # 🎯 终极解法：从 packet_len_seq 的正负号中无损还原 directions
    # 在建图阶段，前向包载荷为正，后向包载荷为负，padding 为 0
    # BERT 需要 0 代表前向/padding，1 代表后向
    # =====================================================================
    if 'directions' not in batch and 'packet_len_seq' in batch:
        # 小于 0 的位置是后向包 (True -> 1)，大于等于 0 的位置是前向包或 Padding (False -> 0)
        batch['directions'] = (batch['packet_len_seq'] < 0).long().to(device)

    return batch


def extract_embeddings(bert_ckpt_path, raw_data_dir, output_dir, dataset_config_name, device='cuda'):
    logger.info(f"🔧 正在通过 原生 Hydra 解析配置: datasets={dataset_config_name}")

    config_abs_dir = os.path.abspath(os.path.join(current_dir, "..", "flow_bert_multiview", "config"))

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_abs_dir):
        cfg = compose(config_name="flow_bert_multiview_config",
                      overrides=[f"datasets={dataset_config_name}", "concept_drift.enabled=False"])

    # 配置数据集路径
    cfg.data.flow_data_path = os.path.join(raw_data_dir, "all_embedded_flow.csv")
    cfg.data.session_split.session_split_path = os.path.join(raw_data_dir, "all_split_session.csv")

    from src.models.flow_bert_multiview.data.prepare_flow_file_pipeline import prepare_sampled_data_files
    prepare_sampled_data_files(cfg)

    # 1. 强制建立物理输出目录
    splits_names = ["train", "val", "test"]
    for split in splits_names:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    logger.info(f"✅ 已确认输出目录结构: {output_dir}/{{train, val, test}}")

    # 2. 仅初始化 DataModule 以构建训练集字典映射（供模型加载时校验）
    logger.info("📦 正在初始化 DataModule... (仅作环境构建，不再使用其迭代器)")
    datamodule = MultiviewFlowDataModule(cfg)
    datamodule.setup("fit")

    # 3. 唤醒 Flow-BERT 模型
    logger.info(f"🚀 正在加载 Flow-BERT 权重: {bert_ckpt_path}")
    model = FlowBertMultiview.load_from_checkpoint(
        bert_ckpt_path,
        cfg=cfg,
        dataset=datamodule.train_dataset
    )
    model.to(device)
    model.eval()

    # ==========================================================
    # 🎯 终极解法：直接加载已经完美构建的 DGL 图打包文件
    # ==========================================================
    bin_files = glob.glob(os.path.join(raw_data_dir, "all_session_graph*.bin"))
    # 过滤掉可能的历史遗留杂散文件，确保找的是大包文件
    bin_files = [f for f in bin_files if "all_session_graph" in os.path.basename(f)]

    if not bin_files:
        logger.error(f"❌ 未找到 DGL 图包文件: {raw_data_dir}/all_session_graph*.bin")
        return

    bin_path = bin_files[0]
    info_path = bin_path.replace(".bin", "_info.pkl")

    logger.info(f"📂 正在加载已构建的全量 DGL 图: {bin_path}")
    g_list, _ = load_graphs(bin_path)
    logger.info(f"📂 正在加载图标签信息: {info_path}")
    info = load_info(info_path)

    logger.info(f"📊 成功在内存中加载 {len(g_list)} 个会话图结构！")

    # 建立 split 映射字典
    split_map = {}
    for idx in info.get('train_index', []): split_map[idx] = 'train'
    # 注意这里兼容 validate_index -> val 映射
    for idx in info.get('validate_index', []): split_map[idx] = 'val'
    for idx in info.get('test_index', []): split_map[idx] = 'test'

    success_count = {'train': 0, 'val': 0, 'test': 0}

    # 4. 批量转化并落盘
    with torch.no_grad():
        for i, g in enumerate(tqdm(g_list, desc="🧬 提取多视图图表征")):
            split_name = split_map.get(i)
            if not split_name:
                continue

            # 4.1 从打包的 g.ndata 里，无损还原出 Flow-BERT 需要的字典输入
            batch = reconstruct_batch_from_ndata(g.ndata, device)

            try:
                # 4.2 前向传播提取 128 维特征
                outputs = model(batch)
                if isinstance(outputs, dict) and 'multiview_embeddings' in outputs:
                    embeds = outputs['multiview_embeddings'].cpu()
                elif isinstance(outputs, tuple) and len(outputs) >= 2:
                    embeds = outputs[1].cpu()
                else:
                    embeds = outputs.cpu()

                # 4.3 无损提取拓扑关系与边特征
                src, dst = g.edges()
                edge_index = torch.stack([src.cpu(), dst.cpu()], dim=0)

                edge_attr = g.edata.get('etype', None)
                if edge_attr is not None:
                    edge_attr = edge_attr.cpu()

                # 4.4 对齐标签信息
                y_bin = torch.tensor([info['is_malicious'][i]], dtype=torch.float)
                y_multi = torch.tensor([info['label_id'][i]], dtype=torch.long)

                # 4.5 构建 PyG 终极数据结构
                pyg_data = Data(
                    x=embeds,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y_bin=y_bin,
                    y_multi=y_multi
                )

                # 保存至对应目录
                save_path = os.path.join(output_dir, split_name, f"graph_{i}.bin")
                torch.save(pyg_data, save_path)
                success_count[split_name] += 1

            except Exception as e:
                logger.error(f"❌ 处理第 {i} 个会话图时出错: {e}")

    logger.info("🎉 纯净图表征提取并分发完成！物理文件夹结果如下：")
    for sp in splits_names:
        logger.info(f"  📂 {sp} 目录: 成功生成 {success_count[sp]} 个图文件")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_ckpt", type=str, required=True, help="Flow-BERT 模型权重路径")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="包含原始数据及 CSV 的根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="生成的图数据保存根目录")
    parser.add_argument("--dataset_config", type=str, default="cic_ids_2017", help="数据集配置名称")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    extract_embeddings(
        bert_ckpt_path=args.bert_ckpt,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        dataset_config_name=args.dataset_config,
        device=args.device
    )


if __name__ == "__main__":
    main()