import os
import glob
import torch
import logging
import argparse
from tqdm import tqdm
from torch_geometric.data import Data

# 假设你的 Flow-BERT 模型定义在此处
from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
# 假设你有解析流特征和会话结构的工具（根据你的代码库结构适配）
from src.build_session_graph.session_parser import SessionParser
from src.build_session_graph.flow_node_builder import FlowNodeBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_embeddings(bert_ckpt_path, raw_session_dir, output_dir, device='cuda'):
    """
    使用训练好的 Flow-BERT 提取会话中各个流的表征，并构建图结构保存为 .bin 文件
    """
    logger.info(f"正在加载 Flow-BERT 模型: {bert_ckpt_path}")
    # 1. 加载冻结的 BERT 模型用于推理提取特征
    model = FlowBertMultiview.load_from_checkpoint(bert_ckpt_path)
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # 2. 找到所有需要处理的会话原始文件
    session_files = glob.glob(os.path.join(raw_session_dir, "*.pcap"))  # 或 .csv 等你的原始格式
    logger.info(f"共发现 {len(session_files)} 个会话文件等待处理...")

    parser = SessionParser()
    node_builder = FlowNodeBuilder()

    success_count = 0
    with torch.no_grad():
        for file_path in tqdm(session_files, desc="提取图表征"):
            try:
                # A. 解析会话文件，获取包含流属性的列表和流之间的边关系
                # flows: list of flow raw features, edge_index: shape [2, num_edges]
                flows, edge_index, label_bin, label_multi = parser.parse_session(file_path)

                if len(flows) == 0:
                    continue

                # B. 将每个流转换为多视图 BERT 需要的输入格式
                flow_inputs = node_builder.build_inputs(flows)  # 返回 input_ids, attention_mask 等

                # 转移到对应设备
                input_ids = flow_inputs['input_ids'].to(device)
                attention_mask = flow_inputs['attention_mask'].to(device)

                # C. 送入 BERT 提取 768 维特征
                # 假设 BERT 返回 shape: [num_nodes, 768]
                flow_embeddings = model(input_ids=input_ids, attention_mask=attention_mask)

                # D. 构建 PyTorch Geometric 的 Data 对象
                graph_data = Data(
                    x=flow_embeddings.cpu(),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    y_bin=torch.tensor([label_bin], dtype=torch.float),
                    y_multi=torch.tensor([label_multi], dtype=torch.long)
                )

                # E. 保存为 GraphMAE 可读的 .bin 文件
                base_name = os.path.basename(file_path).split('.')[0]
                save_path = os.path.join(output_dir, f"{base_name}.bin")
                torch.save(graph_data, save_path)
                success_count += 1

            except Exception as e:
                logger.error(f"处理文件 {file_path} 时发生错误: {e}")

    logger.info(f"提取完成！成功生成 {success_count} 个图文件，存放在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract Flow Embeddings using Flow-BERT to build Session Graphs")
    parser.add_argument("--bert_ckpt", type=str, required=True, help="训练好的 Flow-BERT checkpoint 路径")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="原始会话数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="生成的 .bin 图数据保存目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")

    args = parser.parse_args()
    extract_embeddings(args.bert_ckpt, args.raw_data_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()