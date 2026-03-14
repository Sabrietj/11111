import os
import sys
import subprocess
import argparse
import logging

# 🟢 彻底杜绝 HuggingFace 分词器死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🚀 %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd, step_name):
    logger.info(f"开始执行阶段: 【{step_name}】")
    logger.info(f"执行命令: {cmd}")

    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        logger.error(f"❌ 阶段【{step_name}】执行失败，流水线终止！")
        sys.exit(1)

    logger.info(f"✅ 阶段【{step_name}】执行成功！\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MultiViewBert-GraphMAE 统一实验流水线")
    parser.add_argument("--mode", type=str, choices=["train_all", "test_only"], default="train_all")
    parser.add_argument("--dataset_config", type=str, default="cic_ids_2017",
                        help="Hydra数据集配置文件名，例如 cic_ids_2017")

    # 🎯 核心修复 1：将默认路径直接设定为你服务器上的真实绝对路径
    parser.add_argument("--base_data_dir", type=str,
                        default="/tmp/pycharm_project_908/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto",
                        help="会话数据的根目录")

    args = parser.parse_args()
    base_cmd = f"{sys.executable}"

    # 🎯 确保 base_data_dir 是绝对路径
    abs_base_dir = os.path.abspath(args.base_data_dir)

    # 路径规划：全部存放在该会话目录下，互不干扰
    bert_ckpt_dir = os.path.join(abs_base_dir, "saved_models", "flow_bert")
    bert_ckpt_path = os.path.join(bert_ckpt_dir, "best_model.ckpt")

    graphmae_ckpt_dir = os.path.join(abs_base_dir, "saved_models", "graphmae")
    graph_data_dir = os.path.join(abs_base_dir, "processed_graphs")

    logger.info(f"🌟 欢迎使用 MultiViewBert-GraphMAE 统一流水线 🌟")
    logger.info(f"当前模式: {args.mode}")
    logger.info(f"配置文件: datasets={args.dataset_config}")
    logger.info(f"真实数据总目录: {abs_base_dir}")
    logger.info(f"FlowBert 最佳模型将保存至: {bert_ckpt_path}")
    logger.info(f"GraphMAE 最佳模型将保存至: {graphmae_ckpt_dir}/best_model.ckpt")
    logger.info(f"GraphMAE 图特征将保存至: {graph_data_dir}")
    logger.info("=" * 60)

    if args.mode == "train_all":
        # -------------------------------------------------------------
        # 阶段 1：预训练 FlowBert
        # -------------------------------------------------------------
        # run_command(
        #     f"{base_cmd} src/models/flow_bert_multiview/train.py datasets={args.dataset_config} concept_drift.enabled=False",
        #     "1. FlowBert 流级别预训练 (纯净离线评估)")

        # -------------------------------------------------------------
        # 阶段 2：提取 Graph Embeddings (带自动 train/val/test 划分)
        # -------------------------------------------------------------
        run_command(f"{base_cmd} src/models/session_graphmae/extract_graph_embeddings.py "
                    f"--bert_ckpt {bert_ckpt_path} "
                    f"--raw_data_dir {abs_base_dir} "
                    f"--output_dir {graph_data_dir} "
                    f"--dataset_config {args.dataset_config}",
                    "2. 提取图节点融合表征 (带数据划分)")

        # -------------------------------------------------------------
        # 阶段 3：预训练 GraphMAE
        # -------------------------------------------------------------
        run_command(f"{base_cmd} src/models/session_graphmae/train_graphmae.py "
                    f"--train_data_dir {os.path.join(graph_data_dir, 'train')} "
                    f"--val_data_dir {os.path.join(graph_data_dir, 'val')} "
                    f"--test_data_dir {os.path.join(graph_data_dir, 'test')} "  # 🎯 核心新增：传入测试集
                    f"--output_dir {graphmae_ckpt_dir}",
                    "3. GraphMAE 图级别无监督/半监督预训练与最终测试")

    # =============================================================
    # 阶段 4：流式增量推断与适应
    # =============================================================
    run_command(f"{base_cmd} src/concept_drift_detect/run_experiment_session.py "
                f"--dataset_dir {abs_base_dir} "
                f"--model_ckpt {bert_ckpt_path}",
                "4. 全链路流式推断与漂移适应")

    logger.info("🎉 流水线全部执行完毕！")


if __name__ == "__main__":
    main()