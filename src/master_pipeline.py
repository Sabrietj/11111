import os
import sys
import subprocess
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🚀 %(message)s')
logger = logging.getLogger(__name__)

# 获取项目根目录 (假设该脚本在 src 目录下，向上一级即为项目根目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd, step_name):
    """执行子进程并捕获错误"""
    logger.info(f"开始执行阶段: 【{step_name}】")
    logger.info(f"执行命令: {cmd}")

    # 执行命令，如果失败则抛出异常并终止流水线
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        logger.error(f"❌ 阶段【{step_name}】执行失败，流水线终止！")
        sys.exit(1)

    logger.info(f"✅ 阶段【{step_name}】执行成功！\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MultiViewBert-GraphMAE 统一实验流水线")
    parser.add_argument("--mode", type=str, choices=["train_all", "test_only"], default="train_all",
                        help="运行模式：train_all (执行完整预训练+测试) 或 test_only (直接加载最佳模型测试)")

    args = parser.parse_args()

    # 🌟 核心改进：使用 sys.executable 自动获取当前激活的 Python 解释器绝对路径
    # 彻底告别写死的 conda 环境名，避免环境迁移导致的运行失败
    base_cmd = f"{sys.executable}"

    logger.info(f"🌟 欢迎使用 MultiViewBert-GraphMAE 统一流水线 🌟")
    logger.info(f"当前模式: {args.mode}")
    logger.info(f"当前 Python 环境: {base_cmd}")
    logger.info("=" * 60)

    if args.mode == "train_all":
        # -------------------------------------------------------------
        # 阶段 1：预训练流级别的 FlowBertMultiview
        # -------------------------------------------------------------
        run_command(f"{base_cmd} src/models/flow_bert_multiview/train.py",
                    "1. FlowBert 流级别预训练")

        # -------------------------------------------------------------
        # 阶段 3：使用预训练的 FlowBert 离线提取 768 维节点特征图
        # -------------------------------------------------------------
        run_command(f"{base_cmd} src/models/session_graphmae/extract_graph_embeddings.py",
                    "3. 提取图节点融合表征 (Graph Embeddings)")

        # -------------------------------------------------------------
        # 阶段 4：预训练图级别的 GraphMAE (GIN)
        # -------------------------------------------------------------
        run_command(f"{base_cmd} src/models/session_graphmae/train_graphmae.py",
                    "4. GraphMAE 图级别无监督/半监督预训练")

    # =============================================================
    # 阶段 5：流式增量联合推断与适应 (Test Only 也会直接跳到这里)
    # 🌟 核心改进：指向了包含图缓存逻辑的 run_experiment_session.py
    # =============================================================
    run_command(f"{base_cmd} src/concept_drift_detect/run_experiment_session.py",
                "5. 全链路流式推断、漂移适应与最终成绩单输出")

    logger.info("🎉 流水线全部执行完毕！你可以去查看终端的测试成绩或运行下游评估脚本了。")


if __name__ == "__main__":
    main()