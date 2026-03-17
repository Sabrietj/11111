import os
import sys
import glob
import torch
import logging
import argparse
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# ============================================================================
# 🚨 暴力屏蔽第三方库的 DEBUG 刷屏日志，还你一个清净的终端
# ============================================================================
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("fsspec.local").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# ============================================================================
# 🚨 路径修复：确保项目根目录在系统路径中
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ============================================================================

from src.models.session_graphmae.graphmae_model import SessionGraphMAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_id_to_label_map(dataset_name="cic_iomt_2024"):
    id_to_label = {}
    try:
        from src.utils import config_manager as ConfigManager
        if hasattr(ConfigManager, 'get_config'):
            config = ConfigManager.get_config()
            if dataset_name in config and 'session_label_id_map' in config[dataset_name]:
                label_to_id = config[dataset_name]['session_label_id_map']
                id_to_label = {int(v): str(k) for k, v in label_to_id.items()}
                return id_to_label

        if hasattr(ConfigManager, 'session_label_id_map'):
            label_to_id = ConfigManager.session_label_id_map
            id_to_label = {int(v): str(k) for k, v in label_to_id.items()}
            return id_to_label

    except Exception as e:
        pass

    id_to_label = {
        0: "Benign",
        8: "malicious_Recon-OS_Scan",
        9: "malicious_Recon-VulScan",
        12: "malicious_TCP_IP-DDoS-ICMP",
        15: "malicious_TCP_IP-DDoS-UDP",
        16: "malicious_TCP_IP-DoS-ICMP",
        19: "malicious_TCP_IP-DoS-UDP",
        20: "malicious_MQTT-DDoS-Publish_Flood",
        23: "malicious_MQTT-DoS-Connect_Flood"
    }
    return id_to_label


def load_graph_dataset(data_dir):
    logger.info(f"正在从 {data_dir} 加载图数据...")
    pack_path = os.path.join(data_dir, "graphs.pt")
    if os.path.exists(pack_path):
        logger.info(f"⚡ 发现极速打包文件 {pack_path}，正在载入内存...")
        dataset = torch.load(pack_path, weights_only=False)
        logger.info(f"✅ 成功加载 {len(dataset)} 个纯净图样本。")
        return dataset

    logger.warning("未找到 graphs.pt，尝试读取散装 .bin 文件...")
    files = glob.glob(os.path.join(data_dir, "*.bin"))
    files = [f for f in files if os.path.basename(f) != "graphs.bin"]
    dataset = []
    for f in files:
        try:
            data = torch.load(f, weights_only=False)
            dataset.append(data)
        except Exception as e:
            pass

    logger.info(f"✅ 成功加载 {len(dataset)} 个纯净图样本。")
    return dataset


def get_num_classes(dataset):
    max_cls = 0
    for data in dataset:
        if hasattr(data, 'y_multi'):
            val = data.y_multi.item()
            if val > max_cls:
                max_cls = val
    return int(max_cls) + 1


def main():
    parser = argparse.ArgumentParser(description="Train GraphMAE for Session Graph Representation")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练集图数据存放目录")
    parser.add_argument("--val_data_dir", type=str, required=True, help="验证集图数据存放目录")
    parser.add_argument("--test_data_dir", type=str, default=None, help="测试集图数据目录")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (大数据集建议调大)")
    parser.add_argument("--epochs", type=int, default=20, help="最大训练轮数")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--output_dir", type=str, default="checkpoints/graphmae", help="模型保存路径")
    args = parser.parse_args()

    train_dataset = load_graph_dataset(args.train_data_dir)
    val_dataset = load_graph_dataset(args.val_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_classes = get_num_classes(train_dataset)
    logger.info(f"🎯 自动探测到多分类最大类别 ID 为: {num_classes - 1}，分类器维度: {num_classes}")

    id_to_label_map = get_id_to_label_map()

    model = SessionGraphMAE(
        num_attack_families=num_classes,
        lr=args.lr,
        id_to_label_map=id_to_label_map
    )

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='graphmae-best-{epoch:02d}-{train_loss:.4f}',
        save_top_k=1,
        monitor='train_loss',
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='train_loss',
        patience=5,
        mode='min'
    )

    tb_logger = TensorBoardLogger(save_dir="tb_logs", name="session_graphmae")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        devices=1,
        accelerator="gpu",
        log_every_n_steps=10
    )

    logger.info("🚀 开始训练 GraphMAE 模型...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("✅ GraphMAE 训练完成！")

    logger.info("📊 正在生成最终的评估报告...")
    model.print_final_report = True
    trainer.validate(model, dataloaders=val_loader, ckpt_path="best")


if __name__ == "__main__":
    main()