import os
import glob
import torch
import logging
import argparse
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.session_graphmae.graphmae_model import SessionGraphMAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_graph_dataset(data_dir):
    """加载由 BERT 提取后保存的 .bin 图数据集"""
    logger.info(f"正在从 {data_dir} 加载图数据...")
    files = glob.glob(os.path.join(data_dir, "*.bin"))
    dataset = []
    for f in files:
        data = torch.load(f)
        dataset.append(data)
    logger.info(f"成功加载 {len(dataset)} 个图样本。")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train GraphMAE for Session Graph Representation")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练集目录")
    parser.add_argument("--val_data_dir", type=str, required=True, help="验证集目录")
    parser.add_argument("--test_data_dir", type=str, required=True, help="测试集目录")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="最大训练轮数")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--output_dir", type=str, required=True, help="模型保存路径")
    args = parser.parse_args()

    train_dataset = load_graph_dataset(args.train_data_dir)
    val_dataset = load_graph_dataset(args.val_data_dir)
    test_dataset = load_graph_dataset(args.test_data_dir)  # 🎯 加载测试集

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)  # 🎯 测试集DataLoader

    model = SessionGraphMAE(
        in_dim=768, hidden_dim=128, enc_layers=4, dec_layers=4, mask_rate=0.5, lr=args.lr
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"💾 GraphMAE 最佳模型将保存在: {args.output_dir}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='best_model',
        save_top_k=1,
        monitor='val_is_mal_f1',  # 监控指标可以根据你的需要改，比如 train_loss
        mode='max'
    )

    early_stopping = EarlyStopping(monitor='train_loss', patience=10, mode='min')
    tb_logger = TensorBoardLogger(save_dir="tb_logs", name="session_graphmae")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        devices="auto",
        accelerator="auto",
        log_every_n_steps=10
    )

    logger.info("🚀 开始训练 GraphMAE 模型...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("✅ GraphMAE 训练完成！")

    logger.info("🔍 正在独立的 测试集 (Test Set) 上生成最终评估报告...")
    # 🎯 核心修改：使用 test_loader 进行真正的测试
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()