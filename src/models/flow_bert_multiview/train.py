import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import timedelta
from pytorch_lightning.strategies import DDPStrategy

from data.flow_bert_multiview_datamodule import MultiviewFlowDataModule
from models.flow_bert_multiview import FlowBertMultiview
from data.prepare_flow_file_pipeline import prepare_sampled_data_files

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import sys
import logging
from pytorch_lightning import seed_everything

import os
utils_path = os.path.join(os.path.dirname(__file__),  '..', '..', 'utils')
sys.path.insert(0, utils_path) 
# 设置日志
from logging_config import setup_preset_logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.INFO)

def setup_environment(cfg: DictConfig):
    """根据硬件可用性自动设置训练环境"""
    
    # 移除不必要的TensorFlow环境变量设置
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 移除这行
    # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 移除这行

    force_single_gpu = cfg.get('force_single_gpu', False)

    # ✅ 必须最先做：强制单 GPU（在任何 torch.cuda 调用之前）
    if force_single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ["PYTORCH_MPS_DISABLE"] = "1"  # 禁用MPS以避免潜在冲突
        cfg.data.num_workers = 0  # 强制单GPU时，设置num_workers为0以避免 DataLoader 多进程复杂性
        logger.info("✅ 强制使用单 GPU 训练，已设置 CUDA_VISIBLE_DEVICES=0 和 PYTORCH_MPS_DISABLE=1")  

    # 使用正确的精度设置API
    torch.set_float32_matmul_precision('high')  # 推荐使用 'high'

    # 设置确定性操作（有助于调试）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 检测CUDA可用性
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        logger.info(f"检测到 {cuda_count} 个CUDA设备可用")

        for i in range(torch.cuda.device_count()):
            logger.info(f"设备 {i}: {torch.cuda.get_device_name(i)}")
        
        os.environ['PYTORCH_MPS_DISABLE'] = '1'  # 禁用MPS
        
        accelerator = "gpu"
        if force_single_gpu:
            devices = 1
            strategy = "auto"
            logger.info("✅ 强制使用单 GPU 训练")
        else:
            devices = torch.cuda.device_count()
            # 确保 strategy 有值
            if devices > 1:
                ddp_timeout = timedelta(hours=6)
                strategy = DDPStrategy(
                    find_unused_parameters=True,    # 启用未使用参数检测
                    timeout=ddp_timeout,
                )  
                logger.info("使用多GPU DDP策略，启用未使用参数检测")
            else:
                strategy = "auto"
                logger.info("使用单GPU 训练策略")
        
        logger.info("使用GPU进行训练")

    else:
        # 检测MPS可用性
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("检测到MPS（Apple Silicon）可用")
            accelerator = "mps"
            devices = 1
            strategy = "auto"
            os.environ['PYTORCH_MPS_DISABLE'] = '0'  # 启用MPS
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 启用MPS回退
            logger.info("使用MPS进行训练")
        else:
            logger.info("未检测到GPU/MPS，使用CPU进行训练")
            accelerator = "cpu"
            devices = 1
            strategy = "auto"
    
    return accelerator, devices, strategy


def validate_config(cfg: DictConfig):
    """验证配置完整性"""
    required_fields = {
        'data.tabular_features': dict,
        'data.tabular_features.numeric_features.flow_features': list,
        'data.tabular_features.numeric_features.x509_features': list,
        'data.tabular_features.numeric_features.dns_features': list,         
        'data.is_malicious_column': str,
        'data.multiclass_label_column': str,
        'data.text_features.enabled': bool,
        'data.domain_name_embedding_features.enabled': bool,
        'data.domain_name_embedding_features.column_list': list,
        'data.sequence_features.enabled': bool,
        'model.sequence.embedding_dim': int,
        'model.bert.model_name': str,
        'model.multiview.fusion.method': str,        
    }

    # 可选字段 - 修正期望类型
    optional_fields = {
        'data.sequence_features': dict,  # 正确：期望字典类型
        'data.text_features': dict,      # 正确：期望字典类型  
        'data.domain_name_embedding_features': dict  # 正确：期望列表类型
    }
    
    # 验证必需字段
    for field, expected_type in required_fields.items():
        value = OmegaConf.select(cfg, field)
        if value is None:
            raise ValueError(f"缺少必要配置: {field}")
        
        # 处理不同类型的验证
        if expected_type == list:
            if not isinstance(value, (list, ListConfig)):
                raise ValueError(f"配置 {field} 类型错误，期望列表类型，实际 {type(value)}")
            if len(value) == 0:
                raise ValueError(f"配置 {field} 不能为空列表")
                
        elif expected_type == dict:
            if not isinstance(value, (dict, DictConfig)):
                raise ValueError(f"配置 {field} 类型错误，期望字典类型，实际 {type(value)}")
            if len(value) == 0:
                raise ValueError(f"配置 {field} 不能为空字典")
                
        elif expected_type == str:
            if not isinstance(value, str):
                raise ValueError(f"配置 {field} 类型错误，期望字符串类型，实际 {type(value)}")
            if not value.strip():
                raise ValueError(f"配置 {field} 不能为空字符串")
                
        elif expected_type == int:
            if not isinstance(value, int):
                raise ValueError(f"配置 {field} 类型错误，期望整数类型，实际 {type(value)}")
            if value <= 0:
                raise ValueError(f"配置 {field} 必须是正整数")
    
    # 验证可选字段 - 修正验证逻辑
    for field, expected_type in optional_fields.items():
        value = OmegaConf.select(cfg, field)
        if value is not None:  # 只有当字段存在时才验证
            # 处理不同类型的验证
            if expected_type == list:
                if not isinstance(value, (list, ListConfig)):
                    raise ValueError(f"配置 {field} 类型错误，期望列表类型，实际 {type(value)}")
                # 可选字段允许空列表
                
            elif expected_type == dict:
                if not isinstance(value, (dict, DictConfig)):
                    raise ValueError(f"配置 {field} 类型错误，期望字典类型，实际 {type(value)}")
                # 可选字段允许空字典
                
            elif expected_type == str:
                if not isinstance(value, str):
                    raise ValueError(f"配置 {field} 类型错误，期望字符串类型，实际 {type(value)}")
                if not value.strip():
                    raise ValueError(f"配置 {field} 不能为空字符串")
                    
            elif expected_type == int:
                if not isinstance(value, int):
                    raise ValueError(f"配置 {field} 类型错误，期望整数类型，实际 {type(value)}")
                if value <= 0:
                    raise ValueError(f"配置 {field} 必须是正整数")

    if not hasattr(cfg.optimizer, 'default_total_steps'):
        logger.warning("optimizer.default_total_steps 未配置，将使用默认值10000")
        cfg.optimizer.default_total_steps = 10000

    # 验证融合方法配置
    fusion_method = cfg.model.multiview.fusion.method
    
    if fusion_method == "cross_attention":
        if not hasattr(cfg.model.multiview.fusion, 'cross_attention_heads'):
            raise ValueError("交叉注意力融合需要配置 cross_attention_heads")
        if not hasattr(cfg.model.multiview.fusion, 'cross_attention_dropout'):
            raise ValueError("交叉注意力融合需要配置 cross_attention_dropout")
    
    elif fusion_method == "weighted_sum":
        if not hasattr(cfg.model.multiview.fusion, 'weighted_sum'):
            raise ValueError("加权求和融合需要配置 weighted_sum")
    
    elif fusion_method == "concat":
        # 🔴 拼接方法不再需要额外配置验证
        pass  # 拼接方法不需要额外配置
    
    else:
        raise ValueError(f"不支持的融合方法: {fusion_method}")
    
    # 额外的配置逻辑验证
    logger.info("配置验证通过")


def resolve_dataset_paths(cfg):
    dataset_cfg = cfg.datasets

    # 写回统一使用的字段
    assert "flow_data_path" in dataset_cfg
    cfg.data.flow_data_path = dataset_cfg.flow_data_path
    
    assert "session_split_path" in dataset_cfg
    cfg.data.session_split.session_split_path = dataset_cfg.session_split_path

    logger.info(
        f"📦 使用数据集: {cfg.data.dataset}\n"
        f"  flow_data_path = {cfg.data.flow_data_path}\n"
        f"  session_split_path = {cfg.data.session_split.session_split_path}"
    )

@hydra.main(config_path="./config", config_name="flow_bert_multiview_config", version_base=None)
def main(cfg: DictConfig):
    os.environ['HYDRA_FULL_ERROR'] = '1'

    logger = setup_preset_logging(log_level=logging.INFO)
    
    try:
        # ------------------------------------------------------------------
        # 0. 基础环境 & 可复现性
        # ------------------------------------------------------------------
        seed_everything(cfg.data.random_state, workers=True)
        os.chdir(hydra.utils.get_original_cwd())

        accelerator, devices, strategy = setup_environment(cfg)

        # logger.info("多视图BERT-Multiview训练配置:")
        # logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

        # 打印配置以调试
        # logger.info("BERT模型名称:", cfg.model.bert.model_name)

        # 添加配置验证
        validate_config(cfg)

        # 选择数据集配置
        resolve_dataset_paths(cfg)

        # sample flow data if needed
        prepare_sampled_data_files(cfg)
        logger.info(f"✅ 最终用于训练的 flow_data_path = {cfg.data.flow_data_path}")

        # ------------------------------------------------------------------
        # 1. DataModule（只加载，不构建）
        # ------------------------------------------------------------------
        datamodule = MultiviewFlowDataModule(cfg)
        datamodule.setup("fit")   # ⭐ 先构建数据

        # ------------------------------------------------------------------
        # 2. Model
        # ------------------------------------------------------------------
        model = FlowBertMultiview(cfg, dataset=datamodule.train_dataset)

        # ------------------------------------------------------------------
        # 3. Callbacks
        # ------------------------------------------------------------------
        callbacks = []
        
        # ------------------------------------------------------------------
        # 4. Logger
        # ------------------------------------------------------------------
        logger_tb = TensorBoardLogger(
            save_dir=cfg.logging.save_dir,
            name=cfg.logging.name,
            version=cfg.logging.get('version')
        )

        base_data_dir = os.path.dirname(cfg.data.session_split.session_split_path)

        ckpt_dir = os.path.join(base_data_dir, "saved_models", "flow_bert")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.info(f"💾 MultiViewBert 最佳模型将保存在: {ckpt_dir}")

        # 🎯 固定文件名 best_model.ckpt，每次性能提升直接覆盖替换
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best_model",
            monitor=cfg.training.model_checkpoint.monitor,
            mode=cfg.training.model_checkpoint.mode,
            save_top_k=1,
            auto_insert_metric_name=False
        )

        early_stopping = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            min_delta=cfg.training.early_stopping.get('min_delta', 0.001)
        )

        # ------------------------------------------------------------------
        # 5. Trainer
        # ------------------------------------------------------------------
        # 🔧 确保detect_anomaly配置生效
        detect_anomaly = cfg.training.get('detect_anomaly', False)
        logger.info(f"设置模型训练的异常检测开关: detect_anomaly = {detect_anomaly}")
        
        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=cfg.training.precision,
            gradient_clip_val=cfg.training.gradient_clip_val,
            gradient_clip_algorithm=cfg.training.get('gradient_clip_algorithm', 'norm'),            
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            log_every_n_steps=cfg.training.log_every_n_steps,
            logger=logger_tb,
            callbacks=[checkpoint_callback, early_stopping, LearningRateMonitor(logging_interval='step')],
            strategy=strategy,
            enable_progress_bar=True,
            detect_anomaly=detect_anomaly,
        )

        # ------------------------------------------------------------------
        # 6. Train
        # ------------------------------------------------------------------
        logger.info("开始训练多视图BERT模型...")

        # 💥 新增的断点恢复逻辑 💥
        resume_ckpt_path = os.path.join(ckpt_dir, "best_model-v2.ckpt")

        if os.path.exists(resume_ckpt_path):
            logger.info(f"💥 检测到断点文件，尝试从 {resume_ckpt_path} 恢复训练...")
            try:
                # 传入 ckpt_path 进行恢复
                trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt_path)
            except Exception as e:
                logger.warning(f"⚠️ 从断点恢复失败，可能文件损坏或配置不匹配。错误: {e}")
                logger.info("🌱 退回全新训练...")
                trainer.fit(model, datamodule=datamodule)
        else:
            logger.info("🌱 未检测到断点，开始全新训练...")
            trainer.fit(model, datamodule=datamodule)

        logger.info("训练完成！")
        
        logger.info("模型保存验证:")
        logger.info(f"最佳模型路径: {checkpoint_callback.best_model_path}")
        logger.info(f"最佳模型分数: {checkpoint_callback.best_model_score}")

        # 列出所有保存的模型
        if hasattr(checkpoint_callback, 'best_k_models'):
            logger.info("保存的top-k模型:")
            for path, score in checkpoint_callback.best_k_models.items():
                logger.info(f"  {path}: {score}")

        # ------------------------------------------------------------------
        # 7. Test（用最佳 checkpoint）
        # ------------------------------------------------------------------
        logger.info("开始测试（加载最佳模型）...")
        best_model_path = checkpoint_callback.best_model_path
        logger.info(f"最佳模型路径: {best_model_path}")
        best_model = FlowBertMultiview.load_from_checkpoint(
            best_model_path, 
            cfg=cfg,
            dataset=datamodule.train_dataset
        )

        # # 测试阶段用单卡，解决了 Lightning 的这个告警：
        # # Using DistributedSampler with the dataloaders during trainer.test()
        # # 原因你已经看到了：多卡 test的时候，Lightning 会自动用DistributedSampler，
        # # 为了 batch 对齐，会复制样本，导致：混淆矩阵不准、分类报告异常
        # trainer = pl.Trainer(
        #     accelerator="gpu" if torch.cuda.is_available() else "cpu",
        #     devices=1,
        #     logger=False,
        #     enable_checkpointing=False,
        # )
        trainer.test(best_model, datamodule=datamodule)
        
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() # pyright: ignore[reportCallIssue]
