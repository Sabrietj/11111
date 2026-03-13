import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
import os,sys
import ast
from typing import List, Dict, Any, Tuple
import re
from tqdm import tqdm  # 添加tqdm导入
from collections import Counter
from .flow_bert_multiview_dataset import MultiviewFlowDataset

# 导入配置管理器和相关模块
try:
    # 添加../../utils目录到Python搜索路径
    utils_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils')
    sys.path.insert(0, utils_path)    
    import config_manager as ConfigManager
    from logging_config import setup_preset_logging
    # 使用统一的日志配置
    logger = setup_preset_logging(log_level=logging.INFO)

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目结构正确，所有依赖模块可用")
    sys.exit(1)

class MultiviewFlowDataModule(pl.LightningDataModule):
    """多视图流量数据模块"""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # 缓存常用配置
        self.flow_data_path = self.cfg.data.flow_data_path
        self.session_split_path = self.cfg.data.session_split.session_split_path
        self.batch_size = self.cfg.data.batch_size
        self.num_workers = self.cfg.data.num_workers

        # 缓存会话划分配置
        self.split_config = cfg.data.session_split
        self.split_column = self.split_config.split_column
        self.flow_uid_list_column = self.split_config.flow_uid_list_column
        self.train_split = self.split_config.train_split
        self.validate_split = self.split_config.validate_split
        self.test_split = self.split_config.test_split

        self.train_dataset = None
        self.validate_dataset = None
        self.test_dataset = None

        # 其他配置
        self.is_malicious_column = cfg.data.is_malicious_column
        self.multiclass_label_column = cfg.data.multiclass_label_column
        self.debug_mode = cfg.debug.debug_mode
                
        
    def prepare_data(self):
        # 下载或准备数据
        pass
    
    def setup(self, stage=None):
        # 处理不同的stage输入类型
        if stage is None:
            stage_name = "fit" # "fit" 是 Lightning 的规范名称，含义是 “训练 + 验证”
        elif hasattr(stage, 'value'):  # 如果是枚举类型
            stage_name = stage.value.lower()  # 直接使用枚举的value属性
        elif isinstance(stage, str):
            stage_name = stage.lower()
        else:
            # 其他情况，尝试转换为字符串
            stage_name = str(stage).lower()
        
        # 检查是否已经初始化
        if self._is_already_setup(stage_name):
            logger.info(f"{stage_name} 阶段数据集已初始化，跳过重复setup")
            return

        logger.info(f"数据模块 setup 阶段: {stage_name}")        

        # 一次性读取所有必要数据（支持缓存）
        self._load_data_with_cache(stage_name)

        if stage_name == "fit":
            self._create_datasets(stage_name)

        if stage in (None, "fit") and self.train_dataset is not None and self.validate_dataset is not None:
            train_is_malicious_labels = self.train_dataset.is_malicious_labels
            val_is_malicious_labels = self.validate_dataset.is_malicious_labels
            logger.info("==========================================")            
            logger.info(f"[训练集] 正样本={sum(train_is_malicious_labels)}, 负样本={len(train_is_malicious_labels)-sum(train_is_malicious_labels)}, 比例={sum(train_is_malicious_labels)/len(train_is_malicious_labels):.4f}")
            logger.info(f"[验证集] 正样本={sum(val_is_malicious_labels)}, 负样本={len(val_is_malicious_labels)-sum(val_is_malicious_labels)}, 比例={sum(val_is_malicious_labels)/len(val_is_malicious_labels):.4f}")

        if stage in (None, "test") and self.test_dataset is not None:
            test_is_malicious_labels = self.test_dataset.is_malicious_labels
            logger.info("==========================================")            
            logger.info(f"[测试集] 正样本={sum(test_is_malicious_labels)}, 比例={sum(test_is_malicious_labels)/len(test_is_malicious_labels):.4f}")


    def _is_already_setup(self, stage_name):
        """检查数据集是否已经初始化"""
        if stage_name == 'fit':
            # fit阶段需要训练集和验证集都初始化
            return self.train_dataset is not None and self.validate_dataset is not None
        elif stage_name == 'validate':
            # validate阶段只需要验证集
            return self.validate_dataset is not None
        elif stage_name == 'test':
            # test阶段只需要测试集
            return self.test_dataset is not None
        else:
            # 其他情况返回False
            return False

    def read_large_csv_with_progress(self, filepath, description="读取数据", verbose=True):
        if not verbose:
            return pd.read_csv(filepath)

        logger.info(f"{description} 从 {filepath}...")
        file_size = os.path.getsize(filepath) / (1024**3)
        logger.info(f"文件大小: {file_size:.2f} GB")

        sample_df = pd.read_csv(filepath, nrows=5)
        logger.info(f"检测到 {len(sample_df.columns)} 列，开始分块读取...")

        with open(filepath, "r") as f:
            total_rows = sum(1 for _ in f) - 1

        chunk_size = 100_000  # 每次读取10万行
        chunks = []

        # ⭐ 关键：只有 rank 0 才开 tqdm，否则每个gpu都打印进度条，会显示错乱
        if self.trainer is not None:
            is_rank_zero = self.trainer.is_global_zero
        else:
            is_rank_zero = True  # 非 Lightning 场景兜底

        pbar = tqdm(
            total=total_rows,
            desc=description,
            unit="rows",
            dynamic_ncols=True,
            leave=True,
            disable=not is_rank_zero,   # ⭐⭐ 关键
        )

        for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
            pbar.update(len(chunk))

        pbar.close()

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"{description} 完成! 数据形状: {df.shape}")
        return df
    
    def _load_data_with_cache(self, stage_name):
        """带缓存的数据加载"""
        # 如果flow_df已经存在，直接使用缓存的数据
        if hasattr(self, 'flow_df') and self.flow_df is not None:
            logger.info("使用缓存的flow_df数据，跳过重复读取")
        else:
            # 只在第一次需要时读取数据
            flow_df = self.read_large_csv_with_progress(self.flow_data_path)
            # 增强数据质量检查
            self.flow_df = self._validate_data_quality(flow_df, stage=stage_name)  # ✅ 正确接收修改后的flow_df

        # 如果session_df已经存在，直接使用缓存的数据
        if hasattr(self, 'session_df') and self.session_df is not None:
            logger.info("使用缓存的session_df数据，跳过重复读取")
        else:
            # 只在第一次需要时读取session数据
            self.session_df = self.read_large_csv_with_progress(self.session_split_path)

            # 检查必要的列是否存在
            required_columns = [self.split_column, self.flow_uid_list_column]
            missing_columns = [col for col in required_columns if col not in self.session_df.columns]
            if missing_columns:
                raise ValueError(f"session_df缺少必要列: {missing_columns}")
            
            # 检查split值的有效性
            valid_splits = [self.train_split, self.validate_split, self.test_split]
            actual_splits = self.session_df[self.split_column].unique()
            invalid_splits = [split for split in actual_splits if split not in valid_splits]
            if invalid_splits:
                logger.warning(f"发现无效的split值: {invalid_splits}")
        
        
    def _validate_data_quality(self, df, stage="unknown"):
        """仅校验配置文件中使用到的列是否包含 NaN，不检查其它无关列。
        若 target 列存在 NaN，立即抛异常，要求用户处理。"""

        logger.info(f"检查 {stage} 阶段数据质量（仅检查 cfg 中引用的列）...")

        # ============ 1. 收集所有需要检查的列 ============
        required_columns = set()

        # is_malicious 和 multiclass_label 标签列（最关键）
        if self.is_malicious_column is not None:
            required_columns.add(self.is_malicious_column)
        if self.multiclass_label_column is not None:
            required_columns.add(self.multiclass_label_column)

        # 数值特征列
        if hasattr(self.cfg.data.tabular_features.numeric_features, "flow_features"):
            required_columns.update(self.cfg.data.tabular_features.numeric_features.flow_features)

        # 序列特征列（可选）
        if hasattr(self.cfg.data, "sequence_features") and self.cfg.data.sequence_features is not None:
            seq_cfg = self.cfg.data.sequence_features
            required_columns.update([
                seq_cfg.packet_direction,
                seq_cfg.packet_iat,
                seq_cfg.packet_payload,
            ])

        # 文本特征列（可选）
        if hasattr(self.cfg.data, "text_features") and self.cfg.data.text_features is not None:
            txt_cfg = self.cfg.data.text_features
            required_columns.update([
                txt_cfg.ssl_server_name,
                txt_cfg.dns_query,
                txt_cfg.cert0_subject,
                txt_cfg.cert0_issuer,
            ])

        # 域名嵌入特征列（可选）
        if hasattr(self.cfg.data, "domain_name_embedding_features"):
            if hasattr(self.cfg.data.domain_name_embedding_features, "column_list"):
                required_columns.update(self.cfg.data.domain_name_embedding_features.column_list)

        required_columns = list(required_columns)

        # ============ 2. 必须存在的列检查 ============
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ 数据缺少配置文件要求的列: {missing_cols}")

        # ============ 3. 只检查这些列中的 NaN ============
        df_sub = df[required_columns]
        nan_counts = df_sub.isna().sum()

        # 全部使用的列是否有 NaN
        if nan_counts.any():
            nan_cols = nan_counts[nan_counts > 0]

            # ---- 特殊处理: is_malicious 列出现 NaN → 直接报错 ----
            if nan_counts[self.is_malicious_column] > 0:
                raise ValueError(
                    f"❌ is_malicious 列 '{self.is_malicious_column}' 出现 NaN 值: {nan_counts[self.is_malicious_column]} 行。\n"
                    f"请在加载 CSV 前手动清洗数据，否则模型无法训练。"
                )

            # 其它列的 NaN → 给 warning，让用户处理
            logger.warning("⚠️ 以下使用到的特征列包含 NaN：")
            for col, count in nan_cols.items():
                logger.warning(f"  {col}: {count} 个 NaN ({count / len(df) * 100:.2f}%)")

        logger.info(f"{stage} 阶段数据质量检查完成（仅检查 cfg 使用的列）")
        return df

    def _create_datasets(self, stage_name):
        split_mode = self.cfg.data.get("split_mode", "session").lower()
        logger.info(f"数据划分模式：{split_mode}")

        if split_mode == "session":
            logger.info("使用基于 session_df 的会话级划分策略")
            assert self.split_column in self.session_df.columns
            assert "uid" in self.flow_df.columns

            # 使用session_df会话进行数据集划分
            train_session_df = self.session_df[self.session_df[self.split_column] == self.train_split]
            validate_session_df = self.session_df[self.session_df[self.split_column] == self.validate_split]
            test_session_df = self.session_df[self.session_df[self.split_column] == self.test_split]

            # 提取所有训练flow的UID
            train_flow_uids = []
            for uid_list in train_session_df[self.flow_uid_list_column]:
                if pd.notna(uid_list) and uid_list != '[]':
                    try:
                        uids = ast.literal_eval(uid_list) if isinstance(uid_list, str) else uid_list
                        train_flow_uids.extend(uids)
                    except:
                        continue

            # 提取所有验证flow的UID
            validate_flow_uids = []
            for uid_list in validate_session_df[self.flow_uid_list_column]:
                if pd.notna(uid_list) and uid_list != '[]':
                    try:
                        uids = ast.literal_eval(uid_list) if isinstance(uid_list, str) else uid_list
                        validate_flow_uids.extend(uids)
                    except:
                        continue

            # 提取所有测试flow的UID
            test_flow_uids = []
            for uid_list in test_session_df[self.flow_uid_list_column]:
                if pd.notna(uid_list) and uid_list != '[]':
                    try:
                        uids = ast.literal_eval(uid_list) if isinstance(uid_list, str) else uid_list
                        test_flow_uids.extend(uids)
                    except:
                        continue

            # 根据UID划分flow数据集
            train_flow_df = self.flow_df[self.flow_df['uid'].isin(train_flow_uids)]
            validate_flow_df = self.flow_df[self.flow_df['uid'].isin(validate_flow_uids)]
            test_flow_df = self.flow_df[self.flow_df['uid'].isin(test_flow_uids)]

        elif split_mode == "flow":
            
            logger.info("使用基于 flow_df 的随机逐流划分策略")

            self.flow_train_ratio = self.cfg.data.flow_split.train_ratio
            self.flow_validate_ratio = self.cfg.data.flow_split.validate_ratio
            self.flow_test_ratio = self.cfg.data.flow_split.test_ratio

            is_malicious_labels = self.flow_df[self.is_malicious_column].values

            from sklearn.model_selection import train_test_split

            train_df, temp_df = train_test_split(
                self.flow_df,
                test_size=1 - self.flow_train_ratio,
                stratify=is_malicious_labels,
                random_state=42
            )

            temp_is_malicious_labels = temp_df[self.is_malicious_column].values
            val_ratio_in_temp = self.flow_validate_ratio / (self.flow_validate_ratio + self.flow_test_ratio)

            validate_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_ratio_in_temp,
                stratify=temp_is_malicious_labels,
                random_state=42
            )

            train_flow_df = train_df
            validate_flow_df = validate_df
            test_flow_df = test_df
            
        else:
            raise ValueError(
                f"❌ 无效的 split_mode='{split_mode}'。"
                f"请在配置文件中使用 'session' 或 'flow'"
            )

        if self.trainer is not None:
            global_rank = self.trainer.global_rank
            local_rank = self.trainer.local_rank
        else:
            global_rank = 0
            local_rank = 0
            
        # 创建训练集 - 传入完整的 cfg 对象
        logger.info(f">>>> [global_rank={global_rank}, local_rank={local_rank}] 创建训练集MultiviewFlowDataset ...")
        self.train_dataset = MultiviewFlowDataset(train_flow_df, self.cfg, is_training=True)
        
        # 创建验证集：在构造函数中注入训练集映射
        logger.info(f">>>> [global_rank={global_rank}, local_rank={local_rank}] 创建验证集MultiviewFlowDataset ...")
        self.validate_dataset = MultiviewFlowDataset(
            validate_flow_df, 
            self.cfg, 
            is_training=False,
            train_categorical_mappings=self.train_dataset.categorical_val2idx_mappings,
            train_categorical_columns_effective=self.train_dataset.categorical_columns_effective
        )
        
        # 创建测试集：在构造函数中注入训练集映射
        logger.info(f">>>> [global_rank={global_rank}, local_rank={local_rank}] 创建测试集MultiviewFlowDataset ...")
        self.test_dataset = MultiviewFlowDataset(
            test_flow_df, 
            self.cfg, 
            is_training=False,
            train_categorical_mappings=self.train_dataset.categorical_val2idx_mappings,
            train_categorical_columns_effective=self.train_dataset.categorical_columns_effective
        )

        self._validate_categorical_consistency()
        
        # 应该确保验证集使用训练集的统计信息
        if hasattr(self.train_dataset, 'numeric_stats'):
            self.validate_dataset.numeric_stats = self.train_dataset.numeric_stats  # ✅ 传递统计信息
            self.test_dataset.numeric_stats = self.train_dataset.numeric_stats  # ✅ 传递统计信息
            
            # 重新应用归一化
            logger.info(f"✅ 重新应用数值特征标准化: 验证集使用训练集的统计信息")
            logger.info(f"   覆盖特征数量: {len(self.train_dataset.numeric_stats)}")
            self.validate_dataset.apply_numeric_stats()
            
            logger.info(f"✅ 重新应用数值特征标准化: 测试集使用训练集的统计信息")
            logger.info(f"   覆盖特征数量: {len(self.train_dataset.numeric_stats)}")
            self.test_dataset.apply_numeric_stats()            
            
        else:
            logger.info("🔄 使用默认统计信息进行数值特征标准化")
            # 为每个数值列创建默认统计信息
            flow_columns = self.cfg.data.tabular_features.numeric_features.flow_features
            default_stats = {}
            for col in flow_columns:
                default_stats[col] = {'mean': 0, 'std': 1}
            
            self.validate_dataset.numeric_stats = default_stats
            self.test_dataset.numeric_stats = default_stats
            logger.info(f"   默认统计信息覆盖特征数量: {len(default_stats)}")
        
        logger.info(f"数据集划分: 训练集 {len(train_flow_df)}, 验证集 {len(validate_flow_df)}, 测试集 {len(test_flow_df)}")

    # 验证映射一致性
    def _validate_categorical_consistency(self):
        """验证所有数据集的类别映射一致性"""
        train_mappings = self.train_dataset.categorical_val2idx_mappings
        val_mappings = self.validate_dataset.categorical_val2idx_mappings
        test_mappings = self.test_dataset.categorical_val2idx_mappings
        
        assert train_mappings == val_mappings == test_mappings, "类别映射不一致！"
        logger.info("✅ 所有数据集的类别映射验证一致")
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.validate_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )