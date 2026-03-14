import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from torchmetrics.classification import MultilabelF1Score

from .sequence_encoder import SequenceEncoder
from .multiview_feature_fusion import MultiViewFusionFactory, WeightedSumFusion, ConcatFusion, CrossAttentionFusion


# 注释掉原来的SHAP导入，使用新的通用框架
# import shap    # 可视化 added  by qinyf
from pathlib import Path
import json

# 导入配置管理器和相关模块
try:
    # 添加../../utils目录到Python搜索路径
    utils_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils')
    sys.path.insert(0, utils_path)
    import config_manager as ConfigManager
    from logging_config import setup_preset_logging
    # 使用统一的日志配置
    logger = setup_preset_logging(log_level=logging.INFO)
    from tensor_debugger import _safe_check_nan_in_tensor, _debug_tensor_for_nan_inf_count, _check_input_dict_for_nan
    from file_system import get_project_root
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目结构正确，所有依赖模块可用")
    sys.exit(1)

try:
    from src.concept_drift_detect.detectors import BNDMDetector, ADWINDetector
    from src.concept_drift_detect.adapter import IncrementalAdapter
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Concept Drift modules not found: {e}. Drift detection will be disabled.")
    BNDMDetector = None
    ADWINDetector = None
    IncrementalAdapter = None


# 导入新的通用SHAP分析框架
try:
    # 添加../../utils目录到Python搜索路径
    hyper_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'hyper_optimus')
    sys.path.insert(0, hyper_path)
    # from shap_analysis import SHAPAnalyzeMixin,ShapAnalyzer

    from shap_analysis import ShapAnalyzer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目结构正确，所有依赖模块可用")
    sys.exit(1)

# =============================================
# Safe Transformers Import（仅保留 BERT + schedulers）
# =============================================
try:
    from transformers import (
        BertModel,
        BertTokenizer,
        BertConfig,
        get_linear_schedule_with_warmup,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
    )
    _TRANSFORMERS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - optional dependency guard
    BertModel = BertTokenizer = BertConfig = None  # type: ignore[assignment]
    get_linear_schedule_with_warmup = None  # type: ignore[assignment]
    get_constant_schedule_with_warmup = None  # type: ignore[assignment]
    get_cosine_schedule_with_warmup = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc

try:
    from transformers import AdamW as _HFAdamW
except ImportError:  # pragma: no cover - optional dependency guard
    _HFAdamW = None

from torch.optim import AdamW as _TorchAdamW

AdamW = _HFAdamW or _TorchAdamW

def _require_transformers() -> None:
    """确保 transformers 已正确安装"""
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(
            "需要安装 transformers 才能使用 `FlowBertMultiview`，请运行 `pip install transformers`。"
        ) from _TRANSFORMERS_IMPORT_ERROR

def safe_mean(x: torch.Tensor):
    if x.numel() == 0:
        # DDP-safe + 数值稳定
        return x.sum() * 0.0
    else:
        return x.mean()

class DriftHandler:
    """
    处理概念漂移检测与适应的中间层
    """
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

        self.enabled = cfg.get("concept_drift", {}).get("enabled", False)

        self.detector = None
        self.adapter = None
        self.last_drift_result = (False, 0.0, {})
        self.drift_count = 0

        # 引入联合微调所需的双标签缓冲区
        self.buffer_features = []
        self.buffer_bin_labels = []
        self.buffer_mul_labels = []

        if self.enabled:
            self._init_components()

    def _init_components(self):
        from omegaconf import OmegaConf
        cd_cfg = self.cfg.concept_drift
        algo = cd_cfg.get('algorithm', 'bndm')

        det_params = {'seed': cd_cfg.get('seed', 2026)}

        if 'detectors' in cd_cfg and algo in cd_cfg.detectors:
            algo_cfg = OmegaConf.to_container(cd_cfg.detectors[algo], resolve=True)
            if 'max_tree_level' in algo_cfg:
                algo_cfg['max_level'] = algo_cfg.pop('max_tree_level')
            det_params.update(algo_cfg)

        logger.info(f"🛡️ 初始化漂移检测器: {algo.upper()} | 最终配置: {det_params}")

        if algo == 'bndm':
            if BNDMDetector is not None:
                self.detector = BNDMDetector(det_params)
        elif algo == 'adwin':
            if ADWINDetector is not None:
                self.detector = ADWINDetector(det_params)
        else:
            logger.warning(f"未知检测算法 {algo}，漂移检测未启用")
            self.enabled = False

        if 'adaptation' in cd_cfg and self.enabled:
            adapter_config = OmegaConf.to_container(cd_cfg.adaptation, resolve=True)
            task_mode = "multiclass" if self.model.attack_family_classifier is not None else "binary"
            if IncrementalAdapter is not None:
                self.adapter = IncrementalAdapter(self.model, adapter_config, task_mode=task_mode)
                logger.info(f"🔄 初始化增量适配器 (Mode: {task_mode}) | 配置: {adapter_config}")

    def update_batch(self, features: torch.Tensor, batch: dict, adapt: bool = False):
        if self.detector is None or not self.enabled:
            return False

        batch_drift_detected = False

        # 从 batch 中提取二分类与多分类的真实标签
        lbl_bin = batch['is_malicious_label'].view(-1)
        if 'attack_family_label' in batch:
            lbl_mul = batch['attack_family_label']
            if lbl_mul.dim() > 1:
                lbl_mul = torch.argmax(lbl_mul, dim=1)
        else:
            lbl_mul = torch.zeros_like(lbl_bin)

        for i in range(features.size(0)):
            feat = features[i]
            feat_input = feat.unsqueeze(0)
            val = self.detector.preprocess(feat_input)
            is_drift = self.detector.update(val)

            # 更新联合缓冲
            self.buffer_features.append(feat)
            self.buffer_bin_labels.append(lbl_bin[i])
            self.buffer_mul_labels.append(lbl_mul[i])

            if is_drift:
                batch_drift_detected = True
                self.drift_count += 1

                current_b = 0.0
                if hasattr(self.detector, '_get_total_bf'):
                    current_b = self.detector._get_total_bf()

                logger.info(f"🚨 [漂移触发] 样本 {i}, 贝叶斯因子 B={current_b:.5f}")

                # 触发适应
                if adapt and self.adapter is not None:
                    window = self.cfg.concept_drift.get('adaptation', {}).get('window', 500)
                    if len(self.buffer_features) >= 32:
                        adapt_feats = torch.stack(self.buffer_features[-window:])
                        adapt_bins = torch.stack(self.buffer_bin_labels[-window:])
                        adapt_muls = torch.stack(self.buffer_mul_labels[-window:])
                        logger.info(f"🔄 正在使用前 {len(adapt_feats)} 个历史样本进行联合微调适应...")

                        # 传入双标签进行联合微调
                        self.adapter.adapt(adapt_feats, adapt_bins, adapt_muls)

                self.detector.reset()

        buffer_limit = self.cfg.concept_drift.get('adaptation', {}).get('buffer_size', 2000)
        if len(self.buffer_features) > buffer_limit:
            self.buffer_features = self.buffer_features[-buffer_limit:]
            self.buffer_bin_labels = self.buffer_bin_labels[-buffer_limit:]
            self.buffer_mul_labels = self.buffer_mul_labels[-buffer_limit:]

        return batch_drift_detected

    def detect_drift(self):
        return self.last_drift_result

    def get_statistics(self):
        stats = {
            "algorithm": self.cfg.get('concept_drift', {}).get('algorithm', 'unknown').upper(),
            "drift_count_handler": self.drift_count
        }
        if self.detector:
            if hasattr(self.detector, 'get_statistics'):
                stats.update(self.detector.get_statistics())
            else:
                stats['total_samples'] = getattr(self.detector, 'total_samples', 0)
                stats['status'] = 'running' if self.enabled else 'disabled'
        return stats

    def reset(self):
        if self.detector: self.detector.reset()
        self.last_drift_result = (False, 0.0, {})
        self.drift_count = 0
        self.buffer_features.clear()
        self.buffer_bin_labels.clear()
        self.buffer_mul_labels.clear()



class FlowBertMultiview(pl.LightningModule):
    """多视图BERT模型"""

    def __init__(self, cfg: DictConfig, dataset):

        # 显式初始化两个父类，避免MRO问题 ， 2025-12-02 del by qinyf
        # pl.LightningModule.__init__(self)
        # SHAPAnalyzeMixin.__init__(self, cfg)

        super().__init__()

        # 保存 cfg，但忽略 dataset（不可序列化）
        self.save_hyperparameters(
            "cfg",
            logger=False,
            ignore=["dataset"]
        )
        self.cfg = cfg
        self.labels_cfg = cfg.datasets.labels

        # 1. 从 dataset 获取类别型特征的映射和有效列
        if dataset is None:
            raise ValueError("dataset must be provided when initializing FlowBertMultiview "
                            "because categorical embeddings depend on dataset statistics.")

        self.categorical_val2idx_mappings = dataset.categorical_val2idx_mappings
        assert self.categorical_val2idx_mappings is not None, \
                "Model loaded without dataset stats — categorical val2idx embeddings invalid!"

        self.categorical_columns_effective = dataset.categorical_columns_effective
        assert self.categorical_columns_effective is not None, \
                "Model loaded without dataset stats — categorical columns effective invalid!"

        # 2. 初始化 SHAP 组件 (放在 __init__ 最后)  2025-12-02 added by qinyf
        if self.cfg.shap.enable_shap:
            self.shap_analyzer = ShapAnalyzer(self)

        self.debug_mode = cfg.debug.debug_mode
        # 🔴 根据debug_mode 设置 nan_check_enabled 属性
        self.nan_check_enabled = getattr(cfg.debug, 'nan_check_enabled', self.debug_mode)

        # 初始化BERT模型和配置
        _require_transformers()
        self.bert, self.bert_config, self.tokenizer = self._load_bert_model(cfg)
        logger.info(f"加载的BERT模型的每个 token 的隐藏向量维度（hidden dimension）：bert_config.hidden_size = {self.bert_config.hidden_size}")

        # 检查各视图是否启用
        self.text_features_enabled = False
        if hasattr(cfg.data, 'text_features') and cfg.data.text_features is not None:
            if hasattr(cfg.data.text_features, 'enabled') and cfg.data.text_features.enabled:
                self.text_features_enabled = True

        self.domain_embedding_enabled = False
        if hasattr(cfg.data, 'domain_name_embedding_features') and hasattr(cfg.data.domain_name_embedding_features, 'enabled') and hasattr(cfg.data.domain_name_embedding_features, 'column_list'):
            if cfg.data.domain_name_embedding_features.enabled and len(cfg.data.domain_name_embedding_features.column_list) > 0:
                self.domain_embedding_enabled = True

        self.sequence_features_enabled = False
        if hasattr(cfg.data, 'sequence_features') and cfg.data.sequence_features is not None:
            if hasattr(cfg.data.sequence_features, 'enabled') and cfg.data.sequence_features.enabled:
                self.sequence_features_enabled = True

        logger.info(f"视图启用状态: 数值特征向量必选，数据包序列={self.sequence_features_enabled}, 文本={self.text_features_enabled}, 域名嵌入={self.domain_embedding_enabled}")
        # 安全检查
        if self.text_features_enabled and not hasattr(self, 'bert_config'):
            raise ValueError("文本特征已启用但BERT配置未初始化")

        # 初始化所有投影层
        self._init_projection_layers(cfg)

        # 计算实际启用的视图数量
        self.num_views = 1  # 表格数据特征：数值特征（必选） + 域名嵌入特征（可选）
        if self.text_features_enabled:
            self.num_views += 1 # 文本视图可选
        if self.sequence_features_enabled:
            self.num_views += 1 # 数据包序列视图可选

        logger.info(f"模型使用的视图数量: {self.num_views}")
        logger.info(f"多视图融合方法: {cfg.model.multiview.fusion.method}")

        # 初始化多视图融合层
        self.fusion_layer = MultiViewFusionFactory.create_fusion_layer(
            cfg=cfg,
            hidden_size=self.bert_config.hidden_size,
            num_views=self.num_views
        )

        # 初始化分类器
        self._init_classifier(cfg)

        # 初始化分类损失函数
        self._init_loss_function(cfg)

        # 初始化分类损失函数
        self._init_loss_function(cfg)

        # ====================== 新增 ======================
        # 初始化概念漂移处理器 (集成 BNDM + Adapter)
        self.drift_detector = DriftHandler(self, cfg)

        # Test metric buffers 新增漂移步骤记录
        self.test_step_accuracies = []
        self.test_drift_steps = []
        # =================================================



        # SHAP分析现在由SHAPAnalyzeMixin统一管理，无需重复配置

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        # 强制传递 dataset
        return super().load_from_checkpoint(checkpoint_path, **kwargs)

    def _load_bert_model(self, cfg: DictConfig) -> tuple:
        """
        加载BERT模型、配置和tokenizer

        Args:
            cfg: 配置对象

        Returns:
            tuple: (bert_model, bert_config, tokenizer)
        """
        # 添加参数验证
        if not hasattr(cfg.model.bert, 'model_name') or cfg.model.bert.model_name is None:
            raise ValueError("BERT模型名称未在配置中设置。请检查配置文件中的 bert.model_name 字段。")

        logger.info(f"使用BERT模型: {cfg.model.bert.model_name}")
        project_root = get_project_root()
        model_path = os.path.join(project_root, 'models_hub', cfg.model.bert.model_name)

        try:
            # 首先尝试从本地缓存加载
            logger.info("尝试从本地缓存加载BERT模型...")
            bert_config = BertConfig.from_pretrained(
                model_path,
                hidden_dropout_prob=cfg.model.bert.hidden_dropout_prob,
                attention_probs_dropout_prob=cfg.model.bert.attention_probs_dropout_prob)
            bert_model = BertModel.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(model_path)

            # 🔴 确保BERT模型设置为训练模式
            bert_model.train()
            logger.info(f"成功从本地缓存加载BERT模型，设置为训练模式")

        except (OSError, ValueError) as e:
            logger.warning(f"本地模型未找到: {e}, 尝试在线下载...")
            logger.warning(f"如果网络不可用，请手动下载模型并放置在本地路径: {model_path}")
            # 如果本地没有，下载并保存
            bert_config = BertConfig.from_pretrained(
                cfg.model.bert.model_name,
                hidden_dropout_prob=cfg.model.bert.hidden_dropout_prob,
                attention_probs_dropout_prob=cfg.model.bert.attention_probs_dropout_prob)
            bert_model = BertModel.from_pretrained(cfg.model.bert.model_name)
            tokenizer = BertTokenizer.from_pretrained(cfg.model.bert.model_name)

            # 🔴确保BERT模型设置为训练模式
            bert_model.train()

            # 保存到本地
            bert_config.save_pretrained(model_path)
            bert_model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logger.info("BERT模型在线下载并保存到本地完成，设置为训练模式")

        return bert_model, bert_config, tokenizer

    def _init_projection_layers(self, cfg: DictConfig):
        # 🔴 1. 初始化文本特征投影层（可选）
        if self.text_features_enabled:
            logger.info("初始化文本特征编码器")
        else:
            logger.info("跳过文本特征编码器初始化")

        # 🔴 2. 初始化数据包特征投影层（可选）
        if self.sequence_features_enabled:
            # 当前的两层设计（嵌入层 + 投影层）
            self.sequence_encoder = SequenceEncoder(
                embedding_dim=cfg.model.sequence.embedding_dim, # embedding_dim可以独立于 BERT 的隐藏层大小进行调优
                num_layers=cfg.model.sequence.num_layers,
                num_heads=cfg.model.sequence.num_heads,
                dropout=cfg.model.sequence.dropout,
                max_packet_seq_length=cfg.data.max_seq_length
            )
            # sequence_projection 是最轻量的跨模态 Adapter
            self.sequence_projection = nn.Linear(cfg.model.sequence.embedding_dim, self.bert_config.hidden_size)
            logger.info("初始化数据包序列编码器")
        else:
            self.sequence_encoder = None
            self.sequence_projection = None
            logger.info("跳过数据包序列编码器初始化")

        # 🔴 3. 初始化域名嵌入特征维度（可选），不考虑
        if self.domain_embedding_enabled:
            label_id_map = ConfigManager.read_session_label_id_map(self.cfg.data.dataset)
            self.prob_list_length = len(label_id_map)
            logger.info(f"域名嵌入特征概率列表长度: {self.prob_list_length}")
            self.domain_feature_dim = len(cfg.data.domain_name_embedding_features.column_list) * self.prob_list_length
            logger.info(f"域名嵌入特征长度: {self.domain_feature_dim}")
        else:
            self.domain_feature_dim = 0
            logger.info(f"域名嵌入特征长度: {self.domain_feature_dim}")

        # 4. 初始化类别型特征嵌入层（必选）
        # 类别型特征始终启用，因为所有流都有 conn.proto、service 等类别语义
        self.categorical_embedding_layers = nn.ModuleDict()

        # ⭐ 从 dataset 补充 category → index 映射
        for col, mapping in self.categorical_val2idx_mappings.items():
            # num_classes = (最大 index) + 1
            # 因为 index 范围为 [0 ... K]，共 K+1 个 embedding 向量
            # 其中 index=0 保留给 OOV （Out-Of-Vocabulary）
            num_classes = max(mapping.values()) + 1
            self.categorical_embedding_layers[col] = nn.Embedding(
                num_embeddings=num_classes,
                embedding_dim=self.bert_config.hidden_size
            )
            logger.info(f"初始化 categorical embedding: {col} → {num_classes} 类别")

        # 🔹 初始化 categorical LayerNorm（拼接后做归一化）
        self.categorical_norm = nn.LayerNorm(
            normalized_shape=self.bert_config.hidden_size * len(self.categorical_columns_effective)
        )

        # 5. 初始化数值型特征（必选）+域名嵌入特征+类别型特征（必选）的表格数据投影层
        # Delay to forward 时计算 tabular_feature_dim
        self.numeric_feature_dim = (
            len(cfg.data.tabular_features.numeric_features.flow_features)
            + len(cfg.data.tabular_features.numeric_features.x509_features)
            + len(cfg.data.tabular_features.numeric_features.dns_features)
        )
        logger.info(f"数值型流特征数: {self.numeric_feature_dim}")
        self.categorical_feature_dim = len(self.categorical_columns_effective) * self.bert_config.hidden_size
        logger.info(f"总类别型特征数: {len(self.categorical_columns_effective)}")
        logger.info(f"总类别型特征维度: {self.bert_config.hidden_size} * {len(self.categorical_columns_effective)} = {self.bert_config.hidden_size * len(self.categorical_columns_effective)}")
        self.tabular_feature_dim = self.numeric_feature_dim + self.domain_feature_dim + self.categorical_feature_dim
        logger.info(f"表格数据总特征维度: 数值特征({self.numeric_feature_dim}) + 域名嵌入({self.domain_feature_dim}) + 类别型特征({self.categorical_feature_dim}) = {self.tabular_feature_dim}")

        self.tabular_projection = nn.Linear(
            self.tabular_feature_dim,
            self.bert_config.hidden_size
        )
        logger.info(f"初始化表格特征线性投影层: 输入维度={self.tabular_feature_dim}, 输出维度={self.bert_config.hidden_size}")


    def _init_classifier(self, cfg: DictConfig):
        """初始化分类器 - 根据融合方法调整输入维度"""
        self.classifier_input_dim = self._get_classifier_input_dim(cfg)
        self._init_is_malicious_classifier(cfg)
        self._init_attack_family_classifier(cfg)
        self._init_attack_type_classifier(cfg)

    def _get_classifier_input_dim(self, cfg: DictConfig) -> int:
        fusion_method = cfg.model.multiview.fusion.method

        if fusion_method == "concat":
            # 对于concat方法，分类器输入维度是所有启用视图的维度之和
            # classifier_input_dim = self.bert_config.hidden_size  # 数值特征投影后的维度（必选）

            # if self.sequence_features_enabled:
            #     classifier_input_dim += self.bert_config.hidden_size
            # if self.text_features_enabled:
            #     classifier_input_dim += self.bert_config.hidden_size
            # if self.domain_embedding_enabled:
            #     classifier_input_dim += self.bert_config.hidden_size

            # logger.info(f"视图启用状态: 数值特征向量必选，数据包序列={self.sequence_features_enabled}, 文本={self.text_features_enabled}, 域名嵌入={self.domain_embedding_enabled}")
            # logger.info(f"拼接融合总维度: {classifier_input_dim}")

            # 🔴 四个视图的concat，改成了ConcatFusion，其内部做了拼接解释特征的线性投影到 hidden_size
            classifier_input_dim = self.bert_config.hidden_size
            logger.warning(
                f"[Fusion Warning] 'concat' 融合原始特征维度 = bert_config.hidden_size * num_views "
                f"但 ConcatFusion 会自动投影回 bert_config.hidden_size，因此分类器输入维度固定为 bert_config.hidden_size = {self.bert_config.hidden_size} "
                f"(num_views={self.num_views})"
            )
            return classifier_input_dim
        else:
            # 其他方法都输出 hidden_size 维度
            classifier_input_dim = self.bert_config.hidden_size
            logger.info(f"[Fusion Info] 使用 {fusion_method} 融合，输出维度 = bert_config.hidden_size = {self.bert_config.hidden_size}")
            return classifier_input_dim

    def _init_is_malicious_classifier(self, cfg: DictConfig):
        is_malicious_classifier_cfg = cfg.datasets.tasks.outputs.is_malicious.classifier

        logger.info(f"is_malicious善意/恶意流量分类器的输入维度: {self.classifier_input_dim} ，"
                    f"hidden_dims={is_malicious_classifier_cfg.hidden_dims}")


        # 添加 is_malicious（二分类，善意 / 恶意）的 flow-level 分类器隐藏层
        is_malicious_classifier_layers = []
        current_dim = self.classifier_input_dim
        for i, hidden_dim in enumerate(is_malicious_classifier_cfg.hidden_dims):
            logger.info(f"is_malicious善意/恶意分类器隐藏层 {i+1}: {current_dim} -> {hidden_dim}")

            is_malicious_classifier_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU() if is_malicious_classifier_cfg.activation == "gelu" else nn.ReLU(),
                nn.Dropout(is_malicious_classifier_cfg.dropout),
                nn.LayerNorm(hidden_dim)
            ])
            current_dim = hidden_dim  # 更新当前维度

        # 添加is_malicious善意/恶意流量分类器的输出层
        # 输出层：1 维，对应 BCEWithLogitsLoss
        is_malicious_classifier_layers.append(nn.Linear(current_dim, 1))

        self.is_malicious_classifier = nn.Sequential(*is_malicious_classifier_layers)

        logger.info(
            f"is_malicious 分类器结构: "
            f"input_dim={self.classifier_input_dim} -> "
            f"hidden_dims={is_malicious_classifier_cfg.hidden_dims} -> "
            f"output_dim=1 (binary classification)"
        )

    def _init_attack_family_classifier(self, cfg: DictConfig):
        """
        初始化 attack_family 分类器（OVR 多分类）

        attack_family 采用 One-vs-Rest (OVR) 形式：
        - 每个攻击家族对应一个二分类器
        - 输出维度 = 攻击家族数量
        - 配合 BCEWithLogitsLoss + multi-hot 标签使用
        """

        # ---------- 1. 配置检查 ----------
        if "attack_family" not in cfg.datasets.tasks.outputs:
            logger.info("未配置 attack_family 任务，跳过 attack_family 分类器初始化")
            self.attack_family_classifier = None
            return

        attack_family_cls_task_cfg = cfg.datasets.tasks.outputs.attack_family
        assert attack_family_cls_task_cfg.strategy == "ovr", \
            f"attack_family 目前仅支持 OVR 策略，当前配置为 {attack_family_cls_task_cfg.strategy}"

        if attack_family_cls_task_cfg.get("enabled", False) is False:
            self.attack_family_classifier = None
            logger.info("[attack_family] cfg.datasets.tasks.outputs.attack_family.enabled开关的数值为False，跳过 attack_family 分类器初始化。")
            return

        # ---------- 2. 解析攻击家族信息 ----------
        assert "attack_family" in self.labels_cfg, \
            "attack_family 已启用，但 labels_cfg.attack_family 未定义"

        # 1️⃣ 类别定义：唯一来源
        self.attack_family_classes = [
            c.strip() for c in self.labels_cfg.attack_family.classes
        ]

        # 2️⃣ 模型 / loss / logits 统一使用同一顺序
        self.attack_family_names = self.attack_family_classes
        attack_family_number = len(self.attack_family_names)

        logger.info(
            f"[attack_family] 启用 OVR 攻击家族分类任务，共 {attack_family_number} 个攻击家族: "
            f"{self.attack_family_names}"
        )

        assert self.attack_family_names == self.labels_cfg.attack_family.classes, (
            f"attack_family_names 与 dataset attack_family_classes 顺序不一致:\n"
            f"model:   {self.attack_family_names}\n"
            f"dataset: {self.labels_cfg.attack_family.classes}"
        )

        # 3️⃣ 校验 class_weights 是否与 labels_cfg 一致（fail-fast）
        weight_keys = list(attack_family_cls_task_cfg.class_weights.keys())
        if set(weight_keys) != set(self.attack_family_names):
            raise ValueError(
                "[attack_family] labels_cfg.attack_family.classes 与 "
                "task.outputs.attack_family.class_weights 键不一致!\n"
                f"labels_cfg: {self.attack_family_names}\n"
                f"class_weights: {weight_keys}"
            )

        # ---------- 3. 分类器输入维度 ----------
        input_dim = self.classifier_input_dim
        logger.info(
            f"[attack_family] 分类器输入维度 = {input_dim} "
            f"(来自多视图特征融合输出)"
        )

        # ---------- 4. 构建 OVR 分类器网络 ----------
        layers = []
        current_dim = input_dim

        for i, hidden_dim in enumerate(attack_family_cls_task_cfg.classifier.hidden_dims):
            logger.info(
                f"[attack_family] 分类器隐藏层 {i+1}: {current_dim} -> {hidden_dim}"
            )
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(attack_family_cls_task_cfg.classifier.dropout),
                nn.LayerNorm(hidden_dim),
            ])
            current_dim = hidden_dim

        # ---------- 5. 输出层 ----------
        # 输出维度 = 攻击家族数量（OVR，每一维对应一个家族）
        layers.append(nn.Linear(current_dim, attack_family_number))

        self.attack_family_classifier = nn.Sequential(*layers)

        logger.info(
            f"[attack_family] OVR 分类器结构构建完成: "
            f"input_dim={input_dim} -> "
            f"hidden_dims={attack_family_cls_task_cfg.classifier.hidden_dims} -> "
            f"output_dim={attack_family_number} (OVR, multi-class)"
        )

    def _init_attack_type_classifier(self, cfg: DictConfig):
        """
        初始化 attack_type 分类器
        """
        # ---------- 1. 配置检查 ----------
        if "attack_type" not in cfg.datasets.tasks.outputs:
            logger.info("[attack_type] 未配置 attack_type 任务，跳过 attack_type 分类器初始化")
            self.attack_type_classifier = None
            return

        attack_type_cls_task_cfg = cfg.datasets.tasks.outputs.attack_type
        assert attack_type_cls_task_cfg.strategy == "softmax", \
            f"attack_type 目前仅支持 softmax 策略，当前配置为 {attack_type_cls_task_cfg.strategy}"

        if attack_type_cls_task_cfg.get("enabled", False) is False:
            self.attack_type_classifier = None
            logger.info("[attack_type] cfg.datasets.tasks.outputs.attack_type.enabled开关的数值为False，跳过 attack_type 分类器初始化。")
            return

        # ---------- 2. 解析攻击类别信息 ----------
        assert "attack_type" in self.labels_cfg, \
            "attack_type 已启用，但 labels_cfg.attack_type 未定义"

        # 1️⃣ 类别定义：唯一来源
        self.attack_type_classes = [
            c.strip() for c in self.labels_cfg.attack_type.classes
        ]

        # 2️⃣ 模型 / loss / logits 统一使用同一顺序
        self.attack_type_names = self.attack_type_classes
        attack_type_number = len(self.attack_type_names)

        logger.info(
            f"[attack_type] 启用攻击类型分类任务，共 {attack_type_number} 个攻击类型: "
            f"{self.attack_type_names}"
        )

        assert self.attack_type_names == self.labels_cfg.attack_type.classes, (
            f"attack_type_names 与 dataset attack_type_classes 顺序不一致:\n"
            f"model:   {self.attack_type_names}\n"
            f"dataset: {self.labels_cfg.attack_type.classes}"
        )

        # 3️⃣ 校验 class_weights 是否与 labels_cfg 一致（fail-fast）
        weight_keys = list(attack_type_cls_task_cfg.class_weights.keys())
        if set(weight_keys) != set(self.attack_type_names):
            raise ValueError(
                "[attack_type] labels_cfg.attack_type.classes 与 "
                "task.outputs.attack_type.class_weights 键不一致!\n"
                f"labels_cfg: {self.attack_type_names}\n"
                f"class_weights: {weight_keys}"
            )

        # ---------- 3. 分类器输入维度 ----------
        input_dim = self.classifier_input_dim
        logger.info(
            f"[attack_type] 分类器输入维度 = {input_dim} "
            f"(来自多视图特征融合输出)"
        )

        # ---------- 4. 构建 Softmax 分类器网络 ----------
        layers = []
        current_dim = input_dim

        for i, hidden_dim in enumerate(attack_type_cls_task_cfg.classifier.hidden_dims):
            logger.info(
                f"[attack_type] 分类器隐藏层 {i+1}: {current_dim} -> {hidden_dim}"
            )
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(attack_type_cls_task_cfg.classifier.dropout),
                nn.LayerNorm(hidden_dim),
            ])
            current_dim = hidden_dim

        # ---------- 5. 输出层 ----------
        layers.append(nn.Linear(current_dim, attack_type_number))

        self.attack_type_classifier = nn.Sequential(*layers)

        logger.info(
            f"[attack_type] Softmax 分类器结构构建完成: "
            f"input_dim={input_dim} -> "
            f"hidden_dims={attack_type_cls_task_cfg.classifier.hidden_dims} -> "
            f"output_dim={attack_type_number} (Softmax, multi-class)"
        )


    def _init_loss_function(self, cfg: DictConfig):
        self._init_is_malicious_loss_function(cfg)
        self._init_attack_family_loss_function(cfg)
        self._init_attack_type_loss_function(cfg)

    def _init_is_malicious_loss_function(self, cfg: DictConfig):
        """
        初始化 is_malicious（二分类，善意/恶意）的损失函数
        使用 BCEWithLogitsLoss + 类别权重
        """
        # 初始化分类损失函数
        self.is_malicious_class_loss = nn.BCEWithLogitsLoss(reduction='none')

        # 🔹 默认权重（不加权）
        class_weights = [1.0, 1.0]

        # 🔹 从 task.outputs 中读取（如果配置了）
        try:
            task_cfg = cfg.datasets.tasks.outputs.get("is_malicious", None)
            if task_cfg is not None and "class_weights" in task_cfg:
                cw = task_cfg.class_weights
                if isinstance(cw, (list, tuple)) and len(cw) == 2:
                    class_weights = list(map(float, cw))
        except Exception as e:
            logger.warning(f"读取 task.outputs.is_malicious.class_weights 失败，使用默认权重: {e}")

        if class_weights is None or len(class_weights) != 2:
            logger.warning("[is_malicious] class_weights 非法，重置为 [1.0, 1.0]")
            # 使用默认权重 [1.0, 1.0]
            class_weights = [1.0, 1.0]

        self.is_malicious_class_weights = torch.tensor(class_weights, dtype=torch.float32)
        logger.info(
            f"[is_malicious] 损失函数初始化完成: "
            f"BCEWithLogitsLoss, class_weights={class_weights}"
        )

    def _init_attack_family_loss_function(self, cfg: DictConfig):
        """
        初始化 attack_family 的 OVR 多分类损失函数
        - 每个攻击家族一个 BCE loss
        - 使用 per-family [neg, pos] 权重
        """

        if not hasattr(self, "attack_family_classifier") or \
        self.attack_family_classifier is None:
            self.attack_family_loss_fn = None
            self.attack_family_class_weights = None
            logger.info("[attack_family] 未启用任务，跳过损失函数初始化")
            return

        attack_family_cls_task_cfg = cfg.datasets.tasks.outputs.attack_family

        self.attack_family_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        # 为攻击家族构建二分类的权重矩阵 [num_families, 2]
        weights = []
        for family in self.attack_family_names:
            w = attack_family_cls_task_cfg.class_weights.get(family, [1.0, 1.0])
            if len(w) != 2:
                logger.warning(
                    f"[attack_family] 家族 {family} 的 class_weights 非法，使用默认 [1.0, 1.0]"
                )
                w = [1.0, 1.0]
            weights.append(w)

        self.attack_family_class_weights = torch.tensor(
            weights, dtype=torch.float32
        )

        logger.info(
            f"[attack_family] OVR 损失函数初始化完成: "
            f"BCEWithLogitsLoss, "
            f"class_weights shape={self.attack_family_class_weights.shape}"
        )

    def _init_attack_type_loss_function(self, cfg: DictConfig):
        """
        初始化 attack_type 的 Softmax 多分类损失函数
        """
        attack_type_cls_task_cfg = cfg.datasets.tasks.outputs.attack_type
        # 检查使能开关配置项
        if attack_type_cls_task_cfg.get("enabled", False) is False:
            self.attack_type_loss_fn = None
            self.attack_type_class_weights = None
            logger.info("[attack_type] 未启用任务，跳过损失函数初始化")
            return

        # 构建所有攻击类型的分类权重
        if attack_type_cls_task_cfg.loss.use_class_weight:
            self.attack_type_class_weights = torch.tensor(
                [attack_type_cls_task_cfg.class_weights[name]
                for name in self.attack_type_names],
                dtype=torch.float32
            )
        else:
            self.attack_type_class_weights = None

        self.attack_type_loss_fn_name = attack_type_cls_task_cfg.loss.name

        if self.attack_type_loss_fn_name == "ce":
            pass

        elif self.attack_type_loss_fn_name == "focal":
            self.attack_type_focal_gamma = attack_type_cls_task_cfg.loss.gamma

        else:
            raise ValueError(
                f"[attack_type] 不支持的 loss 函数类型: {self.attack_type_loss_fn_name}, "
                f"仅支持 'ce' 或 'focal'"
            )

    def _compute_losses(self, outputs, batch, stage: str):
        """计算分类损失 + tabular特征重建损失 + 总损失"""

        # ===== is_malicious 分类损失 =====
        is_malicious_cls_logits = outputs['is_malicious_cls_logits']
        # is_malicious_prob = torch.sigmoid(is_malicious_cls_logits)
        # is_malicious_pred = (is_malicious_prob > 0.5).float()
        is_malicious_label = batch['is_malicious_label']
        is_malicious_class_loss = self._compute_is_malicious_class_loss(is_malicious_cls_logits, is_malicious_label)

        # 检查is_malicious分类损失值
        if is_malicious_class_loss is None or torch.isnan(is_malicious_class_loss) or torch.isinf(is_malicious_class_loss):
            logger.error(f"🚨 {stage} is_malicious_class_loss 异常: {is_malicious_class_loss}")
            # 尝试使用小损失值继续训练
            is_malicious_class_loss = is_malicious_cls_logits.sum() * 0.0

        is_malicious_weight = getattr(self.cfg.datasets.tasks.outputs.is_malicious, "weight", 1.0)
        total_loss = is_malicious_weight * is_malicious_class_loss

        # ===== attack_family 分类损失 =====
        attack_family_class_loss = None
        if self.attack_family_classifier is not None:
            attack_family_cls_logits = outputs["attack_family_cls_logits"]
            assert "attack_family_label" in batch, \
                f"attack_family_label not found in batch keys: {batch.keys()}"
            attack_family_label = batch["attack_family_label"]
            attack_family_class_loss = self._compute_attack_family_class_loss(attack_family_cls_logits, attack_family_label, is_malicious_label)
            # 检查attack_family分类损失值
            if attack_family_class_loss is None or torch.isnan(attack_family_class_loss) or torch.isinf(attack_family_class_loss):
                logger.error(f"🚨 {stage} attack_family_loss 异常: {attack_family_class_loss}")
                attack_family_class_loss = attack_family_cls_logits.sum() * 0.0

            attack_family_weight = getattr(self.cfg.datasets.tasks.outputs.attack_family, "weight", 1.0)
            total_loss = total_loss + attack_family_weight * attack_family_class_loss

        # ===== attack_type 分类损失 =====
        attack_type_class_loss = None
        if self.attack_type_classifier is not None:
            attack_type_cls_logits = outputs['attack_type_cls_logits']
            assert 'attack_type_label' in batch, \
                f"attack_type_label not found in batch keys: {batch.keys()}"
            attack_type_label = batch['attack_type_label']
            attack_type_class_loss = self._compute_attack_type_class_loss(attack_type_cls_logits, attack_type_label, is_malicious_label)
            # 检查attack_type分类损失值
            if attack_type_class_loss is None or torch.isnan(attack_type_class_loss) or torch.isinf(attack_type_class_loss):
                logger.error(f"🚨 {stage} attack_type_loss 异常: {attack_type_class_loss}")
                attack_type_class_loss = attack_type_cls_logits.sum() * 0.0

            attack_type_weight = getattr(self.cfg.datasets.tasks.outputs.attack_type, "weight", 1.0)
            total_loss = total_loss + attack_type_weight * attack_type_class_loss

        # ===== 返回所有损失 =====
        return {
            "total_loss": total_loss,
            "is_malicious_class_loss": is_malicious_class_loss,
            "attack_family_class_loss": attack_family_class_loss,
            "attack_type_class_loss": attack_type_class_loss,
        }

    def _compute_is_malicious_class_loss(self, logits, labels):
        """计算分类损失"""
        # 确保标签形状正确 [batch_size, 1]
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        # 基本损失计算
        base_loss = self.is_malicious_class_loss(logits, labels)

        # 确保 is_malicious_class_weights 已初始化
        if self.is_malicious_class_weights is not None:
            # 应用类别权重
            binary_weights = self.is_malicious_class_weights.to(logits.device) # ✅ 确保权重在正确的设备上
            binary_weights = torch.where(
                labels == 1,
                binary_weights[1],  # is_malicious=1 的权重
                binary_weights[0],  # is_malicious=0 的权重
            )
            weighted_loss = base_loss * binary_weights

            return safe_mean(weighted_loss)
        else:
            return safe_mean(base_loss)

    def _compute_attack_family_class_loss(self, logits, label, is_malicious_label):
        """
        计算 attack_family 的 OVR 多分类损失（条件损失）

        说明：
        - attack_family 仅在真实恶意样本（is_malicious_label == 1）上具有语义定义；
        - benign 样本在 attack_family 维度上属于 not applicable，
        不参与 attack_family 的损失计算与梯度反传；
        - 因此，该损失函数刻画的是条件分布下的分类性能：
            P(attack_family | is_malicious = 1)。

        参数说明：
        - logits: [B, K]
            attack_family 分类器的原始输出（每个攻击家族一个 OVR logit）
        - label: [B, K]（multi-hot）
            attack_family 的真实标签（OVR / multi-label 表示）
        - is_malicious_label: [B]（0/1）
            is_malicious 的真实标签，用于筛选具有 attack_family 语义的样本
        """
        assert logits.shape == label.shape, \
            f"attack_family logits/labels shape mismatch: {logits.shape} vs {label.shape}"

        # 1️⃣ 先算完整 batch 的 per-sample per-class loss
        # base_loss: [B, K]
        # 基础 BCE 损失（OVR）：[B_malicious, K]
        base_loss = self.attack_family_loss_fn(logits, label.float())

        # 2️⃣ class weight（如果有）
        if self.attack_family_class_weights is not None:
            # class_weights 形状为 [K, 2]，表示每个攻击家族的 [neg, pos] 权重
            # 用于缓解不同 attack_family 之间的类别不平衡问题
            weights = self.attack_family_class_weights.to(logits.device)

            # 对每个 OVR 维度：
            # - label == 1 时使用正类权重（pos）
            # - label == 0 时使用负类权重（neg）
            # 通过 broadcasting 扩展到 [B_malicious, K]
            per_elem_weight = torch.where(
                label == 1,
                weights[:, 1],
                weights[:, 0]
            )

            base_loss = base_loss * per_elem_weight

        # 3️⃣ mask benign（不 return）
        mask = (is_malicious_label.view(-1) == 1)

        if mask.any():
            masked_loss = base_loss[mask]
            loss_val = masked_loss.sum() / mask.sum().clamp_min(1)
        else:
            loss_val = base_loss.sum() * 0.0

        return loss_val

    def _compute_attack_type_class_loss(self, logits, label, is_malicious_label):
        """
        attack_type:
        - 单标签 multiclass
        - 仅在 is_malicious == 1 的样本上计算
        """
        # logits: [B, C]
        # label:  [B] (LongTensor, class index)
        assert logits.dim() == 2, f"attack_type logits must be [B,C], got {logits.shape}"
        assert label.dim() == 1,  f"attack_type label must be [B], got {label.shape}"

        assert self.attack_type_loss_fn_name is not None, \
            "[attack_type] loss_fn_name 未初始化，请检查配置 attack_type.loss.name"

        weight = None
        if self.attack_type_class_weights is not None:
            weight = self.attack_type_class_weights.to(logits.device)

        if self.attack_type_loss_fn_name == "ce":
            # 1️⃣ 先算“完整 batch”的 loss（per-sample）
            # ignore_index: index to ignore (e.g., benign = -1)，具体查看dataset.py里面的_compute_attack_family_and_type_labels()函数
            per_sample_loss = F.cross_entropy(
                logits, label,
                weight=weight,
                label_smoothing=0.0,
                ignore_index=-1,
                reduction="none",
            )

            # 2️⃣ mask benign（但不 early return）
            mask = (label != -1)

            if mask.any():
                masked_loss = per_sample_loss[mask]
                loss_val = masked_loss.sum() / mask.sum().clamp_min(1)
            else:
                loss_val = per_sample_loss.sum() * 0.0  # DDP-safe

        elif self.attack_type_loss_fn_name == "focal":
            loss_val = self.focal_ce_loss(
                logits, label,
                weight=weight,
                gamma=self.attack_type_focal_gamma,
                ignore_index=-1,)

        else:
            loss_val = logits.sum() * 0.0

        return loss_val

    @staticmethod
    def focal_ce_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index = -1,
    ):
        """
        Multi-class focal loss with softmax (index-based).

        Args:
            logits: [N, C]
            labels: [N] (LongTensor, class index)
            weight: [C] or None
            gamma: focusing parameter
            ignore_index: index to ignore (e.g., benign = -1)，具体查看dataset.py里面的_compute_attack_family_and_type_labels()函数
        """

        # ===== 1️⃣ 形态与语义硬校验（非常重要） =====
        if labels.dtype != torch.long:
            raise RuntimeError(
                f"attack_type expects LongTensor index labels, got {labels.dtype}"
            )

        if labels.dim() != 1:
            raise RuntimeError(
                f"attack_type labels must be 1-D class indices, got shape={labels.shape}"
            )

        valid_mask = labels != ignore_index
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0  # DDP-safe

        logits = logits[valid_mask]
        labels = labels[valid_mask]

        # ===== 2️⃣ 标准 CE（index-based，禁止 smoothing） =====
        ce = F.cross_entropy(
            logits, labels,
            weight=weight,
            reduction="none",
            label_smoothing=0.0
        )

        # ===== 4️⃣ focal scaling（注意 device 对齐） =====
        probs = F.softmax(logits, dim=1)
        pt = probs[
            torch.arange(len(labels), device=labels.device),
            labels
        ]

        loss = ((1.0 - pt) ** gamma) * ce
        return safe_mean(loss)

    def _process_text_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """处理文本特征，返回[CLS] token嵌入"""
        if not self.text_features_enabled:
            # 返回零向量占位符
            batch_size = batch['numeric_features'].shape[0] # 利用必备的numeric_features维度
            return torch.zeros(batch_size, self.bert_config.hidden_size if self.bert_config else 512,
                            device=self.device)

        combined_texts = batch['combined_text']

        encoding = self.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.data.max_seq_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        text_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        return text_outputs.last_hidden_state[:, 0]  # 返回[CLS] token嵌入

    # ---------------- 安全训练模式 ----------------
    def on_train_start(self) -> None:
        """确保所有子模块都在 train 模式"""
        super().on_train_start()

        # 设置所有模块为训练模式
        self.train()

        # 设置序列编码器训练模式（如果存在）
        if hasattr(self, 'sequence_encoder') and self.sequence_encoder is not None:
            self.sequence_encoder.train()
            if self.debug_mode:
                logger.info("序列编码器设置为训练模式")

        # 设置用于文本特征表征的BERT（如果存在），进入训练模式
        if self.bert is not None:
            self.bert.train()

        # 设置数值投影层训练模式（必选）
        if hasattr(self, 'tabular_projection') and self.tabular_projection is not None:
            self.tabular_projection.train()
            if self.debug_mode:
                logger.info("表格特征的投影层设置为训练模式")

        # 设置分类器训练模式
        if hasattr(self, 'is_malicious_classifier') and self.is_malicious_classifier is not None:
            self.is_malicious_classifier.train()

        if hasattr(self, 'attack_family_classifier') and self.attack_family_classifier is not None:
            self.attack_family_classifier.train()

        # 设置所有投影层为训练模式
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Dropout)):
                module.train()

        # 额外检查：列出所有模块的训练状态
        if self.debug_mode:
            eval_modules = []
            for name, module in self.named_modules():
                if not module.training:
                    eval_modules.append(name)

            if eval_modules:
                logger.warning(f"以下模块仍在评估模式: {eval_modules}")
            else:
                logger.info("✅ 所有模块都已正确设置为训练模式")

    def _build_tabular_features(self, batch):
        """构建表格数据特征向量"""
        # 1. 数值型特征处理（必选）
        numeric_features = batch['numeric_features'].to(self.device)
        # print("DEBUG numeric_features shape:", numeric_features.shape)

        # 2. 类别型特征处理（必选）
        categorical_ids = batch["categorical_features"].to(self.device)   # [B, C]
        batch_size, num_cat_cols = categorical_ids.shape
        expected_num_cat_cols = len(self.categorical_columns_effective)

        # categorical_ids 形状应为 [B, num_effective_cols]
        assert num_cat_cols == expected_num_cat_cols, \
            f"⚠ categorical_features 列数不匹配：batch 中 {num_cat_cols}，dataset 中{expected_num_cat_cols}"

        categorical_embedded_list = []
        for i, cat_col in enumerate(self.categorical_columns_effective):
            cat_emb_layer = self.categorical_embedding_layers[cat_col]
            assert cat_emb_layer is not None, f"⚠ categorical_embedding_layer for column={cat_col} 找不到!"
            cat_col_ids = categorical_ids[:, i]       # [B]
            cat_col_emb = cat_emb_layer(cat_col_ids)  # [B, H]
            categorical_embedded_list.append(cat_col_emb)

        categorical_features = torch.cat(categorical_embedded_list, dim=1) # [B, C*H]
        if self.debug_mode:
            _debug_tensor_for_nan_inf_count(categorical_features, "categorical_features (before_norm)")
            # print("DEBUG categorical_features shape (before_norm):", categorical_features.shape)

        # ⭐ 规范化类别特征
        categorical_features = self.categorical_norm(categorical_features)
        if self.debug_mode:
            _debug_tensor_for_nan_inf_count(categorical_features, "categorical_features (after_norm)")
            # print("DEBUG categorical_features shape (after_norm):", categorical_features.shape)

        # 3. 域名嵌入特征处理（可选）
        if self.domain_embedding_enabled:
            domain_embeddings = batch['domain_embedding_features'].to(self.device)
            # print("DEBUG domain_embeddings shape:", domain_embeddings.shape)
        else:
            domain_embeddings = None

        # 4. 表格数据特征融合
        if self.domain_embedding_enabled:
            tabular_features = torch.cat([numeric_features, categorical_features, domain_embeddings], dim=1)
        else:
            tabular_features = torch.cat([numeric_features, categorical_features], dim=1)

        # print("DEBUG tabular_features shape:", tabular_features.shape)
        return tabular_features

    # ---------------- forward ----------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """前向传播"""

        # 检查输入数据是否有NaN/Inf
        if self.debug_mode:
            _check_input_dict_for_nan(batch)

        # 添加维度调试
        # logger.debug(f"数值特征维度: {batch['numeric_features'].shape}")
        # logger.debug(f"域名嵌入特征维度: {batch['domain_embedding_features'].shape}")

        # 1. 数据包序列特征处理（可选）
        if self.sequence_features_enabled and self.sequence_encoder is not None:
            sequence_data = {
                'directions': batch['directions'],
                'iat_times': batch['iat_times'],
                'payload_sizes': batch['payload_sizes'],
                'packet_numbers': batch['packet_numbers'],
                'avg_payload_sizes': batch['avg_payload_sizes'],
                'durations': batch['durations'],
                'sequence_mask': batch['sequence_mask'],  # 有效token掩码
            }

            # 检查输入数据
            if self.debug_mode:
                _debug_tensor_for_nan_inf_count(sequence_data['directions'], "输入_directions")
                _debug_tensor_for_nan_inf_count(sequence_data['payload_sizes'], "输入_payload_sizes")
                _debug_tensor_for_nan_inf_count(sequence_data['iat_times'], "输入_iat_times")
                _debug_tensor_for_nan_inf_count(sequence_data['packet_numbers'], "输入_packet_numbers")
                _debug_tensor_for_nan_inf_count(sequence_data['avg_payload_sizes'], "输入_avg_payload_sizes")
                _debug_tensor_for_nan_inf_count(sequence_data['durations'], "输入_durations")
                _debug_tensor_for_nan_inf_count(sequence_data['sequence_mask'], "输入_sequence_mask")

            seq_outputs = self.sequence_encoder(sequence_data)
            sequence_embedding = seq_outputs["sequence_embedding"]
            if self.debug_mode:
                _debug_tensor_for_nan_inf_count(sequence_embedding, "sequence_encoder输出")
                sequence_embedding = _safe_check_nan_in_tensor(sequence_embedding, "sequence_emb", "序列编码")

            # 把数据包序列的embedding投影降维到和bert的dimension一样的维度，方便后续的多视图特征融合
            sequence_projected_embedding = self.sequence_projection(sequence_embedding)
            if self.debug_mode:
                _debug_tensor_for_nan_inf_count(sequence_projected_embedding, "sequence_projection输出")
                sequence_projected_embedding = _safe_check_nan_in_tensor(sequence_projected_embedding, "sequence_projected_embedding", "序列嵌入的投影")

        else:
            # 创建空的序列特征输出
            batch_size = batch['numeric_features'].shape[0]  # 借用必选的numeric_features，生成特征维度
            sequence_projected_embedding = torch.zeros(batch_size, self.bert_config.hidden_size, device=self.device)

        # 2. 文本特征处理（可选）
        if self.text_features_enabled:
            text_outputs = self._process_text_features(batch)
            if self.debug_mode:
                _debug_tensor_for_nan_inf_count(text_outputs, "text_outputs")
                text_outputs = _safe_check_nan_in_tensor(text_outputs, "text_outputs", "BERT处理")
        else:
            # 创建空的文本特征输出
            batch_size = batch['numeric_features'].shape[0]  # 借用必选的numeric_features，生成特征维度
            text_outputs = torch.zeros(batch_size, self.bert_config.hidden_size, device=self.device)

        # 3. 表格数据特征构建
        tabular_features = self._build_tabular_features(batch)
        if self.debug_mode:
            _debug_tensor_for_nan_inf_count(tabular_features, "输入_tabular_features")

        # 4. 表格数据特征投影
        tabular_outputs = self.tabular_projection(tabular_features)
        if self.debug_mode:
            _debug_tensor_for_nan_inf_count(tabular_outputs, "tabular_outputs输出")
            tabular_outputs = _safe_check_nan_in_tensor(tabular_outputs, "tabular_outputs", "表格数据投影")

        # 5. 多视图特征融合：数据包序列特征+文本特征+表格数据特征
        multiview_outputs = self._fuse_multi_views(sequence_projected_embedding, text_outputs, tabular_outputs)
        if self.debug_mode:
            _debug_tensor_for_nan_inf_count(multiview_outputs, "融合后_multiview_outputs")
            multiview_outputs = _safe_check_nan_in_tensor(multiview_outputs, "multiview_outputs", "多视图融合")

        # 6. 分类器
        is_malicious_cls_logits = self.is_malicious_classifier(multiview_outputs)
        if self.debug_mode:
            _debug_tensor_for_nan_inf_count(is_malicious_cls_logits, "is_malicious_classifier输出")
            is_malicious_cls_logits = _safe_check_nan_in_tensor(is_malicious_cls_logits, "is_malicious_cls_logits", "is_malicious_分类器")

        if self.attack_family_classifier is not None:
            attack_family_cls_logits = self.attack_family_classifier(multiview_outputs)
            if self.debug_mode:
                _debug_tensor_for_nan_inf_count(attack_family_cls_logits, "attack_family_cls_logits输出")
                attack_family_cls_logits = _safe_check_nan_in_tensor(attack_family_cls_logits, "attack_family_cls_logits", "attack_family_分类器")
        else:
            attack_family_cls_logits = multiview_outputs.sum() * 0.0  # 占位

        if self.attack_type_classifier is not None:
            attack_type_cls_logits = self.attack_type_classifier(multiview_outputs)
            if self.debug_mode:
                _debug_tensor_for_nan_inf_count(attack_type_cls_logits, "attack_type_cls_logits输出")
                attack_type_cls_logits = _safe_check_nan_in_tensor(attack_type_cls_logits, "attack_type_class_logits", "attack_type_分类器")
        else:
            attack_type_cls_logits = multiview_outputs.sum() * 0.0  # 占位

        return {
            'sequence_embeddings': sequence_projected_embedding,
            'text_embeddings': text_outputs if self.text_features_enabled else None,
            'tabular_embeddings': tabular_outputs,
            'multiview_embeddings': multiview_outputs,
            'is_malicious_cls_logits': is_malicious_cls_logits,
            'attack_family_cls_logits': attack_family_cls_logits,
            'attack_type_cls_logits': attack_type_cls_logits,
        }

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        total_loss = self._shared_step(batch, batch_idx, stage = "train")
        return {"loss": total_loss}

    def _shared_step(self, batch, batch_idx, stage: str, return_outputs: bool = False):
        """共享的训练/验证/测试步骤"""
        # ======== Forward ========
        outputs = self(batch)  # 前向传播

        # 检查梯度相关的NaN
        if stage == "train":
            self._check_gradients("前向传播后")

        # 计算损失函数
        losses = self._compute_losses(outputs, batch, stage)

        # 获取batch_size
        batch_size = batch['numeric_features'].shape[0] if 'numeric_features' in batch else 1

        is_malicious_class_loss = losses.get("is_malicious_class_loss")
        # 为了在 tensorboard/logging 中记录指标
        if stage == "train":
            self.log(f"{stage}_is_mal_cls_loss", is_malicious_class_loss, on_step=True, on_epoch=False, sync_dist=False, batch_size=batch_size)

        attack_family_class_loss = losses.get("attack_family_class_loss")
        if stage == "train" and attack_family_class_loss is not None:
            self.log(f"{stage}_att_fam_cls_loss", attack_family_class_loss, on_step=True, on_epoch=False, sync_dist=False)

        attack_type_class_loss = losses.get("attack_type_class_loss")
        if stage == "train" and attack_type_class_loss is not None:
            self.log(f"{stage}_att_tp_cls_loss", attack_type_class_loss, on_step=True, on_epoch=False, sync_dist=False)

        total_loss = losses["total_loss"]
        if stage == "train" and total_loss is not None:
            self.log(f"{stage}_total_loss", total_loss, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True, batch_size=batch_size)

        # 为is_malicious任务计算和记录性能Metrics
        self._compute_and_log_is_malicious_batch_metrics(stage, outputs, batch, batch_size)

        # 为attack_family多分类任务计算和记录性能Metrics
        self._compute_and_log_attack_family_batch_metrics(stage, outputs, batch)

        # 为attack_type多分类任务计算和记录性能Metrics
        self._compute_and_log_attack_type_batch_metrics(stage, outputs, batch)

        # 反向传播前检查
        if stage == "train":
            self._check_gradients("反向传播前")

        if return_outputs:
            return outputs

        return total_loss

    def _compute_and_log_is_malicious_batch_metrics(self, stage, outputs, batch, batch_size):
        is_malicious_cls_logits = outputs['is_malicious_cls_logits']
        is_malicious_probs = torch.sigmoid(is_malicious_cls_logits)
        is_malicious_preds = (is_malicious_probs > 0.5).float()

        is_malicious_labels = batch['is_malicious_label'].to(is_malicious_cls_logits.device).float()
        if is_malicious_labels.dim() == 1:
            is_malicious_labels = is_malicious_labels.unsqueeze(1)

        if stage == "train":
            # 计算 accuracy / precision / recall / f1
            try:
                is_malicious_trues_np = is_malicious_labels.squeeze(1).cpu().numpy()
                is_malicious_preds_np = is_malicious_preds.squeeze(1).cpu().numpy()

                accuracy = accuracy_score(is_malicious_trues_np, is_malicious_preds_np)
                precision = precision_score(is_malicious_trues_np, is_malicious_preds_np, zero_division=0)
                recall = recall_score(is_malicious_trues_np, is_malicious_preds_np, zero_division=0)
                f1 = f1_score(is_malicious_trues_np, is_malicious_preds_np, zero_division=0)
            except Exception as e:
                logger.warning(f"计算指标时出错: {e}")
                accuracy = precision = recall = f1 = 0.0

            self.log(f"{stage}_is_mal_acc", accuracy, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True, batch_size=batch_size)
            self.log(f"{stage}_is_mal_prec", precision, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True, batch_size=batch_size)
            self.log(f"{stage}_is_mal_rec", recall, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True, batch_size=batch_size)
            self.log(f"{stage}_is_mal_f1", f1, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True, batch_size=batch_size)

        elif stage == "val":
            # 保存每个 batch 的结果（必须 detach + cpu）
            self.val_is_malicious_epoch_labels.append(is_malicious_labels.detach().cpu())
            self.val_is_malicious_epoch_probs.append(is_malicious_probs.detach().cpu())
            self.val_is_malicious_epoch_preds.append(is_malicious_preds.detach().cpu())

        elif stage == "test":
            # 保存每个 batch 的结果（必须 detach + cpu）
            self.test_is_malicious_epoch_labels.append(is_malicious_labels.detach().cpu())
            self.test_is_malicious_epoch_probs.append(is_malicious_probs.detach().cpu())
            self.test_is_malicious_epoch_preds.append(is_malicious_preds.detach().cpu())

        else:
            raise ValueError(f"不支持的stage字符串: {stage}")

        return

    def _compute_and_log_attack_family_batch_metrics(self, stage, outputs, batch):
        """
        计算并记录 attack_family 的 batch 级指标
        - 仅在 malicious 样本子集上评估
        - 使用 OVR + macro-F1
        """
        if "attack_family_cls_logits" not in outputs or \
            self.attack_family_classifier is None:
            return

        # logits / labels
        attack_family_logits = outputs["attack_family_cls_logits"]          # [B, K]
        attack_family_labels = batch["attack_family_label"].to(attack_family_logits.device)  # [B, K]

        # 只在 malicious 子集上评估
        is_malicious_mask = (
            batch["is_malicious_label"]
            .to(attack_family_logits.device)
            .view(-1) == 1
        )

        if not is_malicious_mask.any():
            return

        # 仅在「真实恶意流量」样本上评估 attack_family 分类结果。
        # 这里使用的是 is_malicious 的真实标签（ground truth），而不是模型预测结果。
        # 因此评估的是条件分类性能：
        #   P(attack_family | is_malicious = 1)，
        # 而不是“先预测是否恶意，再预测攻击家族”的级联预测流程。
        attack_family_logits = attack_family_logits[is_malicious_mask]
        attack_family_labels = attack_family_labels[is_malicious_mask]

        # OVR threshold
        attack_family_probs = torch.sigmoid(attack_family_logits)
        attack_family_preds = (attack_family_probs > 0.5).int()

        if stage == "train":
            # 这些指标在val和test阶段epoch级展示就够了
            # return
            try:
                # 转 numpy
                labels_np = attack_family_labels.cpu().numpy()
                preds_np = attack_family_preds.cpu().numpy()

                macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
                micro_f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)
            except Exception as e:
                logger.warning(f"计算 attack_family 指标时出错: {e}")
                macro_f1 = micro_f1 = 0.0

            self.log(f"{stage}_att_fam_macro_f1", macro_f1, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)
            self.log(f"{stage}_att_fam_micro_f1", micro_f1, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)

        elif stage == "val":
            self.val_attack_family_epoch_labels.append(attack_family_labels.detach().cpu())
            self.val_attack_family_epoch_probs.append(attack_family_probs.detach().cpu())
            self.val_attack_family_epoch_preds.append(attack_family_preds.detach().cpu())

        elif stage == "test":
            self.test_attack_family_epoch_labels.append(attack_family_labels.detach().cpu())
            self.test_attack_family_epoch_probs.append(attack_family_probs.detach().cpu())
            self.test_attack_family_epoch_preds.append(attack_family_preds.detach().cpu())

        else:
            raise ValueError(f"不支持的stage字符串: {stage}")


    def _compute_and_log_attack_type_batch_metrics(self, stage, outputs, batch):
        """
        计算并记录 attack_type 的 batch 级指标
        """
        if not hasattr(self, "attack_type_classifier") or \
            self.attack_type_classifier is None:
                return

        attack_type_cls_logits = outputs["attack_type_cls_logits"]
        attack_type_label = batch["attack_type_label"]

        # 🔒 强校验 attack_type_label是用long型的一维attack type id来构造标签
        assert attack_type_label.dtype == torch.long
        assert attack_type_label.dim() == 1

        # 只在 malicious 子集上评估
        is_malicious_label = batch["is_malicious_label"].to(attack_type_cls_logits.device).view(-1) == 1

        # 仅在「真实恶意流量」样本上评估 attack_type 分类结果。
        # 这里使用的是 is_malicious 的真实标签（ground truth），而不是模型预测结果。
        attack_type_cls_logits = attack_type_cls_logits[is_malicious_label]
        attack_type_label = attack_type_label[is_malicious_label]

        if stage == "train":
            # 推迟到 val 和 test 数据集的 epoch end 再打印模型性能报告
            # return
            with torch.no_grad():
                if attack_type_label.numel() == 0:
                    attack_type_pred_accuracy = attack_type_cls_logits.sum() * 0.0
                    batch_size = attack_type_cls_logits.shape[0]  # 保证一致
                else:
                    attack_type_pred = torch.argmax(attack_type_cls_logits, dim=-1)
                    attack_type_pred_accuracy = (attack_type_pred == attack_type_label).float().mean()
                    batch_size = attack_type_label.numel()

            # 明确 detach，语义更清晰
            attack_type_pred_accuracy = attack_type_pred_accuracy.detach()
            # ⚠️ 必须所有 rank 可对齐
            self.log(f"{stage}_att_tp_acc", attack_type_pred_accuracy, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True, batch_size=batch_size,)

        elif stage == "val":
            self.val_attack_type_epoch_logits.append(attack_type_cls_logits.detach().cpu())
            self.val_attack_type_epoch_labels.append(attack_type_label.detach().cpu())

        elif stage == "test":
            self.test_attack_type_epoch_logits.append(attack_type_cls_logits.detach().cpu())
            self.test_attack_type_epoch_labels.append(attack_type_label.detach().cpu())

        else:
            raise ValueError(f"不支持的stage字符串: {stage}")


    def _check_gradients(self, stage: str):
        """检查梯度状态"""
        if not self.debug_mode:
            return

        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_count = torch.isnan(param.grad).sum().item()
                    logger.warning(f"⚠️ 梯度NaN: {name} 在 {stage} 有 {nan_count} 个NaN梯度")
                if torch.isinf(param.grad).any():
                    inf_count = torch.isinf(param.grad).sum().item()
                    logger.warning(f"⚠️ 梯度Inf: {name} 在 {stage} 有 {inf_count} 个Inf梯度")

    def on_after_backward(self):
        """反向传播后的钩子"""
        if self.debug_mode:
            self._check_gradients("反向传播后")

            # 梯度裁剪（预防梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def on_validation_epoch_start(self):
        """验证阶段开始：清空缓存"""
        self.val_is_malicious_epoch_labels = []
        self.val_is_malicious_epoch_preds = []
        self.val_is_malicious_epoch_probs = []

        if self.attack_family_classifier is not None:
            self.val_attack_family_epoch_labels = []
            self.val_attack_family_epoch_preds = []
            self.val_attack_family_epoch_probs = []

        if self.attack_type_classifier is not None:
            self.val_attack_type_epoch_logits = []
            self.val_attack_type_epoch_labels = []

    def validation_step(self, batch, batch_idx):
        # 调用 _shared_step 返回 outputs，不重复计算 loss
        total_loss = self._shared_step(batch, batch_idx, stage="val", return_outputs=False)

        # 使用新的通用SHAP分析框架 del by qinyf 2012-12-02
        # if self.cfg.shap.enable_shap:
        #     if self.should_run_shap_analysis(self.current_epoch, batch_idx):
        #         logger.info(f"开始通用SHAP分析，epoch: {self.current_epoch}")

        #         try:
        #             # 执行SHAP分析
        #             shap_results = self.perform_shap_analysis(batch)

        #             if shap_results and not shap_results.get('error'):
        #                 # 调用钩子方法，可以在子类中重写
        #                 self.on_shap_analysis_completed(shap_results)

        #         except Exception as e:
        #             logger.error(f"通用SHAP分析失败: {e}")

        return {"loss": total_loss}

    def on_validation_epoch_end(self):
        """验证集 epoch 结束时统一计算 F1 / precision / recall"""
        self._compute_and_log_is_malicious_epoch_metrics(
            stage="val",
            labels_list=self.val_is_malicious_epoch_labels,
            preds_list=self.val_is_malicious_epoch_preds,
            probs_list=self.val_is_malicious_epoch_probs
        )

        # 清空缓存
        self.val_is_malicious_epoch_labels.clear()
        self.val_is_malicious_epoch_preds.clear()
        self.val_is_malicious_epoch_probs.clear()

        # attack_family 的 epoch-level 逻辑（公共函数）
        if self.attack_family_classifier is not None:
            self._compute_and_log_attack_family_epoch_metrics(
                stage="val",
                labels_list=self.val_attack_family_epoch_labels,
                preds_list=self.val_attack_family_epoch_preds,
            )

            # 清空缓存
            self.val_attack_family_epoch_labels.clear()
            self.val_attack_family_epoch_preds.clear()
            self.val_attack_family_epoch_probs.clear()

        # attack_type 任务的 epoch-level 逻辑（公共函数）
        if self.attack_type_classifier is not None:
            self._compute_and_log_attack_type_epoch_metrics(
                stage="val",
                labels_list=self.val_attack_type_epoch_labels,
                logits_list=self.val_attack_type_epoch_logits,
            )

            # 清空缓存
            self.val_attack_type_epoch_logits.clear()
            self.val_attack_type_epoch_labels.clear()

    def on_test_model_eval(self, *args, **kwargs):
        """
        🚀 破解 PyTorch Lightning 的限制！
        默认情况下，Lightning 会在 test 阶段强制调用 model.eval() 且禁止梯度更新。
        但我们需要在线增量学习，必须允许模型内的某些层（如 Adapter）进入 train() 模式。
        """
        super().on_test_model_eval(*args, **kwargs)

        # 强制将需要微调的分类头解冻并设为训练模式，以便接受梯度更新
        if hasattr(self, 'drift_detector') and self.drift_detector.enabled:
            if hasattr(self, 'is_malicious_classifier'):
                self.is_malicious_classifier.train()
                for param in self.is_malicious_classifier.parameters():
                    param.requires_grad = True

            if hasattr(self, 'attack_family_classifier') and self.attack_family_classifier is not None:
                self.attack_family_classifier.train()
                for param in self.attack_family_classifier.parameters():
                    param.requires_grad = True

            logger.info("🔓 已在 Test 阶段强制解冻分类头并开启 Train 模式，准备接受在线适应更新！")

    def on_test_epoch_start(self):
        """测试阶段开始时初始化全局存储"""
        self.test_is_malicious_epoch_labels = []
        self.test_is_malicious_epoch_preds = []
        self.test_is_malicious_epoch_probs = []

        if self.attack_family_classifier is not None:
            self.test_attack_family_epoch_labels = []
            self.test_attack_family_epoch_preds = []
            self.test_attack_family_epoch_probs = []

        if self.attack_type_classifier is not None:
            self.test_attack_type_epoch_logits = []
            self.test_attack_type_epoch_labels = []

        # 3. 组件重置 added by qinyf 2025-12-02
        if self.cfg.shap.enable_shap:
            self.shap_analyzer.reset()

    def test_step(self, batch, batch_idx):
        # 1. 🟢 【核心逻辑：先知先觉】先用当前 batch 的特征进行漂移检测和适应
        if hasattr(self, 'drift_detector') and self.drift_detector.enabled:
            # 为了不影响主计算图，我们用 torch.no_grad 提取特征来做检测
            with torch.no_grad():
                pre_outputs = self(batch)
                features = pre_outputs['multiview_embeddings'].detach()

            adapt_enabled = self.cfg.concept_drift.get('adaptation', {}).get('enabled', True)

            # 如果触发漂移，这里面会自动进行 `with torch.enable_grad()` 的联合微调
            is_drift = self.drift_detector.update_batch(features, batch, adapt=adapt_enabled)

            if is_drift:
                self.test_drift_steps.append(batch_idx)
                logger.info(f"🚨 Test Step {batch_idx}: 记录漂移事件，模型已完成在线适应!")

        # 2. 🟢 【正常评估】模型可能已经在上面被微调过了，现在使用最新权重进行真正的推理和评估
        outputs = self._shared_step(batch, batch_idx, stage="test", return_outputs=True)

        # 收集每一批次的准确率用于最后绘制漂移曲线图 (如果在 Test_end 里想画图的话)
        preds = (torch.sigmoid(outputs['is_malicious_cls_logits']) > 0.5).float().squeeze()
        trues = batch['is_malicious_label'].float().view(-1)
        acc = (preds == trues).float().mean().item()
        self.test_step_accuracies.append(acc)

        if hasattr(self.cfg, 'shap') and self.cfg.shap.enable_shap:
            if hasattr(self, 'shap_analyzer'):
                self.shap_analyzer.collect_batch(batch)

        return None

    def on_test_epoch_end(self):
        """测试阶段结束，汇总全局指标，多 GPU 下支持同步"""
        # is_malicious 任务的 epoch-level 逻辑（公共函数）
        self._compute_and_log_is_malicious_epoch_metrics(
            stage="test",
            labels_list=self.test_is_malicious_epoch_labels,
            preds_list=self.test_is_malicious_epoch_preds,
            probs_list=self.test_is_malicious_epoch_probs,
        )

        # 清空存储
        self.test_is_malicious_epoch_labels.clear()
        self.test_is_malicious_epoch_preds.clear()
        self.test_is_malicious_epoch_probs.clear()

        if self.attack_family_classifier is not None:
            # attack_family 任务的 epoch-level 逻辑（公共函数）
            self._compute_and_log_attack_family_epoch_metrics(
                stage="test",
                labels_list=self.test_attack_family_epoch_labels,
                preds_list=self.test_attack_family_epoch_preds,
            )

            # 清空存储
            self.test_attack_family_epoch_labels.clear()
            self.test_attack_family_epoch_preds.clear()
            self.test_attack_family_epoch_probs.clear()

        if self.attack_type_classifier is not None:
            # attack_type 任务的 epoch-level 逻辑（公共函数）
            self._compute_and_log_attack_type_epoch_metrics(
                stage="test",
                labels_list=self.test_attack_type_epoch_labels,
                logits_list=self.test_attack_type_epoch_logits,
            )

            # 清空缓存
            self.test_attack_type_epoch_logits.clear()
            self.test_attack_type_epoch_labels.clear()

        # 5. 组件执行分析  added by qinyf 2025-12-02
        if self.cfg.shap.enable_shap:
            self.shap_analyzer.finalize()

    def _compute_and_log_is_malicious_epoch_metrics(
        self,
        stage: str,
        labels_list: List[torch.Tensor],
        preds_list:  List[torch.Tensor],
        probs_list:  List[torch.Tensor],
    ):
        if not labels_list or len(labels_list) == 0:
            logger.warning(f"[{stage}] is_malicious labels empty")

            # For DDP safe
            self.log(f"{stage}_is_mal_acc", 0.0, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_is_mal_prec", 0.0, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_is_mal_rec", 0.0, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_is_mal_f1", 0.0, on_epoch=True, prog_bar=False, sync_dist=True)

            return

        labels = torch.cat(labels_list, dim=0)
        preds  = torch.cat(preds_list, dim=0)
        probs  = torch.cat(probs_list, dim=0)

        labels_np = labels.cpu().numpy()
        preds_np  = preds.cpu().numpy()
        probs_np  = probs.cpu().numpy()

        accuracy  = accuracy_score(labels_np, preds_np)
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall  = recall_score(labels_np, preds_np, zero_division=0)
        f1   = f1_score(labels_np, preds_np, zero_division=0)

        self.log(f"{stage}_is_mal_acc", accuracy, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_is_mal_prec", precision, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_is_mal_rec", recall, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_is_mal_f1", f1, on_epoch=True, prog_bar=False, sync_dist=True)

        # 🔴🔴 只在主进程上输出该任务的val/test阶段的完整报告
        if self.trainer.is_global_zero and labels_list and len(labels_list) > 0:
            # 🔴 val 阶段的简要报告
            if stage == "val":
                logger.info(f"[Epoch {self.current_epoch}] val 阶段的简要报告：")
                logger.info(
                    f"[Epoch {self.current_epoch}] "
                    f"val_is_mal_acc={accuracy:.4f}, "
                    f"val_is_mal_prec={precision:.4f}, "
                    f"val_is_mal_rec={recall:.4f}, "
                    f"val_is_mal_f1={f1:.4f}"
                )
            # 🔴 test 阶段的最终报告
            elif stage == "test":
                logger.info("=" * 60)
                logger.info("🤖 最佳模型的is_malicious任务的测试报告")
                logger.info("=" * 60)

                # ---- 整理详细的模型性能报告 ----
                logger.info(f"📊 is_malicious任务的基础指标:")
                logger.info(f"   准确率: {accuracy:.4f}")
                logger.info(f"   精确率: {precision:.4f}")
                logger.info(f"   召回率: {recall:.4f}")
                logger.info(f"   F1分数: {f1:.4f}")

                # 分类报告
                try:
                    report = classification_report(labels_np, preds_np, digits=4, target_names=['正常', '恶意'])
                    logger.info("📋 is_malicious任务的详细分类报告:")
                    logger.info("\n" + report)
                except Exception as e:
                    logger.warning(f"is_malicious任务的分类报告生成失败: {e}")

                # 混淆矩阵
                try:
                    cm = confusion_matrix(labels_np, preds_np)
                    logger.info("🎯 is_malicious任务的混淆矩阵:")
                    logger.info(f"\n{cm}")
                except Exception as e:
                    logger.warning(f"is_malicious任务的混淆矩阵生成失败: {e}")

                # ROC-AUC 和 PR曲线
                try:
                    auc = roc_auc_score(labels_np, probs_np)
                    avg_precision = average_precision_score(labels_np, probs_np)
                    logger.info(f"📈 is_malicious任务的高级指标:")
                    logger.info(f"   ROC-AUC: {auc:.4f}")
                    logger.info(f"   Average Precision: {avg_precision:.4f}")
                except Exception as e:
                    logger.warning(f"高级指标计算失败: {e}")

                # 样本统计
                logger.info(f"📊 样本的is_malicious标签数量统计:")
                logger.info(f"   总样本数: {len(labels)}")
                logger.info(f"   正样本数: {labels_np.sum()}")
                logger.info(f"   负样本数: {len(labels) - labels_np.sum()}")
                logger.info(f"   正样本比例: {labels_np.mean():.2%}")

                logger.info("=" * 60)


    def _compute_and_log_attack_family_epoch_metrics(self, stage: str, labels_list, preds_list):
        """
        在 epoch 结束时统一计算并 log attack_family 的整体指标（仅 malicious）.
        用于 val / test 阶段（train 不走这里）
        """
        assert stage in ("val", "test"), f"非法 stage={stage}"

        # 计算指标
        if labels_list and len(labels_list) > 0:
            labels_np = torch.cat(labels_list, dim=0).cpu().numpy()
            preds_np = torch.cat(preds_list, dim=0).cpu().numpy()

            if len(labels_np) == 0:
                logger.warning(f"[{stage}] 当前批次没有恶意样本(常见于Sanity Check)，跳过此指标计算。")
                return

            macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
            micro_f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)

        else:
            logger.warning(f"[{stage}] attack_family: 无可用样本")
            macro_f1 = torch.tensor(0.0, device=self.device)
            micro_f1 = torch.tensor(0.0, device=self.device)

        # log（epoch-level，不要 batch_size）
        self.log(f"{stage}_att_fam_macro_f1", macro_f1, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log(f"{stage}_att_fam_micro_f1", micro_f1, on_epoch=True, sync_dist=True, prog_bar=False)

        # 🔴🔴 只在主进程上输出该任务的val/test阶段的完整报告
        if self.trainer.is_global_zero and labels_list and len(labels_list) > 0:
            # 🔴 val 阶段的简要报告
            if stage == "val":
                logger.info(
                    f"[Epoch {self.current_epoch}] "
                    f"val_att_fam_macro_f1={macro_f1:.4f}, "
                    f"val_att_fam_micro_f1={micro_f1:.4f}"
                )

            # 🔴 test 阶段的最终报告
            elif stage == "test":
                logger.info("=" * 60)
                logger.info("🤖 attack_family 任务测试报告（简要）")
                logger.info(f"macro_f1={macro_f1:.4f}, micro_f1={micro_f1:.4f}")

                try:
                    report = classification_report(
                        labels_np,
                        preds_np,
                        target_names=self.attack_family_names,
                        digits=4,
                        zero_division=0,
                    )
                    logger.info("📋 attack_family 任务的分类报告:")
                    logger.info("\n" + report)
                except Exception as e:
                    logger.warning(f"attack_family 分类报告生成失败: {e}")

                per_attack_family_f1 = f1_score(labels_np, preds_np, average=None, zero_division=0)
                logger.info("📊 attack_family per-class F1:")

                for name, f1v in zip(self.attack_family_names, per_attack_family_f1):
                    logger.info(f"  {name:20s}: F1={f1v:.4f}")

                logger.info("=" * 60)


    def _compute_and_log_attack_type_epoch_metrics(self, stage: str, labels_list, logits_list):
        """
        在 epoch 结束时计算 attack_type 的整体指标（仅 malicious）。
        用于 val / test 阶段（train 不走这里）
        """
        assert stage in ("val", "test"), f"非法 stage={stage}"
        num_classes = len(self.attack_type_names)

        # === overall metrics ===
        if labels_list and len(labels_list) > 0:
            labels = torch.cat(labels_list, dim=0)
            logits = torch.cat(logits_list, dim=0)

            # === predictions ===
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            if len(labels_np) == 0:
                logger.warning(f"[{stage}] 当前批次没有恶意样本(常见于Sanity Check)，跳过此指标计算。")
                return

            macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
            micro_f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)

            # === per-class F1 (diagnostic) ===
            class_f1 = f1_score(labels_np, preds_np, average=None, labels=list(range(num_classes)), zero_division=0,)

        else:
            logger.warning(f"[{stage}] attack_type: 无可用样本")
            macro_f1 = torch.tensor(0.0, device=self.device)
            micro_f1 = torch.tensor(0.0, device=self.device)
            class_f1 = [0.0] * num_classes

        self.log(f"{stage}_att_tp_macro_f1", macro_f1, on_epoch=True, sync_dist=True, prog_bar=False,)
        self.log(f"{stage}_att_tp_micro_f1", micro_f1, on_epoch=True, sync_dist=True, prog_bar=False,)

        for idx, f1 in enumerate(class_f1):
            class_name = self.attack_type_names[idx]
            self.log(f"{stage}_att_tp_f1/{class_name}", f1, on_epoch=True, sync_dist=True, prog_bar=False,)

        # 🔴 test 阶段的简要最终报告
        if self.trainer.is_global_zero and labels_list and len(labels_list) > 0:
            # 🔴 val 阶段的简要报告
            if stage == "val":
                logger.info(
                    f"[Epoch {self.current_epoch}] "
                    f"val_att_tp_macro_f1={macro_f1:.4f}, "
                    f"val_att_tp_micro_f1={micro_f1:.4f}"
                )

            elif stage == "test":
                logger.info("=" * 60)
                logger.info("🤖 attack_type 任务测试报告（简要）")

                macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
                micro_f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)

                logger.info(f"macro_f1={macro_f1:.4f}, micro_f1={micro_f1:.4f}")

                logger.info("📋 attack_type 任务的分类报告:")
                logger.info(
                    "\n" + classification_report(
                        labels_np,
                        preds_np,
                        labels=list(range(num_classes)),
                        target_names=self.attack_type_names,
                        digits=4,
                        zero_division=0,
                    )
                )

                logger.info("📊 attack_type per-class F1:")
                for idx, f1 in enumerate(class_f1):
                    class_name = self.attack_type_names[idx]
                    logger.info(f"  {class_name:<20}: F1={f1:.4f}")

                logger.info("=" * 60)


    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 使用 datamodule 获取训练集长度
        if self.trainer and hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            train_loader = self.trainer.datamodule.train_dataloader()
            total_steps = len(train_loader) * self.trainer.max_epochs
        else:
            # 如果 trainer 还未初始化，使用默认步数（可根据 cfg 设置）
            total_steps = self.cfg.optimizer.default_total_steps

        warmup_steps = int(total_steps * self.cfg.optimizer.warmup_ratio)

        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
            eps=getattr(self.cfg.optimizer, 'eps', 1e-8)
        )

        scheduler_type = self.cfg.scheduler.type
        if scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_type == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:  # linear
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }


    def _fuse_multi_views(self, sequence_outputs, text_outputs, tabular_outputs):
        """使用配置的融合方法融合多视图特征"""
        view_embeddings = []

        # 收集所有启用的视图
        view_embeddings.append(tabular_outputs)  # 数值特征必选 + 域名特征可选

        if self.text_features_enabled:
            view_embeddings.append(text_outputs)

        if self.sequence_features_enabled:
            view_embeddings.append(sequence_outputs)

        # 使用配置的融合方法
        if len(view_embeddings) > 1:
            fused_embedding = self.fusion_layer(view_embeddings)
        else:
            # 单视图情况，直接使用
            fused_embedding = view_embeddings[0]

        return fused_embedding
