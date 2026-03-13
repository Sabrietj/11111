"""
多模型适配器 - 支持多种SSL模型变体的统一适配器
"""

import logging
from typing import Dict, List, Any, Optional, Type
import re

logger = logging.getLogger(__name__)

class MultiModelSHAPAdapter:
    """支持多种SSL模型变体的统一适配器"""
    
    SUPPORTED_MODELS = {
        'flow_bert_multiview': {
            'config_path': 'src/models/flow_bert_multiview/config/flow_bert_multiview_config.yaml',
            'dimension_extractor': 'FlowBertMultiviewDimensionExtractor',
            'feature_classifier': 'FlowBertMultiviewClassifier',
            'description': '多视图BERT模型，支持序列、文本、表格特征融合'
        },
        'flow_bert_ssl_mlm': {
            'config_path': 'src/models/flow_bert_ssl_mlm/config/flow_bert_ssl_mlm_config.yaml',
            'dimension_extractor': 'SSLMMLMDimensionExtractor', 
            'feature_classifier': 'SSLMMLMClassifier',
            'description': '带掩码语言建模的SSL模型'
        },
        'flow_bert_ssl_seq2stat': {
            'config_path': 'src/models/flow_bert_ssl_seq2stat/config/flow_bert_ssl_seq2stat_config.yaml',
            'dimension_extractor': 'SSLSeq2StatDimensionExtractor',
            'feature_classifier': 'SSLSeq2StatClassifier',
            'description': '序列到统计特征的SSL模型'
        },
        'flow_bert_ssl': {
            'config_path': 'src/models/flow_bert_ssl/config/flow_bert_ssl_config.yaml',
            'dimension_extractor': 'SSLBaseDimensionExtractor',
            'feature_classifier': 'SSLBaseClassifier',
            'description': '基础SSL模型'
        },
        'autoencoder': {
            'config_path': 'src/models/autoencoder/config/autoencoder_config.yaml',
            'dimension_extractor': 'AutoencoderDimensionExtractor',
            'feature_classifier': 'AutoencoderClassifier',
            'description': '自编码器模型'
        }
    }
    
    def __init__(self):
        self.supported_models = self.SUPPORTED_MODELS.copy()
        self.custom_models = {}
    
    def auto_detect_model_type(self, model) -> str:
        """自动检测模型类型"""
        batch_keys = []
        
        # 尝试获取模型的批次数据键（如果有相关方法或属性）
        if hasattr(model, 'get_batch_keys'):
            batch_keys = model.get_batch_keys()
        elif hasattr(model, 'cfg'):
            # 从配置中推断
            cfg = model.cfg
            if hasattr(cfg.data, 'sequence_features') and cfg.data.sequence_features.enabled:
                batch_keys.extend(['iat_times', 'payload_sizes', 'sequence_mask'])
            if hasattr(cfg.data, 'text_features') and cfg.data.text_features.enabled:
                batch_keys.extend(['ssl_server_name', 'dns_query', 'cert0_subject'])
            if hasattr(cfg.data, 'domain_name_embedding_features') and cfg.data.domain_name_embedding_features.enabled:
                batch_keys.extend(['domain_embedding_features'])
        
        keys = set(batch_keys)
        
        # 优先级检测顺序
        if 'sequence_mlm_mask' in keys:
            detected_type = 'flow_bert_ssl_mlm'
            reason = "检测到 sequence_mlm_mask 特征"
        elif 'seq2stat_targets' in keys:
            detected_type = 'flow_bert_ssl_seq2stat'
            reason = "检测到 seq2stat_targets 特征"
        elif 'ssl_features' in keys:
            detected_type = 'flow_bert_ssl'
            reason = "检测到 ssl_features 特征"
        elif 'combined_text' in keys and 'iat_times' in keys:
            detected_type = 'flow_bert_multiview'
            reason = "检测到 combined_text 和 iat_times 特征"
        elif any('latent_' in str(key) or 'reconstruction_' in str(key) for key in keys):
            detected_type = 'autoencoder'
            reason = "检测到 latent 或 reconstruction 特征"
        else:
            # 默认检测
            model_class_name = model.__class__.__name__.lower()
            if 'multiview' in model_class_name:
                detected_type = 'flow_bert_multiview'
                reason = "基于类名推断"
            elif 'ssl' in model_class_name and 'mlm' in model_class_name:
                detected_type = 'flow_bert_ssl_mlm'
                reason = "基于类名推断"
            elif 'ssl' in model_class_name and 'seq2stat' in model_class_name:
                detected_type = 'flow_bert_ssl_seq2stat'
                reason = "基于类名推断"
            elif 'ssl' in model_class_name:
                detected_type = 'flow_bert_ssl'
                reason = "基于类名推断"
            elif 'autoencoder' in model_class_name:
                detected_type = 'autoencoder'
                reason = "基于类名推断"
            else:
                detected_type = 'flow_bert_multiview'  # 默认
                reason = "使用默认类型"
        
        logger.info(f"自动检测模型类型: {detected_type} ({reason})")
        return detected_type
    
    def get_model_info(self, model_type: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        if model_type in self.supported_models:
            return self.supported_models[model_type]
        elif model_type in self.custom_models:
            return self.custom_models[model_type]
        else:
            logger.warning(f"不支持的模型类型: {model_type}")
            return None
    
    def get_supported_model_types(self) -> List[str]:
        """获取所有支持的模型类型"""
        return list(self.supported_models.keys()) + list(self.custom_models.keys())
    
    def add_model_support(self, model_type: str, model_config: Dict[str, Any]):
        """添加新的模型支持"""
        required_fields = ['config_path', 'dimension_extractor', 'feature_classifier']
        
        for field in required_fields:
            if field not in model_config:
                raise ValueError(f"模型配置缺少必需字段: {field}")
        
        self.custom_models[model_type] = model_config
        logger.info(f"添加新模型支持: {model_type}")
    
    def create_dimension_extractor(self, model_type: str, cfg):
        """创建对应的维度提取器"""
        model_info = self.get_model_info(model_type)
        if not model_info:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        extractor_name = model_info['dimension_extractor']
        
        # 根据提取器名称创建对应的提取器实例
        if extractor_name == 'FlowBertMultiviewDimensionExtractor':
            from .dimension_extractor import ConfigDimensionExtractor
            return ConfigDimensionExtractor(cfg)
        else:
            # 默认使用 ConfigDimensionExtractor
            from .dimension_extractor import ConfigDimensionExtractor
            logger.warning(f"使用默认维度提取器替代: {extractor_name}")
            return ConfigDimensionExtractor(cfg)
    
    def create_feature_classifier(self, model_type: str):
        """创建对应的特征分类器"""
        model_info = self.get_model_info(model_type)
        if not model_info:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        classifier_name = model_info['feature_classifier']
        
        # 根据分类器名称创建对应的分类器实例
        if classifier_name in ['FlowBertMultiviewClassifier', 'SSLBaseClassifier', 
                              'SSLMMLMClassifier', 'SSLSeq2StatClassifier']:
            from .five_tier_classifier import FiveTierFeatureClassifier
            return FiveTierFeatureClassifier()
        else:
            # 默认使用 FiveTierFeatureClassifier
            from .five_tier_classifier import FiveTierFeatureClassifier
            logger.warning(f"使用默认特征分类器替代: {classifier_name}")
            return FiveTierFeatureClassifier()
    
    def get_feature_mapping_for_model(self, model_type: str) -> Dict[str, List[str]]:
        """获取特定模型的特征映射配置"""
        mappings = {
            'flow_bert_multiview': {
                'numeric_patterns': ['flowmeter\\.', 'numeric_features', 'domain_embedding_features'],
                'sequence_patterns': ['iat_times', 'payload_sizes', 'packet_directions'],
                'text_patterns': ['ssl_server_name', 'dns_query', 'cert0_subject', 'cert0_issuer'],
                'exclude_patterns': ['sequence_mask', 'is_malicious_labels', 'split', 'flow_uid', 'labels', 'target', 'enabled', 'idx', 'combined_text'],
                'decompose_combined_text': True
            },
            'flow_bert_ssl_mlm': {
                'numeric_patterns': ['flowmeter\\.', 'numeric_features', 'domain_embedding_features', 'reconstruction_', 'mlm_features'],
                'sequence_patterns': ['iat_times', 'payload_sizes', 'packet_directions', 'sequence_', 'mlm_sequence', 'masked_sequence'],
                'text_patterns': ['ssl_server_name', 'dns_query', 'cert0_subject', 'cert0_issuer', 'masked_text'],
                'exclude_patterns': ['sequence_mask', 'is_malicious_labels', 'split', 'flow_uid', 'labels', 'target', 'enabled', 'mlm_mask', 'idx', 'sequence_mlm_mask', 'combined_text'],
                'decompose_combined_text': True
            },
            'flow_bert_ssl_seq2stat': {
                'numeric_patterns': ['flowmeter\\.', 'numeric_features', 'domain_embedding_features', 'reconstruction_', 'seq2stat_features'],
                'sequence_patterns': ['iat_times', 'payload_sizes', 'packet_directions', 'sequence_', 'seq2stat_sequence'],
                'text_patterns': ['ssl_server_name', 'dns_query', 'cert0_subject', 'cert0_issuer'],
                'exclude_patterns': ['sequence_mask', 'is_malicious_labels', 'split', 'flow_uid', 'labels', 'target', 'enabled', 'seq2stat_mask', 'idx', 'combined_text'],
                'decompose_combined_text': True
            },
            'flow_bert_ssl': {
                'numeric_patterns': ['ssl_features', 'reconstruction_', 'flowmeter\\.', 'numeric_features', 'domain_embedding_features'],
                'sequence_patterns': ['iat_times', 'payload_sizes', 'packet_directions', 'ssl_sequence'],
                'text_patterns': ['ssl_server_name', 'dns_query', 'cert0_subject', 'cert0_issuer'],
                'exclude_patterns': ['ssl_mask', 'sequence_mask', 'is_malicious_labels', 'split', 'flow_uid', 'labels', 'target', 'enabled', 'idx', 'combined_text'],
                'decompose_combined_text': True
            },
            'autoencoder': {
                'numeric_patterns': ['feature_', 'value_', 'latent_', 'reconstruction_'],
                'sequence_patterns': [],
                'text_patterns': [],
                'exclude_patterns': ['mask', 'index', 'labels', 'target', 'enabled', 'idx'],
                'decompose_combined_text': False
            }
        }
        
        return mappings.get(model_type, mappings['flow_bert_multiview'])
    
    def validate_model_compatibility(self, model, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证模型与数据的兼容性"""
        compatibility_result = {
            'is_compatible': True,
            'model_type': None,
            'issues': [],
            'suggestions': []
        }
        
        # 自动检测模型类型
        model_type = self.auto_detect_model_type(model)
        compatibility_result['model_type'] = model_type
        
        # 获取模型信息
        model_info = self.get_model_info(model_type)
        if not model_info:
            compatibility_result['is_compatible'] = False
            compatibility_result['issues'].append(f"不支持的模型类型: {model_type}")
            return compatibility_result
        
        # 验证批次数据与模型类型的兼容性
        feature_mapping = self.get_feature_mapping_for_model(model_type)
        
        # 检查必需的特征
        batch_keys = set(batch_data.keys())
        
        # 检查被排除的特征
        exclude_patterns = feature_mapping.get('exclude_patterns', [])
        excluded_features = []
        for pattern in exclude_patterns:
            for key in batch_keys:
                if re.search(pattern, key):
                    excluded_features.append(key)
        
        # 检查预期的特征类型
        expected_patterns = {
            'numeric': feature_mapping.get('numeric_patterns', []),
            'sequence': feature_mapping.get('sequence_patterns', []),
            'text': feature_mapping.get('text_patterns', [])
        }
        
        feature_presence = {}
        for feature_type, patterns in expected_patterns.items():
            found_features = []
            for pattern in patterns:
                for key in batch_keys:
                    if re.search(pattern, key):
                        found_features.append(key)
            feature_presence[feature_type] = found_features
        
        # 生成兼容性报告
        if model_type in ['flow_bert_multiview', 'flow_bert_ssl'] and not feature_presence.get('text'):
            compatibility_result['issues'].append(f"{model_type}需要文本特征，但批次数据中缺失")
            compatibility_result['suggestions'].append("检查数据是否包含 ssl_server_name, dns_query 等文本特征")
        
        if model_type in ['flow_bert_multiview', 'flow_bert_ssl'] and not feature_presence.get('sequence'):
            compatibility_result['issues'].append(f"{model_type}需要序列特征，但批次数据中缺失")
            compatibility_result['suggestions'].append("检查数据是否包含 iat_times, payload_sizes 等序列特征")
        
        if model_type == 'flow_bert_ssl_mlm' and 'sequence_mlm_mask' not in batch_keys:
            compatibility_result['issues'].append("flow_bert_ssl_mlm需要 sequence_mlm_mask 特征")
        
        if model_type == 'flow_bert_ssl_seq2stat' and 'seq2stat_targets' not in batch_keys:
            compatibility_result['issues'].append("flow_bert_ssl_seq2stat需要 seq2stat_targets 特征")
        
        compatibility_result['feature_presence'] = feature_presence
        compatibility_result['excluded_features'] = excluded_features
        
        if compatibility_result['issues']:
            compatibility_result['is_compatible'] = False
        
        return compatibility_result