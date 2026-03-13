"""
五大特征类别分类器 - 基于配置文件的智能特征分类
"""

import logging
from typing import Dict, List, Any
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class FiveTierFeatureClassifier:
    """基于配置文件的五大特征类别智能分类器"""
    
    FEATURE_HIERARCHY = {
        'numeric_features': {
            'level': 1,
            'target_for': 'both',  # 饼图和柱状图
            'config_sources': ['data.tabular_features.numeric_features']
        },
        'categorical_features': {
            'level': 1,
            'target_for': 'pie_chart', 
            'config_sources': ['data.tabular_features.categorical_features']
        },
        'sequence_features': {
            'level': 1,
            'target_for': 'pie_chart',
            'config_sources': ['data.sequence_features']
        },
        'text_features': {
            'level': 1,
            'target_for': 'pie_chart',
            'config_sources': ['data.text_features']
        },
        'domain_embedding_features': {
            'level': 1,
            'target_for': 'pie_chart',
            'config_sources': ['data.domain_name_embedding_features']
        }
    }
    
    def __init__(self):
        self.feature_hierarchy = self.FEATURE_HIERARCHY.copy()
    
    def classify_from_config(self, cfg: DictConfig, batch_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """基于配置文件进行特征分类"""
        classification = {
            'numeric_features': [],
            'categorical_features': [], 
            'sequence_features': [],
            'text_features': [],
            'domain_embedding_features': []
        }
        
        # 1. 数值特征：从配置文件提取
        if hasattr(cfg.data.tabular_features, 'numeric_features'):
            num_cfg = cfg.data.tabular_features.numeric_features
            
            # flow_features → 流量计特征
            if hasattr(num_cfg, 'flow_features'):
                classification['numeric_features'].extend([
                    f"flow_meter:{feat}" for feat in num_cfg.flow_features
                ])
            
            # x509_features → 证书加密特征  
            if hasattr(num_cfg, 'x509_features'):
                classification['numeric_features'].extend([
                    f"x509_crypto:{feat}" for feat in num_cfg.x509_features
                ])
            
            # dns_features → DNS时序特征
            if hasattr(num_cfg, 'dns_features'):
                classification['numeric_features'].extend([
                    f"dns_timing:{feat}" for feat in num_cfg.dns_features
                ])
        
        # 2. 类别特征：从配置文件提取
        if hasattr(cfg.data.tabular_features, 'categorical_features'):
            cat_features = cfg.data.tabular_features.categorical_features
            classification['categorical_features'] = [
                f"category:{feat}" for feat in cat_features
            ]
        
        # 3. 域名嵌入特征：支持配置文件和实际维度的动态匹配
        if hasattr(cfg.data, 'domain_name_embedding_features'):
            if hasattr(cfg.data.domain_name_embedding_features, 'column_list'):
                domain_list = cfg.data.domain_name_embedding_features.column_list
                # 先使用配置文件中的特征名
                domain_features = [f"domain_freq:{feat}" for feat in domain_list]
                
                # 检查批次数据中的实际维度，如果比配置多，则扩展特征名
                if 'domain_embedding_features' in batch_data:
                    actual_dim = batch_data['domain_embedding_features'].shape[-1]
                    if actual_dim > len(domain_features):
                        # 为额外的维度添加通用特征名
                        for i in range(len(domain_features), actual_dim):
                            domain_features.append(f"domain_feature_{i}")
                
                classification['domain_embedding_features'] = domain_features
        
        # 4. 序列特征：从配置和批次数据验证
        if hasattr(cfg.data, 'sequence_features') and cfg.data.sequence_features.enabled:
            seq_cfg = cfg.data.sequence_features
            classification['sequence_features'] = [
                f"sequence:{seq_cfg.packet_iat}",
                f"sequence:{seq_cfg.packet_payload}", 
                f"sequence:{seq_cfg.packet_direction}",
                "sequence:position_embedding"
            ]
        
        # 5. 文本特征：从配置文件提取
        if hasattr(cfg.data, 'text_features') and cfg.data.text_features.enabled:
            text_cfg = cfg.data.text_features
            # 安全地添加文本特征，只添加存在的配置
            text_features_list = []
            
            if hasattr(text_cfg, 'dns_query'):
                text_features_list.append(f"text:dns_query:{text_cfg.dns_query}")
            if hasattr(text_cfg, 'ssl_server_name'):
                text_features_list.append(f"text:ssl_server:{text_cfg.ssl_server_name}")
            if hasattr(text_cfg, 'cert0_subject'):
                text_features_list.append(f"text:cert0_subject:{text_cfg.cert0_subject}")
            if hasattr(text_cfg, 'cert0_issuer'):
                text_features_list.append(f"text:cert0_issuer:{text_cfg.cert0_issuer}")
            if hasattr(text_cfg, 'cert0_san_dns'):
                text_features_list.append(f"text:cert0_san_dns:{text_cfg.cert0_san_dns}")
            if hasattr(text_cfg, 'cert1_subject'):
                text_features_list.append(f"text:cert1_subject:{text_cfg.cert1_subject}")
            if hasattr(text_cfg, 'cert1_issuer'):
                text_features_list.append(f"text:cert1_issuer:{text_cfg.cert1_issuer}")
            if hasattr(text_cfg, 'cert1_san_dns'):
                text_features_list.append(f"text:cert1_san_dns:{text_cfg.cert1_san_dns}")
            if hasattr(text_cfg, 'cert2_subject'):
                text_features_list.append(f"text:cert2_subject:{text_cfg.cert2_subject}")
            if hasattr(text_cfg, 'cert2_issuer'):
                text_features_list.append(f"text:cert2_issuer:{text_cfg.cert2_issuer}")
            if hasattr(text_cfg, 'cert2_san_dns'):
                text_features_list.append(f"text:cert2_san_dns:{text_cfg.cert2_san_dns}")
            if hasattr(text_cfg, 'dns_answers'):
                text_features_list.append(f"text:dns_answers:{text_cfg.dns_answers}")
            
            classification['text_features'] = text_features_list
        
        # 6. 验证分类结果与批次数据的一致性
        validated_classification = self._validate_with_batch_data(classification, batch_data)
        
        logger.info(f"五大特征类别分类完成:")
        for category, features in validated_classification.items():
            logger.info(f"  {category}: {len(features)} 个特征")
        
        return validated_classification
    
    def _validate_with_batch_data(self, classification: Dict[str, List[str]], 
                                batch_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """验证分类结果与实际批次数据的一致性"""
        validated_classification = classification.copy()
        
        # 检查数值特征是否在批次数据中
        if 'numeric_features' in batch_data:
            actual_numeric_shape = batch_data['numeric_features'].shape[-1]
            expected_numeric_count = len(classification['numeric_features'])
            
            if actual_numeric_shape != expected_numeric_count:
                logger.warning(
                    f"数值特征数量不匹配: 配置={expected_numeric_count}, 实际={actual_numeric_shape}"
                )
                # 使用实际维度信息调整分类结果
                if actual_numeric_shape > 0:
                    validated_classification['numeric_features'] = [
                        f"numeric_feature_{i}" for i in range(actual_numeric_shape)
                    ]
        
        # 检查域名嵌入特征
        if 'domain_embedding_features' in batch_data:
            actual_domain_shape = batch_data['domain_embedding_features'].shape[-1]
            expected_domain_count = len(classification['domain_embedding_features'])
            
            if actual_domain_shape != expected_domain_count:
                logger.warning(
                    f"域名嵌入特征数量不匹配: 配置={expected_domain_count}, 实际={actual_domain_shape}"
                )
                if actual_domain_shape > 0:
                    validated_classification['domain_embedding_features'] = [
                        f"domain_feature_{i}" for i in range(actual_domain_shape)
                    ]
        
        # 验证序列特征配置
        sequence_enabled = False
        if ('iat_times' in batch_data and 'payload_sizes' in batch_data and 
            'sequence_mask' in batch_data):
            sequence_enabled = True
        
        if not sequence_enabled and len(validated_classification['sequence_features']) > 0:
            logger.warning("配置中启用了序列特征，但批次数据中缺少相关字段")
            validated_classification['sequence_features'] = []
        
        # 验证文本特征配置
        text_features_present = any(key in batch_data for key in [
            'ssl_server_name', 'dns_query', 'cert0_subject', 'cert0_issuer'
        ])
        
        if not text_features_present and len(validated_classification['text_features']) > 0:
            logger.warning("配置中启用了文本特征，但批次数据中缺少相关字段")
            validated_classification['text_features'] = []
        
        return validated_classification
    
    def get_feature_hierarchy_info(self) -> Dict[str, Any]:
        """获取特征层次结构信息"""
        return {
            'hierarchy': self.feature_hierarchy,
            'total_categories': len(self.feature_hierarchy),
            'categories_for_bar_chart': [
                cat for cat, info in self.feature_hierarchy.items() 
                if info['target_for'] in ['both', 'bar_chart']
            ],
            'categories_for_pie_chart': [
                cat for cat, info in self.feature_hierarchy.items() 
                if info['target_for'] in ['both', 'pie_chart']
            ]
        }
    
    def debug_classification(self, cfg: DictConfig, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """调试特征分类过程"""
        debug_info = {
            'config_analysis': {},
            'batch_analysis': {},
            'classification_result': {},
            'validation_issues': []
        }
        
        # 分析配置文件
        debug_info['config_analysis'] = {
            'has_numeric_features': hasattr(cfg.data.tabular_features, 'numeric_features'),
            'has_categorical_features': hasattr(cfg.data.tabular_features, 'categorical_features'),
            'has_domain_features': hasattr(cfg.data, 'domain_name_embedding_features'),
            'has_sequence_features': hasattr(cfg.data, 'sequence_features'),
            'has_text_features': hasattr(cfg.data, 'text_features'),
            'sequence_enabled': getattr(cfg.data.sequence_features, 'enabled', False) if hasattr(cfg.data, 'sequence_features') else False,
            'text_enabled': getattr(cfg.data.text_features, 'enabled', False) if hasattr(cfg.data, 'text_features') else False,
            'domain_enabled': getattr(cfg.data.domain_name_embedding_features, 'enabled', False) if hasattr(cfg.data, 'domain_name_embedding_features') else False
        }
        
        # 分析批次数据
        debug_info['batch_analysis'] = {
            'batch_keys': list(batch_data.keys()),
            'tensor_shapes': {
                k: v.shape if hasattr(v, 'shape') else type(v).__name__ 
                for k, v in batch_data.items()
            },
            'has_numeric_data': 'numeric_features' in batch_data,
            'has_domain_data': 'domain_embedding_features' in batch_data,
            'has_sequence_data': all(key in batch_data for key in ['iat_times', 'payload_sizes', 'sequence_mask']),
            'has_text_data': any(key in batch_data for key in ['ssl_server_name', 'dns_query', 'cert0_subject'])
        }
        
        # 执行分类
        classification = self.classify_from_config(cfg, batch_data)
        debug_info['classification_result'] = {
            category: len(features) for category, features in classification.items()
        }
        
        # 验证问题
        batch_analysis = debug_info['batch_analysis']
        config_analysis = debug_info['config_analysis']
        
        if config_analysis['sequence_enabled'] and not batch_analysis['has_sequence_data']:
            debug_info['validation_issues'].append("序列特征已配置但批次数据缺失")
        
        if config_analysis['text_enabled'] and not batch_analysis['has_text_data']:
            debug_info['validation_issues'].append("文本特征已配置但批次数据缺失")
        
        if config_analysis['domain_enabled'] and not batch_analysis['has_domain_data']:
            debug_info['validation_issues'].append("域名嵌入特征已配置但批次数据缺失")
        
        return debug_info
    
    def add_custom_category(self, category_name: str, category_config: Dict[str, Any]):
        """添加自定义特征类别"""
        self.feature_hierarchy[category_name] = category_config
        logger.info(f"添加自定义特征类别: {category_name}")