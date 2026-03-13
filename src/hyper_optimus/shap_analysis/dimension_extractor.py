"""
配置维度提取器 - 基于配置文件动态提取特征维度信息，解决硬编码问题
"""

import logging
from typing import Dict, Any
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class ConfigDimensionExtractor:
    """从配置文件动态提取特征维度信息，解决硬编码问题"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.prob_list_length = 13  # 域名概率列表固定长度
        
    def count_numeric_features(self) -> int:
        """动态统计数值特征数量"""
        count = 0
        if hasattr(self.cfg.data.tabular_features, 'numeric_features'):
            num_cfg = self.cfg.data.tabular_features.numeric_features
            
            # flow_features
            if hasattr(num_cfg, 'flow_features'):
                count += len(num_cfg.flow_features)
            
            # x509_features  
            if hasattr(num_cfg, 'x509_features'):
                count += len(num_cfg.x509_features)
                
            # dns_features
            if hasattr(num_cfg, 'dns_features'):
                count += len(num_cfg.dns_features)
        
        logger.debug(f"数值特征数量统计: {count}")
        return count
    
    def get_bert_hidden_size(self) -> int:
        """从BERT配置获取隐藏层维度"""
        model_name = getattr(self.cfg.bert, 'model_name', '')
        if 'H-128' in model_name:
            return 128
        elif 'H-256' in model_name:
            return 256
        elif 'H-768' in model_name:
            return 768
        elif 'H-512' in model_name:
            return 512
        else:
            # 默认值或从bert_config获取
            return getattr(self.cfg.bert, 'hidden_size', 128)
    
    def get_categorical_feature_count(self) -> int:
        """获取类别特征数量"""
        if hasattr(self.cfg.data.tabular_features, 'categorical_features'):
            return len(self.cfg.data.tabular_features.categorical_features)
        return 0
    
    def get_domain_embedding_count(self) -> int:
        """获取域名嵌入特征数量"""
        if hasattr(self.cfg.data, 'domain_name_embedding_features'):
            if hasattr(self.cfg.data.domain_name_embedding_features, 'column_list'):
                return len(self.cfg.data.domain_name_embedding_features.column_list)
        return 0
    
    def get_sequence_embedding_dim(self) -> int:
        """获取序列嵌入维度"""
        return getattr(self.cfg.model.sequence, 'embedding_dim', 128)
    
    def get_text_embedding_dim(self) -> int:
        """获取文本嵌入维度"""
        return self.get_bert_hidden_size()
    
    def calculate_all_dimensions(self) -> Dict[str, int]:
        """计算所有特征维度"""
        dimensions = {}
        
        # 1. 数值特征维度
        dimensions['numeric_dims'] = self.count_numeric_features()
        
        # 2. 类别特征维度：类别数量 × embedding维度
        cat_count = self.get_categorical_feature_count()
        bert_hidden_size = self.get_bert_hidden_size()
        dimensions['categorical_dims'] = cat_count * bert_hidden_size
        
        # 3. 域名嵌入维度：域名数量 × 概率列表长度
        domain_count = self.get_domain_embedding_count()
        dimensions['domain_dims'] = domain_count * self.prob_list_length
        
        # 4. 序列嵌入维度
        dimensions['sequence_dims'] = self.get_sequence_embedding_dim()
        
        # 5. 文本嵌入维度
        dimensions['text_dims'] = self.get_text_embedding_dim()
        
        # 6. 总表格特征维度
        dimensions['total_tabular_dims'] = (
            dimensions['numeric_dims'] + 
            dimensions['categorical_dims'] + 
            dimensions['domain_dims']
        )
        
        # 7. 总输入维度（用于SHAP分析）
        dimensions['total_input_dims'] = (
            dimensions['total_tabular_dims'] +
            dimensions['sequence_dims'] +
            dimensions['text_dims']
        )
        
        logger.info(f"特征维度计算完成: {dimensions}")
        return dimensions
    
    def validate_dimensions(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证实际批次数据维度与配置维度的一致性"""
        dimensions = self.calculate_all_dimensions()
        validation_result = {
            'is_compatible': True,
            'expected': dimensions,
            'actual': {},
            'mismatches': []
        }
        
        # 验证数值特征维度
        if 'numeric_features' in batch_data:
            actual_numeric = batch_data['numeric_features'].shape[-1]
            validation_result['actual']['numeric_dims'] = actual_numeric
            if actual_numeric != dimensions['numeric_dims']:
                validation_result['mismatches'].append(
                    f"数值特征维度不匹配: 期望{dimensions['numeric_dims']}, 实际{actual_numeric}"
                )
                validation_result['is_compatible'] = False
        
        # 验证域名嵌入特征维度
        if 'domain_embedding_features' in batch_data:
            actual_domain = batch_data['domain_embedding_features'].shape[-1]
            validation_result['actual']['domain_dims'] = actual_domain
            if actual_domain != dimensions['domain_dims']:
                validation_result['mismatches'].append(
                    f"域名嵌入特征维度不匹配: 期望{dimensions['domain_dims']}, 实际{actual_domain}"
                )
                validation_result['is_compatible'] = False
        
        # 验证序列嵌入维度
        if self.cfg.data.sequence_features.enabled:
            expected_seq_dim = dimensions['sequence_dims']
            validation_result['expected']['sequence_dims'] = expected_seq_dim
        
        # 验证文本嵌入维度
        if self.cfg.data.text_features.enabled:
            expected_text_dim = dimensions['text_dims']
            validation_result['expected']['text_dims'] = expected_text_dim
        
        if validation_result['mismatches']:
            logger.warning(f"维度验证发现问题: {validation_result['mismatches']}")
        else:
            logger.info("维度验证通过")
            
        return validation_result
    
    def get_fix_suggestions(self, validation_result: Dict[str, Any]) -> list:
        """根据验证结果提供修复建议"""
        suggestions = []
        
        for mismatch in validation_result.get('mismatches', []):
            if '数值特征维度不匹配' in mismatch:
                suggestions.append("检查配置文件中的flow_features列表是否与数据文件列数一致")
            elif '域名嵌入特征维度不匹配' in mismatch:
                suggestions.append("检查domain_name_embedding_features.column_list配置与实际数据概率列表长度")
            else:
                suggestions.append(f"请检查相关配置: {mismatch}")
        
        return suggestions