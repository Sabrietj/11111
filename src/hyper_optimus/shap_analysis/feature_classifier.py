"""
通用特征分类器
自动识别和分类不同类型的特征
"""

import re
import numpy as np
import torch
from typing import Dict, List, Any, Tuple


class FeatureClassifier:
    """通用特征分类器 - 适用于所有模型"""
    
    def __init__(self):
        # 预定义的特征分类规则
        self.classification_rules = {
            'numeric': self._is_numeric_feature,
            'sequence': self._is_sequence_feature,
            'text': self._is_text_feature,
            'categorical': self._is_categorical_feature,
            'excluded': self._should_exclude
        }
        
        # 模型特定的特征映射（简化版本，无重复）
        self.model_specific_mappings = {
            'flow_bert_multiview': {
                'numeric_patterns': ['flowmeter\\.', 'numeric_features', 'domain_embedding_features'],
                'sequence_patterns': ['iat_times', 'payload_sizes', 'packet_directions', 'sequence_'],
                'text_patterns': ['ssl_server_name', 'dns_query', 'cert0_subject', 'cert0_issuer'],
                'exclude_patterns': ['sequence_mask', 'is_malicious_labels', 'split', 'flow_uid', 'labels', 'target', 'enabled', 'combined_text'],
                'decompose_combined_text': True
            },
            'flow_bert_ssl': {
                'numeric_patterns': ['flowmeter\\.', 'numeric_features', 'domain_embedding_features', 'reconstruction_', 'ssl_features'],
                'sequence_patterns': ['iat_times', 'payload_sizes', 'packet_directions', 'sequence_', 'ssl_sequence'],
                'text_patterns': ['ssl_server_name', 'dns_query', 'cert0_subject', 'cert0_issuer'],
                'exclude_patterns': ['sequence_mask', 'is_malicious_labels', 'split', 'flow_uid', 'labels', 'target', 'enabled', 'ssl_mask', 'idx', 'combined_text'],
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
                'numeric_patterns': ['flowmeter\\.', 'numeric_features', 'domain_embedding_features', 'reconstruction_', 'seq2stat_features', 'statistical_features'],
                'sequence_patterns': ['iat_times', 'payload_sizes', 'packet_directions', 'sequence_', 'seq2stat_sequence', 'statistical_sequence'],
                'text_patterns': ['ssl_server_name', 'dns_query', 'cert0_subject', 'cert0_issuer'],
                'exclude_patterns': ['sequence_mask', 'is_malicious_labels', 'split', 'flow_uid', 'labels', 'target', 'enabled', 'seq2stat_mask', 'idx', 'combined_text'],
                'decompose_combined_text': True
            },
            'autoencoder': {
                'numeric_patterns': ['feature_', 'value_', 'latent_', 'reconstruction_'],
                'exclude_patterns': ['mask', 'index', 'labels', 'target', 'enabled', 'idx', 'combined_text'],
                'decompose_combined_text': False
            }
        }
    
    def classify(self, batch: Dict[str, Any], model_type: str = None) -> Dict[str, List[str]]:
        """自动分类特征，智能模型类型检测"""
        # 智能检测模型类型（如果未提供）
        if model_type is None:
            model_type = self._detect_model_type(batch)
        
        feature_mapping = self.model_specific_mappings.get(model_type, {})
        classification = {rule: [] for rule in self.classification_rules.keys()}
        
        for key, value in batch.items():
            feature_type = self._determine_feature_type(key, value, feature_mapping)
            classification[feature_type].append(key)
        
        return classification
    
    def _determine_feature_type(self, key: str, value: Any, model_mapping: Dict[str, List[str]]) -> str:
        """智能判断特征类型"""
        # 1. 检查排除规则
        if self._should_exclude(key, model_mapping):
            return 'excluded'
        
        # 2. 依次检查各类型规则
        if self._is_text_feature(key, value, model_mapping):
            return 'text'
        elif self._is_sequence_feature(key, value, model_mapping):
            return 'sequence'
        elif self._is_categorical_feature(key, value, model_mapping):
            return 'categorical'
        elif self._is_numeric_feature(key, value, model_mapping):
            return 'numeric'
        
        # 3. 默认分类
        return 'numeric'  # 默认当作数值特征
    
    def _is_numeric_feature(self, key: str, value: Any, model_mapping: Dict[str, List[str]]) -> bool:
        """判断是否为数值特征"""
        if not torch.is_tensor(value):
            return False
        
        if value.dtype == torch.long:
            return False  # 排除长整型（通常为标签）
        
        if len(value.shape) < 2:
            return False  # 至少要是2维张量
        
        # 检查模式匹配
        patterns = model_mapping.get('numeric_patterns', [])
        for pattern in patterns:
            if re.search(pattern, key):
                return True
        
        # 默认数值特征模式
        if any(keyword in key for keyword in ['feature', 'value', 'embedding', 'numeric', 'data']):
            return True
        
        return False
    
    def _is_sequence_feature(self, key: str, value: Any, model_mapping: Dict[str, List[str]]) -> bool:
        """判断是否为序列特征"""
        if not torch.is_tensor(value):
            return False
        
        # 检查模式匹配
        patterns = model_mapping.get('sequence_patterns', [])
        for pattern in patterns:
            if re.search(pattern, key):
                return True
        
        # 默认序列特征模式
        if any(keyword in key for keyword in ['sequence', 'temporal', 'time_', 'iat_', 'payload']):
            return True
        
        # 检查张量维度（3维通常是序列数据：batch × seq_len × feature_dim）
        if len(value.shape) == 3:
            return True
        
        return False
    
    def _is_text_feature(self, key: str, value: Any, model_mapping: Dict[str, List[str]]) -> bool:
        """判断是否为文本特征"""
        # 检查是否为字符串列表/元组
        if isinstance(value, (list, tuple)):
            if all(isinstance(x, str) for x in value):
                return True
        
        # 检查模式匹配
        patterns = model_mapping.get('text_patterns', [])
        for pattern in patterns:
            if re.search(pattern, key):
                return True
        
        # 默认文本特征模式
        if any(keyword in key for keyword in ['text', 'string', 'name', 'query', 'description', 'cert']):
            return True
        
        return False
    
    def _is_categorical_feature(self, key: str, value: Any, model_mapping: Dict[str, List[str]]) -> bool:
        """判断是否为分类特征"""
        if not torch.is_tensor(value):
            return False
        
        if value.dtype == torch.long and len(value.shape) <= 2:
            return True
        
        # 检查是否为类别编码
        if any(keyword in key for keyword in ['category', 'class', 'label', 'id']) and value.dtype == torch.long:
            return True
        
        return False
    
    def _should_exclude(self, key: str, model_mapping: Dict[str, List[str]]) -> bool:
        """判断是否应该排除"""
        # 检查排除模式
        patterns = model_mapping.get('exclude_patterns', [])
        for pattern in patterns:
            if re.search(pattern, key):
                return True
        
        # 默认排除项
        exclude_keywords = ['mask', 'labels', 'target', 'split', 'uid', 'index', 'enabled', 'idx', 'combined_text']
        if any(keyword in key for keyword in exclude_keywords):
            return True
        
        return False
    
    def add_model_mapping(self, model_type: str, mapping: Dict[str, List[str]]):
        """添加新的模型映射"""
        self.model_specific_mappings[model_type] = mapping
    
    def _detect_model_type(self, batch: Dict[str, Any]) -> str:
        """智能检测模型类型"""
        keys = set(batch.keys())
        
        # 检查SSL相关特征
        if any(key in keys for key in ['sequence_mlm_mask']):
            return 'flow_bert_ssl_mlm'
        elif any(key in keys for key in ['seq2stat_targets']):
            return 'flow_bert_ssl_seq2stat'
        elif any(key in keys for key in ['ssl_features', 'ssl_mask']):
            return 'flow_bert_ssl'
        elif 'combined_text' in keys and any(key in keys for key in ['iat_times', 'payload_sizes']):
            return 'flow_bert_multiview'
        
        # 检查其他模型类型
        if any('attention_mask' in key or 'token_type_ids' in key for key in keys):
            return 'transformer'
        elif any('latent_' in key or 'reconstruction_' in key for key in keys):
            return 'autoencoder'
        
        # 默认返回通用模型类型
        return 'flow_bert_multiview'
    
    def _decompose_combined_text(self, combined_text: str) -> Dict[str, str]:
        """将combined_text分解为独立的文本特征"""
        decomposed = {}
        
        if not isinstance(combined_text, str) or not combined_text.strip():
            return decomposed
        
        # 根据常见模式分解文本
        text_parts = combined_text.split()
        
        # SSL证书相关模式
        ssl_server_pattern = None
        dns_query_pattern = None
        cert_subject_pattern = None
        cert_issuer_pattern = None
        
        for part in text_parts:
            if part and not part.isspace():
                # 检测SSL服务器名称模式（通常包含域名）
                if '.' in part and len(part) > 3 and not part.startswith('CN=') and not part.startswith('C='):
                    if ssl_server_pattern is None:
                        ssl_server_pattern = part
                    continue
                
                # 检测DNS查询模式
                if part.startswith('www.') or (len(part) > 3 and '.' in part and not part.startswith('CN=')):
                    if dns_query_pattern is None:
                        dns_query_pattern = part
                    continue
                
                # 检测证书主题模式
                if part.startswith('CN=') or part.startswith('O=') or part.startswith('OU='):
                    if cert_subject_pattern is None:
                        cert_subject_pattern = part
                    continue
                
                # 检测证书颁发者模式
                if 'CA' in part or 'Authority' in part or part.startswith('C='):
                    if cert_issuer_pattern is None:
                        cert_issuer_pattern = part
                    continue
        
        # 如果没有找到特定模式，根据位置分配
        if not ssl_server_pattern and not dns_query_pattern and not cert_subject_pattern and not cert_issuer_pattern:
            # 简单启发式：第一个像域名的作为ssl_server_name
            for i, part in enumerate(text_parts):
                if '.' in part and len(part) > 3:
                    ssl_server_pattern = part
                    # 第二个可能的作为dns_query
                    if i + 1 < len(text_parts) and '.' in text_parts[i + 1]:
                        dns_query_pattern = text_parts[i + 1]
                    break
        
        # 存储分解结果
        if ssl_server_pattern:
            decomposed['ssl_server_name'] = ssl_server_pattern
        if dns_query_pattern:
            decomposed['dns_query'] = dns_query_pattern
        if cert_subject_pattern:
            decomposed['cert0_subject'] = cert_subject_pattern
        if cert_issuer_pattern:
            decomposed['cert0_issuer'] = cert_issuer_pattern
        
        # 如果还是没有分配到任何特征，将整个文本作为ssl_server_name
        if not decomposed and combined_text.strip():
            decomposed['ssl_server_name'] = combined_text.strip()
        
        return decomposed
    
    def get_feature_summary(self, classification: Dict[str, List[str]]) -> str:
        """生成特征分类摘要"""
        summary = []
        for feature_type, features in classification.items():
            if features:
                summary.append(f"{feature_type}: {len(features)} 个特征")
                for feature in features[:3]:  # 只显示前3个
                    summary.append(f"  - {feature}")
                if len(features) > 3:
                    summary.append(f"  - ... 还有 {len(features) - 3} 个特征")
        
        return '\n'.join(summary)