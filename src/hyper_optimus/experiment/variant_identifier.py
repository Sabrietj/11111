"""
消融变体标识模块
提供通用的消融变体ID生成和解析功能
"""

import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VariantIdentifier:
    """消融变体通用标识器"""
    
    def __init__(self):
        self.type_prefixes = {
            'feature_ablation': 'FT',
            'fusion_ablation': 'FU', 
            'loss_ablation': 'LS',
            'architecture_ablation': 'AR',
            'hyperparameter_ablation': 'HP'
        }
        
        self.feature_codes = {
            'sequence_features': 'S',
            'domain_name_embedding_features': 'D', 
            'text_features': 'T'
        }
        
        self.fusion_codes = {
            'concat': 'CN',
            'cross_attention': 'CA',
            'weighted_sum': 'WS'
        }
        
        self.loss_codes = {
            'prediction': 'PR',
            'prediction+reconstruction': 'PR+RC',
            'prediction+reconstruction+mlm': 'PR+RC+MLM'
        }

    def generate_standard_id(self, variant: Dict[str, Any]) -> str:
        """
        生成标准ID (基于配置哈希)
        
        Args:
            variant: 消融变体配置
            
        Returns:
            标准ID字符串
        """
        variant_type = variant['type']
        prefix = self.type_prefixes.get(variant_type, 'XX')
        
        # 检查是否为基线
        if variant.get('baseline', False):
            return f"{prefix}_BASE"
        
        # 根据配置生成唯一标识
        config_hash = self._config_to_hash(variant.get('config', {}))
        return f"{prefix}_{config_hash[:6].upper()}"

    def generate_readable_id(self, variant: Dict[str, Any], counters: Dict[str, int]) -> str:
        """
        生成可读ID (基于序号)
        
        Args:
            variant: 消融变体配置
            counters: 各类型的计数器
            
        Returns:
            可读ID字符串
        """
        variant_type = variant['type']
        prefix = self.type_prefixes.get(variant_type, 'XX')
        
        if variant.get('baseline', False):
            return f"{prefix}_BASE"
        
        # 获取该类型的下一个序号
        if variant_type not in counters:
            counters[variant_type] = 0
        counters[variant_type] += 1
        
        return f"{prefix}{counters[variant_type]:02d}"

    def generate_semantic_id(self, variant: Dict[str, Any]) -> str:
        """
        生成语义化ID (基于配置内容)
        
        Args:
            variant: 消融变体配置
            
        Returns:
            语义化ID字符串
        """
        variant_type = variant['type']
        prefix = self.type_prefixes.get(variant_type, 'XX')
        
        if variant.get('baseline', False):
            return f"{prefix}_BASE"
        
        if variant_type == 'feature_ablation':
            config = variant.get('config', {})
            return self._generate_feature_semantic_id(config)
        elif variant_type == 'fusion_ablation':
            # 融合消融可以直接有method字段，也可以有config.method
            if 'method' in variant:
                return self._generate_fusion_semantic_id({'method': variant['method']})
            else:
                config = variant.get('config', {})
                return self._generate_fusion_semantic_id(config)
        elif variant_type == 'loss_ablation':
            config = variant.get('config', {})
            return self._generate_loss_semantic_id(config)
        else:
            # 其他类型使用哈希
            config_hash = self._config_to_hash(variant.get('config', {}))
            return f"{prefix}_{config_hash[:6].upper()}"

    def _generate_feature_semantic_id(self, config: Dict[str, Any]) -> str:
        """生成特征消融的语义ID"""
        # 检查是否为基线（所有特征都启用）
        enabled_features = config.get('enabled_features', {})
        all_enabled = all(enabled_features.get(feature, True) for feature in ['sequence_features', 'domain_name_embedding_features', 'text_features'])
        
        if all_enabled:
            return "FT_BASE"
        
        # 如果不是基线，检查是否有明确禁用的特征，优先使用单个特征名称
        disabled_count = sum(1 for feature in ['sequence_features', 'domain_name_embedding_features', 'text_features'] 
                           if not enabled_features.get(feature, True))
        
        # 如果只禁用一个特征，使用对应的no_*名称
        if disabled_count == 1:
            if not enabled_features.get('sequence_features', True):
                return "FT_no_sequence_features"
            elif not enabled_features.get('domain_name_embedding_features', True):
                return "FT_no_domain_features"
            elif not enabled_features.get('text_features', True):
                return "FT_no_text_features"
        
        # 如果禁用多个特征，生成组合名称
        disabled_features = []
        if not enabled_features.get('sequence_features', True):
            disabled_features.append('no_sequence_features')
        if not enabled_features.get('domain_name_embedding_features', True):
            disabled_features.append('no_domain_features')
        if not enabled_features.get('text_features', True):
            disabled_features.append('no_text_features')
        
        return f"FT_{'_'.join(disabled_features)}" if disabled_features else "FT_BASE"

    def _generate_fusion_semantic_id(self, config: Dict[str, Any]) -> str:
        """生成融合消融的语义ID"""
        method = config.get('method', 'concat')
        method_code = self.fusion_codes.get(method, 'UNK')
        return f"FU_{method_code}"

    def _generate_loss_semantic_id(self, config: Dict[str, Any]) -> str:
        """生成损失消融的语义ID"""
        loss_type = config.get('type', 'prediction')
        loss_code = self.loss_codes.get(loss_type, 'UNK')
        return f"LS_{loss_code}"

    def parse_variant_id(self, variant_id: str) -> Dict[str, Any]:
        """
        解析变体ID获取信息
        
        Args:
            variant_id: 变体ID字符串
            
        Returns:
            解析信息字典
        """
        if '_' in variant_id:
            parts = variant_id.split('_')
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else ''
        else:
            prefix = variant_id[:2] if len(variant_id) >= 2 else variant_id
            suffix = variant_id[2:] if len(variant_id) > 2 else ''
        
        type_mapping = {
            'FT': 'feature_ablation',
            'FU': 'fusion_ablation', 
            'LS': 'loss_ablation',
            'AR': 'architecture_ablation',
            'HP': 'hyperparameter_ablation'
        }
        
        return {
            'type': type_mapping.get(prefix, 'unknown'),
            'is_baseline': suffix == 'BASE',
            'sequence': suffix.isdigit() and len(suffix) == 2,
            'sequence_number': int(suffix) if suffix.isdigit() else None,
            'prefix': prefix,
            'suffix': suffix
        }

    def assign_variant_ids(self, ablation_variants: Dict[str, Any], 
                          id_type: str = 'semantic') -> Dict[str, Any]:
        """
        自动分配变体ID
        
        Args:
            ablation_variants: 消融变体字典
            id_type: ID类型 ('standard', 'readable', 'semantic')
            
        Returns:
            更新后的消融变体字典
        """
        counters = {}
        updated_variants = {}
        
        # 按类型和基线标记排序
        sorted_variants = self._sort_variants(ablation_variants)
        
        for current_id, variant in sorted_variants:
            variant_type = variant['type']
            
            # 生成新的ID
            if id_type == 'standard':
                new_id = self.generate_standard_id(variant)
            elif id_type == 'readable':
                new_id = self.generate_readable_id(variant, counters)
            elif id_type == 'semantic':
                new_id = self.generate_semantic_id(variant)
            else:
                logger.warning(f"Unknown id_type: {id_type}, using semantic")
                new_id = self.generate_semantic_id(variant)
            
            # 检查ID冲突
            if new_id in updated_variants:
                # 如果有冲突，添加后缀
                counter = 1
                while f"{new_id}_{counter}" in updated_variants:
                    counter += 1
                new_id = f"{new_id}_{counter}"
            
            # 更新变体配置
            variant_copy = variant.copy()
            variant_copy['assigned_id'] = new_id
            variant_copy['original_id'] = current_id
            variant_copy['id_generation'] = id_type
            
            updated_variants[new_id] = variant_copy
            logger.info(f"Assigned ID: {current_id} -> {new_id} ({variant_type})")
        
        return updated_variants

    def _sort_variants(self, variants: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        排序变体：基线优先，然后按类型和名称排序
        
        Args:
            variants: 变体字典
            
        Returns:
            排序后的变体列表
        """
        def sort_key(item):
            variant_id, variant = item
            
            # 基线优先
            if variant.get('baseline', False):
                type_order = 0
            else:
                type_order = 1
            
            # 按类型排序
            type_priority = {
                'feature_ablation': 1,
                'fusion_ablation': 2, 
                'loss_ablation': 3
            }
            variant_type = variant.get('type', 'unknown')
            type_rank = type_priority.get(variant_type, 99)
            
            # 按名称排序
            name = variant.get('name', '')
            
            return (type_order, type_rank, name)
        
        return sorted(variants.items(), key=sort_key)

    def _config_to_hash(self, config: Dict[str, Any]) -> str:
        """
        将配置转换为哈希
        
        Args:
            config: 配置字典
            
        Returns:
            MD5哈希字符串
        """
        # 规范化配置：排序键值
        normalized_config = self._normalize_config(config)
        config_str = str(normalized_config)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化配置：确保字典键的顺序一致
        
        Args:
            config: 原始配置
            
        Returns:
            规范化后的配置
        """
        def normalize_item(item):
            if isinstance(item, dict):
                return {k: normalize_item(item[k]) for k in sorted(item.keys())}
            elif isinstance(item, list):
                return [normalize_item(x) for x in item]
            else:
                return item
        
        return normalize_item(config)

    def validate_variant_ids(self, ablation_variants: Dict[str, Any]) -> List[str]:
        """
        验证变体ID的有效性
        
        Args:
            ablation_variants: 消融变体字典
            
        Returns:
            错误信息列表
        """
        errors = []
        used_ids = set()
        
        for variant_id, variant in ablation_variants.items():
            # 检查ID格式
            if not self._is_valid_id_format(variant_id):
                errors.append(f"Invalid ID format: {variant_id}")
            
            # 检查ID重复
            if variant_id in used_ids:
                errors.append(f"Duplicate ID: {variant_id}")
            used_ids.add(variant_id)
            
            # 检查必需字段
            required_fields = ['name', 'description', 'type']
            for field in required_fields:
                if field not in variant:
                    errors.append(f"Missing required field '{field}' in variant {variant_id}")
        
        return errors

    def _is_valid_id_format(self, variant_id: str) -> bool:
        """
        检查ID格式是否有效
        
        Args:
            variant_id: 变体ID
            
        Returns:
            是否有效
        """
        if not variant_id or not isinstance(variant_id, str):
            return False
        
        # 检查是否以有效前缀开头
        valid_prefixes = list(self.type_prefixes.values())
        has_valid_prefix = any(variant_id.startswith(prefix) for prefix in valid_prefixes)
        
        return has_valid_prefix

    def get_variant_summary(self, ablation_variants: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取变体摘要信息
        
        Args:
            ablation_variants: 消融变体字典
            
        Returns:
            摘要信息
        """
        summary = {
            'total_variants': len(ablation_variants),
            'by_type': {},
            'baseline_variants': [],
            'id_prefixes_used': set()
        }
        
        for variant_id, variant in ablation_variants.items():
            variant_type = variant.get('type', 'unknown')
            
            # 统计类型
            if variant_type not in summary['by_type']:
                summary['by_type'][variant_type] = 0
            summary['by_type'][variant_type] += 1
            
            # 记录基线
            if variant.get('baseline', False):
                summary['baseline_variants'].append(variant_id)
            
            # 记录ID前缀
            parsed = self.parse_variant_id(variant_id)
            summary['id_prefixes_used'].add(parsed['prefix'])
        
        summary['id_prefixes_used'] = list(summary['id_prefixes_used'])
        
        return summary