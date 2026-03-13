"""
增强版SHAP包装器 - 解决维度不匹配问题的增强版SHAP包装器
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Tuple
from .dimension_extractor import ConfigDimensionExtractor

logger = logging.getLogger(__name__)

class EnhancedShapFusionWrapper(nn.Module):
    """
    增强版SHAP包装器，解决维度不匹配问题
    功能：
    1. 动态重建完整表格输入，解决维度不匹配问题
    2. 支持五大特征类别的独立分析
    3. 确保计算图连通性
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cfg = model.cfg
        
        # 初始化维度提取器
        self.dimension_extractor = ConfigDimensionExtractor(model.cfg)
        self.feature_dims = self.dimension_extractor.calculate_all_dimensions()
        
        logger.info(f"EnhancedShapFusionWrapper初始化完成，特征维度: {self.feature_dims}")
    
    def forward(self, numeric_feats, domain_feats, cat_feats, seq_emb, text_emb):
        """
        增强版前向传播，支持完整的五大特征类别分析
        
        Args:
            numeric_feats: 数值特征 [batch_size, num_numeric_dims]
            domain_feats: 域名嵌入特征 [batch_size, num_domain_dims] 
            cat_feats: 类别特征嵌入 [batch_size, num_categorical_dims]
            seq_emb: 序列嵌入 [batch_size, seq_embedding_dim]
            text_emb: 文本嵌入 [batch_size, text_embedding_dim]
            
        Returns:
            logits: 分类器输出 [batch_size, num_classes]
        """
        
        # ==============================================================
        # 1. 表格特征路径 - 支持完整的五大特征类别
        # ==============================================================
        # 动态构建表格输入：数值 + 域名 + 类别特征
        tabular_components = [numeric_feats]
        
        # 添加域名嵌入特征（如果启用）
        if self.model.domain_embedding_enabled:
            tabular_components.append(domain_feats)
        
        # 添加类别特征嵌入（如果启用且有类别特征）
        categorical_columns_effective = getattr(self.model, 'categorical_columns_effective', [])
        if len(categorical_columns_effective) > 0:
            tabular_components.append(cat_feats)
        
        # 拼接所有表格特征
        if len(tabular_components) > 1:
            tabular_input = torch.cat(tabular_components, dim=1)
        else:
            tabular_input = numeric_feats
        
        tabular_out = self.model.tabular_projection(tabular_input)
        
        # ==============================================================
        # 3. 序列特征路径（如果启用）
        # ==============================================================
        if self.model.sequence_features_enabled:
            if seq_emb.shape[-1] != self.feature_dims['sequence_dims']:
                logger.warning(f"序列嵌入维度不匹配: 期望{self.feature_dims['sequence_dims']}, 实际{seq_emb.shape[-1]}")
                # 调整维度
                if seq_emb.shape[-1] > self.feature_dims['sequence_dims']:
                    seq_emb = seq_emb[:, :self.feature_dims['sequence_dims']]
                else:
                    # 填充到期望维度
                    pad_size = self.feature_dims['sequence_dims'] - seq_emb.shape[-1]
                    seq_emb = torch.cat([
                        seq_emb, 
                        torch.zeros(seq_emb.shape[0], pad_size, device=seq_emb.device)
                    ], dim=1)
            
            sequence_out = self.model.sequence_projection(seq_emb)
        else:
            sequence_out = torch.zeros_like(tabular_out)
        
        # ==============================================================
        # 4. 文本特征路径（如果启用）
        # ==============================================================
        if self.model.text_features_enabled:
            if text_emb.shape[-1] != self.feature_dims['text_dims']:
                logger.warning(f"文本嵌入维度不匹配: 期望{self.feature_dims['text_dims']}, 实际{text_emb.shape[-1]}")
                # 调整维度
                if text_emb.shape[-1] > self.feature_dims['text_dims']:
                    text_emb = text_emb[:, :self.feature_dims['text_dims']]
                else:
                    # 填充到期望维度
                    pad_size = self.feature_dims['text_dims'] - text_emb.shape[-1]
                    text_emb = torch.cat([
                        text_emb, 
                        torch.zeros(text_emb.shape[0], pad_size, device=text_emb.device)
                    ], dim=1)
            
            text_out = text_emb  # BERT输出已经是正确维度
        else:
            text_out = torch.zeros_like(tabular_out)
        
        # ==============================================================
        # 5. 多视图融合
        # ==============================================================
        multiview_out = self.model._fuse_multi_views(sequence_out, text_out, tabular_out)
        
        # ==============================================================
        # 6. 分类器
        # ==============================================================
        logits = self.model.classifier(multiview_out)
        
        return logits
    
    def get_input_dimensions(self) -> Dict[str, int]:
        """获取预期的输入维度信息"""
        return self.feature_dims.copy()
    
    def validate_input_compatibility(self, inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """验证输入兼容性"""
        input_names = ['numeric_feats', 'domain_feats', 'cat_feats', 'seq_emb', 'text_emb']
        expected_dims = [
            self.feature_dims['numeric_dims'],
            self.feature_dims['domain_dims'],
            self.feature_dims['categorical_dims'],
            self.feature_dims['sequence_dims'],
            self.feature_dims['text_dims']
        ]
        
        validation_result = {
            'is_compatible': True,
            'detailed_check': {},
            'issues': []
        }
        
        for i, (input_name, expected_dim, actual_input) in enumerate(zip(input_names, expected_dims, inputs)):
            actual_shape = list(actual_input.shape)
            
            validation_result['detailed_check'][input_name] = {
                'expected_dim': expected_dim,
                'actual_shape': actual_shape,
                'is_compatible': actual_shape[-1] == expected_dim or expected_dim == 0
            }
            
            if expected_dim > 0 and actual_shape[-1] != expected_dim:
                validation_result['issues'].append(
                    f"{input_name}维度不匹配: 期望{expected_dim}, 实际{actual_shape[-1]}"
                )
                validation_result['is_compatible'] = False
        
        return validation_result
    
    def get_feature_importance_mapping(self) -> Dict[str, Tuple[int, int]]:
        """获取特征重要性映射关系 (特征名 -> (起始索引, 结束索引))"""
        mapping = {}
        start_idx = 0
        
        # 数值特征
        if self.feature_dims['numeric_dims'] > 0:
            mapping['numeric_features'] = (start_idx, start_idx + self.feature_dims['numeric_dims'])
            start_idx += self.feature_dims['numeric_dims']
        
        # 域名嵌入特征
        if self.feature_dims['domain_dims'] > 0:
            mapping['domain_embedding_features'] = (start_idx, start_idx + self.feature_dims['domain_dims'])
            start_idx += self.feature_dims['domain_dims']
        
        # 类别特征
        if self.feature_dims['categorical_dims'] > 0:
            mapping['categorical_features'] = (start_idx, start_idx + self.feature_dims['categorical_dims'])
            start_idx += self.feature_dims['categorical_dims']
        
        # 序列特征（单独处理）
        if self.feature_dims['sequence_dims'] > 0:
            mapping['sequence_features'] = (0, self.feature_dims['sequence_dims'])
        
        # 文本特征（单独处理）
        if self.feature_dims['text_dims'] > 0:
            mapping['text_features'] = (0, self.feature_dims['text_dims'])
        
        return mapping