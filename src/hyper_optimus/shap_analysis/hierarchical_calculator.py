"""
分层SHAP计算器 - Level 1大类别，Level 2具体特征
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple
try:
    import shap
except ImportError:
    shap = None
from .enhanced_wrapper import EnhancedShapFusionWrapper
from .dimension_extractor import ConfigDimensionExtractor

logger = logging.getLogger(__name__)

class HierarchicalSHAPCalculator:
    """分层SHAP计算器：Level 1大类别，Level 2具体特征"""
    
    def __init__(self, model, feature_classification: Dict[str, List[str]]):
        self.model = model
        self.feature_classification = feature_classification
        self.cfg = model.cfg
        
        # 初始化增强包装器
        self.wrapper = EnhancedShapFusionWrapper(model)
        self.dimension_extractor = ConfigDimensionExtractor(self.cfg)
        self.feature_dims = self.dimension_extractor.calculate_all_dimensions()
        
        logger.info(f"HierarchicalSHAPCalculator初始化完成，特征维度: {self.feature_dims}")
    
    def calculate_level1_importance(self, explainer: Any, 
                                background_inputs: List[torch.Tensor], 
                                eval_inputs: List[torch.Tensor]) -> Dict[str, float]:
        """
        Level 1: 计算5大特征类别的SHAP重要性（饼图目标）
        
        Returns:
            Dict[str, float]: 5大特征类别的SHAP重要性百分比
        """
        try:
            # 计算SHAP值
            shap_values = explainer.shap_values(eval_inputs, check_additivity=False)
            
            # 按五大特征类别分组计算重要性
            category_importance = {
                'numeric_features': 0.0,
                'categorical_features': 0.0, 
                'sequence_features': 0.0,
                'text_features': 0.0,
                'domain_embedding_features': 0.0
            }
            
            # SHAP值格式: [numeric, domain, categorical, sequence, text]
            # 对应 enhanced wrapper forward 的5个输入顺序
            
            # 1. 数值特征重要性 (第一个输入)
            if len(shap_values) > 0 and shap_values[0] is not None:
                numeric_shap = shap_values[0]
                if torch.is_tensor(numeric_shap):
                    numeric_shap = numeric_shap.cpu().numpy()
                # 对所有数值特征维度求和，再对所有样本求平均
                category_importance['numeric_features'] = float(np.abs(numeric_shap).sum())
            
            # 2. 域名嵌入特征重要性 (第二个输入)
            if len(shap_values) > 1 and shap_values[1] is not None and self.model.domain_embedding_enabled:
                domain_shap = shap_values[1]
                if torch.is_tensor(domain_shap):
                    domain_shap = domain_shap.cpu().numpy()
                category_importance['domain_embedding_features'] = float(np.abs(domain_shap).sum())
            
            # 3. 类别特征重要性 (第三个输入)
            if len(shap_values) > 2 and shap_values[2] is not None:
                categorical_columns_effective = getattr(self.model, 'categorical_columns_effective', [])
                if len(categorical_columns_effective) > 0:
                    categorical_shap = shap_values[2]
                    if torch.is_tensor(categorical_shap):
                        categorical_shap = categorical_shap.cpu().numpy()
                    category_importance['categorical_features'] = float(np.abs(categorical_shap).sum())
                else:
                    # 即使没有启用类别特征，也计算SHAP值（因为传递了零张量）
                    categorical_shap = shap_values[2]
                    if torch.is_tensor(categorical_shap):
                        categorical_shap = categorical_shap.cpu().numpy()
                    category_importance['categorical_features'] = float(np.abs(categorical_shap).sum())
            
            # 4. 序列特征重要性 (第四个输入)
            if len(shap_values) > 3 and shap_values[3] is not None and self.model.sequence_features_enabled:
                sequence_shap = shap_values[3]
                if torch.is_tensor(sequence_shap):
                    sequence_shap = sequence_shap.cpu().numpy()
                category_importance['sequence_features'] = float(np.abs(sequence_shap).sum())
            
            # 5. 文本特征重要性 (第五个输入)
            if len(shap_values) > 4 and shap_values[4] is not None and self.model.text_features_enabled:
                text_shap = shap_values[4]
                if torch.is_tensor(text_shap):
                    text_shap = text_shap.cpu().numpy()
                category_importance['text_features'] = float(np.abs(text_shap).sum())
            
            # 归一化为百分比
            total_importance = sum(category_importance.values())
            if total_importance > 0:
                for category in category_importance:
                    category_importance[category] = (category_importance[category] / total_importance) * 100
            
            logger.info(f"Level 1 大类别重要性计算完成: {category_importance}")
            return category_importance
            
        except Exception as e:
            logger.error(f"Level 1 重要性计算失败: {e}", exc_info=True)
            return {}
    
    def calculate_level2_importance(self, explainer: Any,
                                background_inputs: List[torch.Tensor],
                                eval_inputs: List[torch.Tensor]) -> Dict[str, float]:
        """
        Level 2: 计算具体数值特征的详细重要性（柱状图目标）
        
        Returns:
            Dict[str, float]: 具体数值特征的SHAP重要性
        """
        try:
            # 计算SHAP值
            shap_values = explainer.shap_values(eval_inputs, check_additivity=False)
            
            # 获取数值特征的SHAP值（假设是第一个输入）
            if len(shap_values) == 0 or shap_values[0] is None:
                logger.warning("无法获取数值特征的SHAP值")
                return {}
            
            numeric_shap = shap_values[0]
            if torch.is_tensor(numeric_shap):
                numeric_shap = numeric_shap.cpu().numpy()
            
            # 计算每个特征的平均绝对SHAP值
            mean_abs_shap = np.abs(numeric_shap).mean(axis=0)
            
            # 从配置文件动态获取特征名称
            feature_names = self._extract_numeric_feature_names()
            
            # 构建重要性字典
            feature_importance = {}
            for i, (name, importance) in enumerate(zip(feature_names, mean_abs_shap)):
                if i < len(feature_names):
                    feature_importance[name] = float(importance)
            
            # 按重要性排序，取Top 20
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            )
            
            logger.info(f"Level 2 具体特征重要性计算完成，Top {len(sorted_importance)} 特征")
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Level 2 重要性计算失败: {e}", exc_info=True)
            return {}
    
    def _extract_numeric_feature_names(self) -> List[str]:
        """从配置文件中动态提取数值特征名称"""
        feature_names = []
        
        if hasattr(self.cfg.data.tabular_features, 'numeric_features'):
            num_cfg = self.cfg.data.tabular_features.numeric_features
            
            # flow_features
            if hasattr(num_cfg, 'flow_features'):
                feature_names.extend(num_cfg.flow_features)
            
            # x509_features  
            if hasattr(num_cfg, 'x509_features'):
                feature_names.extend(num_cfg.x509_features)
                
            # dns_features
            if hasattr(num_cfg, 'dns_features'):
                feature_names.extend(num_cfg.dns_features)
        
        # 如果特征名称数量与实际维度不匹配，使用通用名称
        expected_count = self.feature_dims['numeric_dims']
        if len(feature_names) != expected_count:
            logger.warning(f"特征名称数量({len(feature_names)})与维度({expected_count})不匹配")
            feature_names = [f"numeric_feature_{i}" for i in range(expected_count)]
        
        return feature_names
    
    def calculate_comprehensive_analysis(self, background_inputs: List[torch.Tensor],
                                     eval_inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """执行完整的分层SHAP分析"""
        # ======================================================================
        # 🔴 终极修复: 退出 Inference Mode + 开启 Grad
        # PyTorch Lightning 的 test 阶段默认处于 inference_mode (比 no_grad 更强)
        # 必须显式退出 inference_mode 才能构建计算图
        # ======================================================================
        with torch.inference_mode(False):  # 1. 退出推理模式
            with torch.enable_grad():      # 2. 开启梯度计算
                
                # 3. 再次确保模型参数允许求导 (双重保险)
                for param in self.wrapper.parameters():
                    param.requires_grad = True
                
                # 确保输入张量具有梯度
                def _ensure_gradients(inputs: List[torch.Tensor]) -> List[torch.Tensor]:
                    return [inp.detach().clone().requires_grad_(True) if isinstance(inp, torch.Tensor) else inp 
                           for inp in inputs]
                
                background_inputs = _ensure_gradients(background_inputs)
                eval_inputs = _ensure_gradients(eval_inputs)
                
                # 验证计算图连通性
                test_output = self.wrapper(*background_inputs)
                if test_output.grad_fn is None:
                    logger.error("❌ [Fatal] Wrapper 输出没有 grad_fn！计算图依然断裂。")
                    logger.error(f"当前梯度状态: is_grad_enabled={torch.is_grad_enabled()}, is_inference_mode={torch.is_inference_mode_enabled()}")
                    raise RuntimeError("无法构建计算图，请检查 PyTorch 版本或 Lightning 配置。")
                
                logger.info(f"✅ 计算图检查通过! Output grad_fn: {test_output.grad_fn}")
                
                # 初始化DeepExplainer
                if shap is None:
                    raise ImportError("SHAP library is not installed. Please install it with: pip install shap")
                explainer = shap.DeepExplainer(self.wrapper, background_inputs)
        
                # Level 1: 大类别分析（饼图）
                level1_results = self.calculate_level1_importance(explainer, background_inputs, eval_inputs)
        
                # Level 2: 具体特征分析（柱状图）
                level2_results = self.calculate_level2_importance(explainer, background_inputs, eval_inputs)
        
                # 构建完整结果
                comprehensive_results = {
                    'level1_category_importance': level1_results,
                    'level2_numeric_importance': level2_results,
                    'feature_classification': self.feature_classification,
                    'dimension_info': self.feature_dims,
                    'analysis_metadata': {
                        'total_background_samples': background_inputs[0].shape[0] if background_inputs[0] is not None else 0,
                        'total_eval_samples': eval_inputs[0].shape[0] if eval_inputs[0] is not None else 0,
                        'shap_method': 'DeepLIFT (check_additivity=False)',
                        'model_type': self._detect_model_type(),
                        'feature_hierarchy': self._get_feature_hierarchy()
                    }
                }
        
                return comprehensive_results
    
    def _detect_model_type(self) -> str:
        """检测当前模型类型"""
        from .multi_model_adapter import MultiModelSHAPAdapter
        adapter = MultiModelSHAPAdapter()
        return adapter.auto_detect_model_type(self.model)
    
    def _get_feature_hierarchy(self) -> Dict[str, Any]:
        """获取特征层次信息"""
        hierarchy = {}
        for category, features in self.feature_classification.items():
            hierarchy[category] = {
                'count': len(features),
                'features': features[:5] if len(features) > 5 else features,  # 只显示前5个
                'target_analysis': 'both' if category == 'numeric_features' else 'pie_chart'
            }
        return hierarchy
    
    def validate_computation_readiness(self, background_inputs: List[torch.Tensor],
                                   eval_inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """验证计算准备情况"""
        validation_result = {
            'is_ready': True,
            'issues': [],
            'warnings': []
        }
        
        # 检查输入数量
        expected_inputs = 5  # numeric, domain, categorical, sequence, text
        if len(background_inputs) != expected_inputs:
            validation_result['issues'].append(
                f"背景输入数量错误: 期望{expected_inputs}, 实际{len(background_inputs)}"
            )
            validation_result['is_ready'] = False
        
        if len(eval_inputs) != expected_inputs:
            validation_result['issues'].append(
                f"评估输入数量错误: 期望{expected_inputs}, 实际{len(eval_inputs)}"
            )
            validation_result['is_ready'] = False
        
        # 检查维度匹配
        for i, (bg_input, eval_input) in enumerate(zip(background_inputs, eval_inputs)):
            if bg_input is None or eval_input is None:
                continue
                
            if bg_input.shape[-1] != eval_input.shape[-1]:
                validation_result['issues'].append(
                    f"输入{i}维度不匹配: 背景{bg_input.shape[-1]}, 评估{eval_input.shape[-1]}"
                )
                validation_result['is_ready'] = False
        
        # 检查特征分类一致性
        enabled_features = []
        if self.model.text_features_enabled:
            enabled_features.append('text')
        if self.model.sequence_features_enabled:
            enabled_features.append('sequence')
        if self.model.domain_embedding_enabled:
            enabled_features.append('domain')
        enabled_features.append('numeric')  # 始终启用
        
        if len(enabled_features) < 2:
            validation_result['warnings'].append(
                f"启用的特征视图较少: {enabled_features}，分析结果可能不够全面"
            )
        
        return validation_result