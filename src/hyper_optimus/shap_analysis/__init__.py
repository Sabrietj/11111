"""
通用SHAP分析框架
提供可复用的特征重要性分析功能，适用于所有模型
"""

# 设置matplotlib日志级别，避免调试信息刷屏
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)

# 核心组件导入
from .dimension_extractor import ConfigDimensionExtractor
from .five_tier_classifier import FiveTierFeatureClassifier
from .multi_model_adapter import MultiModelSHAPAdapter
from .enhanced_wrapper import EnhancedShapFusionWrapper
from .hierarchical_calculator import HierarchicalSHAPCalculator
from .five_tier_analyzer import FiveTierSHAPAnalyzer

# 兼容性导入
from .shap_component import ShapAnalyzer
from .enhanced_shap_component import EnhancedShapAnalyzer

# 策略组件
# from .feature_classifier import FeatureClassifier
# from .analysis_strategies import AnalysisStrategyRegistry, NumericAnalysisStrategy, SequenceAnalysisStrategy, TextAnalysisStrategy
# from .shap_mixin import SHAPAnalyzeMixin
# from .universal_analyzer import UniversalSHAPAnalyzer

__all__ = [
    # 核心组件
    'ConfigDimensionExtractor',
    'FiveTierFeatureClassifier', 
    'MultiModelSHAPAdapter',
    'EnhancedShapFusionWrapper',
    'HierarchicalSHAPCalculator',
    'FiveTierSHAPAnalyzer',
    
    # 兼容性组件
    'ShapAnalyzer',
    'EnhancedShapAnalyzer',
    
    # 策略组件（暂时注释）
    # 'FeatureClassifier',
    # 'AnalysisStrategyRegistry', 
    # 'NumericAnalysisStrategy',
    # 'SequenceAnalysisStrategy',
    # 'TextAnalysisStrategy',
    # 'SHAPAnalyzeMixin',
    # 'UniversalSHAPAnalyzer'
]