"""
SHAP分析混入类
任何模型只需继承此类即可获得SHAP分析能力
"""

import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class SHAPAnalyzeMixin:
    """SHAP分析混入类 - 所有模型只需继承此类"""
    
    def __init__(self, *args, **kwargs):
        # 不调用super()，避免MRO冲突
        # 让子类显式调用父类的super()来处理MRO
        
        # 自动初始化SHAP分析器
        self.shap_analyzer = None
        self._model_type = None
        self._shap_enabled = False
        self._shap_frequency = 5
        self._shap_background_data = None
        self._shap_config = {}
        
        # 延迟初始化，等待模型配置加载完成
        self._shap_initialized = False
    
    def _initialize_shap_analyzer(self):
        """初始化SHAP分析器（延迟初始化）"""
        if self._shap_initialized:
            return
        
        try:
            # 获取模型类型
            self._model_type = self._get_model_type()
            
            # 获取SHAP配置
            shap_config = self._get_shap_config()
            if shap_config and shap_config.get('enable_shap', False):
                self._shap_enabled = True
                self._shap_frequency = shap_config.get('analysis_frequency', 5)
                self._shap_config = shap_config
                
                # 创建通用SHAP分析器
                from .universal_analyzer import UniversalSHAPAnalyzer
                
                # 构建完整的分析器配置，传递整个模型配置以便正确确定输出路径
                analyzer_config = {
                    'enabled_strategies': shap_config.get('enabled_strategies', 
                                                         ['numeric', 'sequence', 'text', 'categorical']),
                    'output_dir': shap_config.get('output_dir', None),  # 设为None让UniversalSHAPAnalyzer自动确定
                    'save_plots': shap_config.get('save_plots', True)
                }
                
                # 如果模型有完整的配置，将其传递给分析器以便正确确定输出路径
                if hasattr(self, 'cfg') and self.cfg:
                    # 传递完整的模型配置，包含logging等配置
                    analyzer_config.update({
                        'logging': self.cfg.get('logging', {}),
                        'hydra': self.cfg.get('hydra', {})
                    })
                
                self.shap_analyzer = UniversalSHAPAnalyzer(analyzer_config)
                
                logger.info(f"SHAP分析器已初始化，模型类型: {self._model_type}, 分析频率: {self._shap_frequency}")
            
            self._shap_initialized = True
            
        except Exception as e:
            logger.error(f"初始化SHAP分析器失败: {e}")
            self._shap_initialized = True  # 标记为已尝试初始化，避免重复尝试
    
    def _get_model_type(self) -> str:
        """自动获取模型类型"""
        class_name = self.__class__.__name__.lower()
        module_path = self.__class__.__module__.lower() if hasattr(self, '__class__') else ''
        
        # 基于模块路径和类名的精确识别 - 使用更精确的匹配
        if 'flow_bert_multiview_ssl_mlm' in module_path and '_mlm' in module_path:
            return 'flow_bert_ssl_mlm'
        elif 'flow_bert_multiview_ssl_seq2stat' in module_path and '_seq2stat' in module_path:
            return 'flow_bert_ssl_seq2stat'
        elif 'flow_bert_multiview_ssl' in module_path and '_mlm' not in module_path and '_seq2stat' not in module_path:
            return 'flow_bert_ssl'
        elif 'flow_bert' in class_name:
            return 'flow_bert'
        elif 'autoencoder' in class_name:
            return 'autoencoder'
        elif 'transformer' in class_name:
            return 'transformer'
        else:
            return 'generic'  # 通用类型
    
    def _get_shap_config(self) -> Dict[str, Any]:
        """获取SHAP配置"""
        # 方法1: 从模型配置中获取
        if hasattr(self, 'cfg') and self.cfg:
            if hasattr(self.cfg, 'shap'):
                return dict(self.cfg.shap)
            elif hasattr(self.cfg, 'shap_config'):
                return dict(self.cfg.shap_config)
        
        # 方法2: 从参数中获取
        if hasattr(self, 'shap_config') and self.shap_config:
            return dict(self.shap_config)
        
        # 方法3: 环境变量
        if os.getenv('SHAP_ENABLE', '').lower() == 'true':
            return {
                'enable_shap': True,
                'analysis_frequency': int(os.getenv('SHAP_FREQUENCY', '5')),
                'save_plots': os.getenv('SHAP_SAVE_PLOTS', 'true').lower() == 'true'
            }
        
        return {}
    
    def should_run_shap_analysis(self, current_epoch: int = None, batch_idx: int = None, **kwargs) -> bool:
        """判断是否应该运行SHAP分析"""
        # 延迟初始化
        if not self._shap_initialized:
            self._initialize_shap_analyzer()
        
        # 检查SHAP是否启用
        if not self._shap_enabled or not self.shap_analyzer:
            return False
        
        # 使用提供的参数或当前状态
        current_epoch = current_epoch or (self.current_epoch if hasattr(self, 'current_epoch') else 0)
        batch_idx = batch_idx or 0
        
        # 检查分析频率和批次条件
        frequency_match = current_epoch % self._shap_frequency == 0
        first_batch = batch_idx == 0
        
        return frequency_match and first_batch
    
    def perform_shap_analysis(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """统一的SHAP分析接口"""
        try:
            # 延迟初始化
            if not self._shap_initialized:
                self._initialize_shap_analyzer()
            
            if not self.shap_analyzer:
                logger.warning("SHAP分析器未初始化")
                return {}
            
            logger.info(f"开始执行SHAP分析，模型类型: {self._model_type}")
            
            # 执行分析
            results = self.shap_analyzer.analyze(self, batch, self._model_type)
            
            # 保存结果
            if self._shap_config.get('save_plots', True):
                filename_prefix = f"{self._model_type}_shap"
                if hasattr(self, 'current_epoch'):
                    filename_prefix += f"_epoch_{self.current_epoch}"
                
                saved_file = self.shap_analyzer.save_results(results, filename_prefix)
                if saved_file:
                    logger.info(f"SHAP分析结果已保存: {saved_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"SHAP分析执行失败: {e}")
            return {'error': str(e), 'analysis_type': 'shap_failed'}
    
    def collect_shap_background_data(self, max_samples: int = 10):
        """收集SHAP背景数据（可选功能）"""
        try:
            if not self._shap_enabled:
                return
            
            # 这个方法可以在训练开始时调用，收集一些背景数据
            # 当前实现中，通用分析器不需要专门的背景数据
            # 但保留这个接口以备将来扩展
            
            logger.info(f"SHAP背景数据收集完成，样本数: {max_samples}")
            
        except Exception as e:
            logger.error(f"收集SHAP背景数据失败: {e}")
    
    def get_shap_summary(self) -> Dict[str, Any]:
        """获取SHAP分析摘要"""
        try:
            if not self._shap_enabled:
                return {'shap_enabled': False}
            
            return {
                'shap_enabled': self._shap_enabled,
                'model_type': self._model_type,
                'analysis_frequency': self._shap_frequency,
                'analyzer_initialized': self.shap_analyzer is not None,
                'supported_feature_types': self.shap_analyzer.get_supported_feature_types() if self.shap_analyzer else [],
                'config': self._shap_config
            }
            
        except Exception as e:
            logger.error(f"获取SHAP摘要失败: {e}")
            return {'error': str(e)}
    
    # 钩子方法 - 可以在子类中重写
    
    def on_validation_epoch_end(self):
        """验证epoch结束时执行SHAP分析"""
        # 调用父类的验证逻辑（如果存在）
        try:
            super().on_validation_epoch_end()
        except AttributeError:
            pass
        
        # 执行SHAP分析
        if self.should_perform_shap_analysis():
            self.analyze_shap_if_needed()
    
    def should_perform_shap_analysis(self) -> bool:
        """判断是否应该执行SHAP分析"""
        # 延迟初始化
        if not self._shap_initialized:
            self._initialize_shap_analyzer()
        
        # 检查SHAP是否启用
        if not self._shap_enabled or not self.shap_analyzer:
            return False
        
        # 检查训练epoch数 - 允许第0个epoch也能执行（用于测试）
        current_epoch = getattr(self, 'current_epoch', 0)
        if current_epoch % self._shap_frequency == 0:
            return True
        
        return False
    
    def on_shap_analysis_completed(self, shap_results: Dict[str, Any]):
        """SHAP分析完成后的钩子方法"""
        pass
    
    def should_include_feature_in_shap(self, feature_name: str, feature_value: Any) -> bool:
        """判断特定特征是否应该包含在SHAP分析中"""
        # 默认包含所有特征，子类可以重写此方法进行过滤
        return True