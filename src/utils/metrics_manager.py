"""
指标管理器 - 严格按照metrics_design.md要求实现（适配DGL约束）
"""

import torch
from typing import Dict, Any, List, Optional, Union
import logging
import configparser
from pathlib import Path
from .metrics_base import BaseMetric, ClassificationMetrics, LossMetric, AttentionWeightMetric

logger = logging.getLogger(__name__)

class MetricsManager:
    """指标管理器（严格遵循DGL blocks数据结构和设计文档要求）"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.metrics = {}
        self.metric_history = {}
        self.phase_metrics = {'train': {}, 'val': {}, 'test': {}}
        self.dgl_constraints_enabled = True
        
        # 加载指标配置
        self.metrics_config = self._load_metrics_config(config_path)
        logger.info(f"指标配置加载完成: {len(self.metrics_config)} 个配置项")
    
    def _load_metrics_config(self, config_path: Optional[str] = None) -> Dict[str, bool]:
        """加载指标配置"""
        default_config = {
            'compute_accuracy': True,
            'compute_precision': True,
            'compute_recall': True,
            'compute_f1_score': True,
            'compute_auc_pr': True,
            'compute_macro_precision': True,
            'compute_macro_recall': True,
            'compute_macro_f1': True,
            'compute_weighted_precision': True,
            'compute_weighted_recall': True,
            'compute_weighted_f1': True,
            'compute_per_class_metrics': True,
            'compute_train_loss': True,
            'compute_val_loss': True,
            'compute_test_loss': True,
            'compute_attention_weights': True,
            'compute_view_correlation': True,
            'compute_view_quality': True,
            'compute_epoch_time': True,
            'compute_memory_usage': True
        }
        
        if not config_path:
            # 尝试自动查找配置文件
            config_paths = [
                'src_qyf/utils/config.cfg',
                '../src_qyf/utils/config.cfg',
                '../../src_qyf/utils/config.cfg'
            ]
            
            for path in config_paths:
                if Path(path).exists():
                    config_path = path
                    break
        
        if config_path and Path(config_path).exists():
            try:
                config = configparser.ConfigParser()
                config.read(config_path)
                
                if config.has_section('METRICS'):
                    metrics_config = {}
                    for key in default_config.keys():
                        if config.has_option('METRICS', key):
                            value = config.getboolean('METRICS', key)
                            metrics_config[key] = value
                        else:
                            metrics_config[key] = default_config[key]
                    
                    logger.info(f"从配置文件加载指标配置: {config_path}")
                    return metrics_config
                else:
                    logger.warning(f"配置文件中未找到METRICS段，使用默认配置")
                    
            except Exception as e:
                logger.warning(f"加载指标配置失败: {e}，使用默认配置")
        else:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        
        return default_config
    
    def add_metric(self, metric: BaseMetric, phases: List[str] = ['train', 'val', 'test']):
        """添加指标到指定阶段"""
        for phase in phases:
            if phase not in self.phase_metrics:
                self.phase_metrics[phase] = {}
            self.phase_metrics[phase][metric.name] = metric
        
        self.metrics[metric.name] = metric
        self.metric_history[metric.name] = []
        logger.info(f"添加指标: {metric.name} (阶段: {phases})")
    
    def remove_metric(self, metric_name: str):
        """移除指标"""
        if metric_name in self.metrics:
            del self.metrics[metric_name]
            del self.metric_history[metric_name]
            
            for phase in self.phase_metrics:
                if metric_name in self.phase_metrics[phase]:
                    del self.phase_metrics[phase][metric_name]
            
            logger.info(f"移除指标: {metric_name}")
    
    def update_metrics_dgl(self, blocks: List, model_output: torch.Tensor, 
                          phase: str = 'train', **kwargs):
        """
        更新指标（严格遵循DGL约束）
        
        Args:
            blocks: DGL采样块列表
            model_output: 模型输出张量
            phase: 阶段（train/val/test）
            **kwargs: 额外参数（如loss, attention_weights等）
        """
        if phase not in self.phase_metrics:
            logger.warning(f"未知的阶段: {phase}")
            return
        
        # 严格遵循DGL约束
        try:
            # 约束1: 从blocks[-1].dstdata获取标签
            labels = blocks[-1].dstdata['label']
            
            # 约束4: 应用掩码过滤
            mask = blocks[-1].dstdata[f'{phase}_mask']
            
            # 约束7: 确保在CPU上计算
            labels = labels.to('cpu')
            mask = mask.to('cpu')
            model_output = model_output.to('cpu')
            
            # 应用mask过滤
            masked_labels = labels[mask]
            masked_predictions = model_output[mask]
            
            # 约束5: 全局平均（在指标内部实现）
            # 更新该阶段的所有指标
            for metric in self.phase_metrics[phase].values():
                try:
                    if isinstance(metric, LossMetric):
                        # 损失指标需要额外的loss参数
                        if 'loss' in kwargs:
                            loss = kwargs['loss'].to('cpu') if isinstance(kwargs['loss'], torch.Tensor) else kwargs['loss']
                            metric.update(masked_predictions, masked_labels, loss=loss)
                    elif isinstance(metric, AttentionWeightMetric):
                        # 注意力权重指标需要attention_weights参数
                        if 'attention_weights' in kwargs:
                            attention_weights = kwargs['attention_weights']
                            if isinstance(attention_weights, torch.Tensor):
                                attention_weights = attention_weights.to('cpu')
                            metric.update(attention_weights)
                    else:
                        # 其他分类指标
                        metric.update(masked_predictions, masked_labels, **kwargs)
                except Exception as e:
                    logger.warning(f"更新指标 {metric.name} 时出错: {e}")
        
        except Exception as e:
            logger.warning(f"从DGL blocks获取数据时出错: {e}")
    
    def compute_metrics(self, phase: str = 'train') -> Dict[str, float]:
        """计算指定阶段的指标"""
        if phase not in self.phase_metrics:
            return {}
        
        all_metrics = {}
        
        for metric_name, metric in self.phase_metrics[phase].items():
            try:
                metric_values = metric.compute()
                for key, value in metric_values.items():
                    all_metrics[f"{metric_name}_{key}"] = value
            except Exception as e:
                logger.warning(f"计算指标 {metric_name} 时出错: {e}")
        
        return all_metrics
    
    def reset_metrics(self, phase: str = None):
        """重置指标（可指定阶段）"""
        if phase:
            if phase in self.phase_metrics:
                for metric in self.phase_metrics[phase].values():
                    metric.reset()
        else:
            for metric in self.metrics.values():
                metric.reset()
    
    def log_metrics(self, epoch: int, phase: str = 'train'):
        """记录指标到历史"""
        metrics = self.compute_metrics(phase)
        
        for metric_name in self.metrics.keys():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []
            
            # 记录当前epoch的指标
            epoch_metrics = {k: v for k, v in metrics.items() if k.startswith(metric_name)}
            if epoch_metrics:  # 只记录有值的指标
                self.metric_history[metric_name].append({
                    'epoch': epoch,
                    'phase': phase,
                    'metrics': epoch_metrics
                })
    
    def get_metric_trend(self, metric_key: str, phase: str = 'val') -> List[float]:
        """获取指标趋势数据"""
        trends = []
        metric_name = metric_key.split('_')[0]
        
        for history_entry in self.metric_history.get(metric_name, []):
            if history_entry['phase'] == phase and metric_key in history_entry['metrics']:
                trends.append(history_entry['metrics'][metric_key])
        
        return trends
    
    def get_best_metric(self, metric_key: str, phase: str = 'val', mode: str = 'max') -> Dict[str, Any]:
        """获取最佳指标值"""
        trends = self.get_metric_trend(metric_key, phase)
        if not trends:
            return {}
        
        best_value = max(trends) if mode == 'max' else min(trends)
        best_epoch = trends.index(best_value) + 1
        
        return {
            'value': best_value,
            'epoch': best_epoch,
            'phase': phase
        }
    
    def get_final_metrics(self) -> Dict[str, float]:
        """获取最终指标（用于测试阶段）"""
        return self.compute_metrics('test')
    
    def get_metrics_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取完整的指标历史"""
        return self.metric_history

def create_default_metrics_manager(num_classes: int, config_path: Optional[str] = None) -> MetricsManager:
    """创建默认的指标管理器（根据配置创建指标）"""
    manager = MetricsManager(config_path)
    
    # 根据配置添加分类指标
    if manager.metrics_config.get('compute_accuracy', True) or \
       manager.metrics_config.get('compute_precision', True) or \
       manager.metrics_config.get('compute_recall', True) or \
       manager.metrics_config.get('compute_f1_score', True) or \
       manager.metrics_config.get('compute_auc_pr', True):
        
        compute_per_class = manager.metrics_config.get('compute_per_class_metrics', True)
        classification_metric = ClassificationMetrics(num_classes=num_classes, compute_per_class=compute_per_class)
        manager.add_metric(classification_metric, phases=['train', 'val', 'test'])
    
    # 根据配置添加训练损失指标
    if manager.metrics_config.get('compute_train_loss', True):
        train_loss_metric = LossMetric(name='train_loss')
        manager.add_metric(train_loss_metric, phases=['train'])
    
    # 根据配置添加验证损失指标
    if manager.metrics_config.get('compute_val_loss', True):
        val_loss_metric = LossMetric(name='val_loss')
        manager.add_metric(val_loss_metric, phases=['val'])
    
    # 根据配置添加测试损失指标
    if manager.metrics_config.get('compute_test_loss', True):
        test_loss_metric = LossMetric(name='test_loss')
        manager.add_metric(test_loss_metric, phases=['test'])
    
    # 根据配置添加注意力权重指标
    if manager.metrics_config.get('compute_attention_weights', True):
        attention_metric = AttentionWeightMetric()
        manager.add_metric(attention_metric, phases=['test'])
    
    logger.info(f"根据配置创建指标管理器: {len(manager.metrics)} 个指标")
    return manager

# 全局指标管理器（使用时需要传入正确的类别数）
metrics_manager = None