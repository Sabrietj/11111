"""
评分标准化器
处理不同特征类型的评分归一化和标准化
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class ScoringNormalizer:
    """评分标准化器，用于平衡不同特征类型的评分尺度"""
    
    def __init__(self):
        # 重新平衡的权重配置
        self.feature_type_weights = {
            'text': {
                'length_importance': 0.2,  # 降低长度权重
                'richness_importance': 0.4,
                'diversity_importance': 0.3,
                'non_empty_ratio': 0.1
            },
            'numeric': {
                'variance_importance': 0.3,
                'range_importance': 0.25,
                'entropy_importance': 0.25,
                'mean_magnitude': 0.2
            },
            'sequence': {
                'temporal_importance': 0.25,
                'complexity_importance': 0.25,
                'entropy_importance': 0.25,
                'pattern_diversity': 0.25
            },
            'categorical': {
                'entropy_importance': 0.4,
                'balance_importance': 0.3,
                'diversity_importance': 0.3
            }
        }
    
    def normalize_within_feature_type(self, detailed_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """在特征类型内部进行归一化"""
        try:
            normalized_analysis = {}
            
            for feature_type, features in detailed_analysis.items():
                if 'error' in features or not isinstance(features, dict):
                    normalized_analysis[feature_type] = features
                    continue
                
                normalized_analysis[feature_type] = {}
                
                # 收集该类型下所有特征的指标值
                all_metrics_values = {metric: [] for metric in self._get_metrics_for_type(feature_type)}
                
                for feature_name, analysis in features.items():
                    if 'error' in analysis or 'metrics' not in analysis:
                        continue
                    
                    metrics = analysis['metrics']
                    for metric_name in all_metrics_values.keys():
                        value = metrics.get(metric_name, 0.0)
                        if not pd.isna(value) and value == value:  # 检查NaN
                            all_metrics_values[metric_name].append(float(value))
                
                # 计算每个指标的统计量
                metric_stats = {}
                for metric_name, values in all_metrics_values.items():
                    if values:
                        metric_stats[metric_name] = {
                            'min': min(values),
                            'max': max(values),
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
                    else:
                        metric_stats[metric_name] = {'min': 0.0, 'max': 1.0, 'mean': 0.0, 'std': 1.0}
                
                # 对每个特征进行归一化
                for feature_name, analysis in features.items():
                    if 'error' in analysis or 'metrics' not in analysis:
                        normalized_analysis[feature_type][feature_name] = analysis
                        continue
                    
                    normalized_metrics = {}
                    original_metrics = analysis['metrics'].copy()
                    
                    # 归一化各个指标
                    for metric_name, stats in metric_stats.items():
                        original_value = original_metrics.get(metric_name, 0.0)
                        
                        if pd.isna(original_value) or original_value != original_value:
                            normalized_metrics[metric_name] = 0.0
                        else:
                            # Min-Max归一化
                            if stats['max'] - stats['min'] > 1e-8:
                                normalized_value = (original_value - stats['min']) / (stats['max'] - stats['min'])
                            else:
                                normalized_value = 0.5  # 如果所有值相同，设为中值
                            
                            normalized_metrics[metric_name] = float(np.clip(normalized_value, 0.0, 1.0))
                    
                    # 使用新的权重计算重新平衡的综合评分
                    weights = self.feature_type_weights.get(feature_type, {})
                    balanced_score = sum(
                        normalized_metrics.get(metric, 0.0) * weight 
                        for metric, weight in weights.items()
                    )
                    
                    # 缩放到合理的范围 (0-100)
                    normalized_metrics['composite_score'] = float(balanced_score * 100)
                    
                    # 保留原始指标用于分析
                    normalized_metrics['original_metrics'] = original_metrics
                    
                    normalized_analysis[feature_type][feature_name] = {
                        'metrics': normalized_metrics,
                        'analysis_type': analysis.get('analysis_type', 'unknown'),
                        'data_shape': analysis.get('data_shape'),
                        'text_type': analysis.get('text_type'),
                        'sequence_type': analysis.get('sequence_type'),
                        'sample_count': analysis.get('sample_count'),
                        'category_count': analysis.get('category_count')
                    }
            
            return normalized_analysis
            
        except Exception as e:
            logger.error(f"特征类型内归一化失败: {e}")
            return detailed_analysis
    
    def standardize_across_feature_types(self, normalized_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """跨特征类型标准化到统一范围"""
        try:
            # 收集所有有效分数
            all_scores = []
            score_sources = []  # 记录每个分数的来源用于后续映射
            
            for feature_type, features in normalized_analysis.items():
                if not isinstance(features, dict) or 'error' in features:
                    continue
                
                for feature_name, analysis in features.items():
                    if 'metrics' in analysis:
                        score = analysis['metrics'].get('composite_score', 0.0)
                        if not pd.isna(score) and score == score and score > 0:
                            all_scores.append(score)
                            score_sources.append((feature_type, feature_name))
            
            if not all_scores:
                return normalized_analysis
            
            # 计算统计量用于Z-score标准化
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            
            # 应用Z-score标准化并缩放到0-100范围
            for idx, (feature_type, feature_name) in enumerate(score_sources):
                original_score = all_scores[idx]
                
                # Z-score标准化
                z_score = (original_score - mean_score) / (std_score + 1e-8)
                
                # 缩放到0-100范围 (使用3-sigma规则覆盖99.7%的数据)
                standardized_score = np.clip((z_score + 3) * 100 / 6, 0, 100)
                
                # 更新标准化后的分数
                normalized_analysis[feature_type][feature_name]['metrics']['standardized_score'] = float(standardized_score)
                normalized_analysis[feature_type][feature_name]['metrics']['z_score'] = float(z_score)
            
            logger.info(f"跨特征类型标准化完成，处理了 {len(all_scores)} 个特征分数")
            return normalized_analysis
            
        except Exception as e:
            logger.error(f"跨特征类型标准化失败: {e}")
            return normalized_analysis
    
    def rebalance_composite_scores(self, detailed_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """重新平衡综合评分的主入口"""
        try:
            logger.info("开始重新平衡综合评分...")
            
            # 阶段1: 特征类型内归一化
            normalized_analysis = self.normalize_within_feature_type(detailed_analysis)
            logger.info("特征类型内归一化完成")
            
            # 阶段2: 跨特征类型标准化
            standardized_analysis = self.standardize_across_feature_types(normalized_analysis)
            logger.info("跨特征类型标准化完成")
            
            return standardized_analysis
            
        except Exception as e:
            logger.error(f"重新平衡评分失败: {e}")
            return detailed_analysis
    
    def _get_metrics_for_type(self, feature_type: str) -> List[str]:
        """获取特征类型相关的指标列表"""
        if feature_type == 'text':
            return ['length_importance', 'richness_importance', 'diversity_importance', 'non_empty_ratio']
        elif feature_type == 'numeric':
            return ['variance_importance', 'range_importance', 'entropy_importance', 'mean_magnitude']
        elif feature_type == 'sequence':
            return ['temporal_importance', 'complexity_importance', 'entropy_importance', 'pattern_diversity']
        elif feature_type == 'categorical':
            return ['entropy_importance', 'balance_importance', 'diversity_importance']
        else:
            return []
    
    def get_scoring_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """获取评分摘要统计"""
        try:
            summary = {
                'feature_types': {},
                'score_ranges': {},
                'normalization_applied': True
            }
            
            for feature_type, features in analysis_results.items():
                if not isinstance(features, dict) or 'error' in features:
                    continue
                
                scores = []
                original_scores = []
                
                for feature_name, analysis in features.items():
                    if 'metrics' in analysis:
                        metrics = analysis['metrics']
                        if 'standardized_score' in metrics:
                            scores.append(metrics['standardized_score'])
                        if 'composite_score' in metrics:
                            original_scores.append(metrics['composite_score'])
                
                if scores:
                    summary['feature_types'][feature_type] = {
                        'count': len(scores),
                        'standardized_mean': np.mean(scores),
                        'standardized_std': np.std(scores),
                        'standardized_range': [min(scores), max(scores)]
                    }
                    
                    if original_scores:
                        summary['feature_types'][feature_type].update({
                            'original_mean': np.mean(original_scores),
                            'original_std': np.std(original_scores),
                            'original_range': [min(original_scores), max(original_scores)]
                        })
            
            return summary
            
        except Exception as e:
            logger.error(f"生成评分摘要失败: {e}")
            return {'error': str(e)}