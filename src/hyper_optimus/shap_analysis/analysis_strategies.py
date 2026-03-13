"""
分析策略实现
提供不同类型特征的专门分析策略
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import math
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class BaseAnalysisStrategy(ABC):
    """分析策略基类"""
    
    @abstractmethod
    def analyze(self, model: Any, features: List[str], batch: Dict[str, Any]) -> Dict[str, Any]:
        """执行分析"""
        pass
    
    @abstractmethod
    def get_importance_metrics(self) -> List[str]:
        """返回重要性指标列表"""
        pass


class NumericAnalysisStrategy(BaseAnalysisStrategy):
    """数值特征分析策略"""
    
    def get_importance_metrics(self) -> List[str]:
        return ['variance_importance', 'range_importance', 'entropy_importance', 'composite_score']
    
    def analyze(self, model: Any, features: List[str], batch: Dict[str, Any]) -> Dict[str, Any]:
        """分析数值特征"""
        importance_scores = {}
        
        for feature_name in features:
            if feature_name in batch:
                feature_data = batch[feature_name]
                
                try:
                    # 通用数值特征分析
                    scores = {
                        'variance_importance': self._calculate_variance_importance(feature_data),
                        'range_importance': self._calculate_range_importance(feature_data),
                        'entropy_importance': self._calculate_entropy_importance(feature_data),
                        'mean_magnitude': self._calculate_mean_magnitude(feature_data),
                        'zero_ratio': self._calculate_zero_ratio(feature_data)
                    }
                    
                    # 综合评分
                    composite_score = self._calculate_composite_score(scores)
                    scores['composite_score'] = composite_score
                    
                    importance_scores[feature_name] = {
                        'metrics': scores,
                        'analysis_type': 'numeric',
                        'data_shape': feature_data.shape if hasattr(feature_data, 'shape') else None,
                        'data_type': str(feature_data.dtype) if hasattr(feature_data, 'dtype') else type(feature_data).__name__
                    }
                    
                except Exception as e:
                    logger.warning(f"分析数值特征 {feature_name} 时出错: {e}")
                    importance_scores[feature_name] = {
                        'metrics': {'error': str(e)},
                        'analysis_type': 'numeric',
                        'error': True
                    }
        
        return importance_scores
    
    def _calculate_variance_importance(self, data: torch.Tensor) -> float:
        """计算方差重要性"""
        if not torch.is_tensor(data):
            return 0.0
        
        try:
            data_cpu = data.cpu()
            if len(data_cpu.shape) == 2:
                # 对于2D数据，计算所有值的方差
                variance = float(torch.var(data_cpu.float()))
            else:
                # 对于高维数据，计算最后一个维度的平均方差
                variance = float(torch.mean(torch.var(data_cpu.float(), dim=-1)))
            
            return max(0.0, variance)  # 确保非负
            
        except Exception as e:
            logger.warning(f"计算方差时出错: {e}")
            return 0.0
    
    def _calculate_range_importance(self, data: torch.Tensor) -> float:
        """计算范围重要性"""
        if not torch.is_tensor(data):
            return 0.0
        
        try:
            data_cpu = data.cpu().float()
            data_range = float(torch.max(data_cpu) - torch.min(data_cpu))
            
            # 归一化范围，避免极大值
            normalized_range = min(data_range / (torch.mean(torch.abs(data_cpu)) + 1e-8), 100.0)
            return max(0.0, normalized_range)
            
        except Exception as e:
            logger.warning(f"计算范围时出错: {e}")
            return 0.0
    
    def _calculate_entropy_importance(self, data: torch.Tensor) -> float:
        """计算信息熵重要性"""
        if not torch.is_tensor(data):
            return 0.0
        
        try:
            data_cpu = data.cpu().float()
            # 将数据离散化计算熵
            data_flat = data_cpu.flatten()
            
            # 创建直方图
            hist = torch.histc(data_flat, bins=50, min=torch.min(data_flat), max=torch.max(data_flat))
            hist = hist[hist > 0]  # 移除零计数
            
            # 计算概率分布
            prob = hist / torch.sum(hist)
            
            # 计算熵
            entropy = -torch.sum(prob * torch.log(prob + 1e-8))
            
            return float(entropy)
            
        except Exception as e:
            logger.warning(f"计算熵时出错: {e}")
            return 0.0
    
    def _calculate_mean_magnitude(self, data: torch.Tensor) -> float:
        """计算平均幅值"""
        if not torch.is_tensor(data):
            return 0.0
        
        try:
            return float(torch.mean(torch.abs(data.cpu().float())))
        except:
            return 0.0
    
    def _calculate_zero_ratio(self, data: torch.Tensor) -> float:
        """计算零值比例"""
        if not torch.is_tensor(data):
            return 0.0
        
        try:
            zero_ratio = float((data.cpu() == 0).float().mean())
            return zero_ratio
        except:
            return 0.0
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """计算综合评分"""
        try:
            # 加权组合各项指标
            weights = {
                'variance_importance': 0.4,
                'range_importance': 0.2,
                'entropy_importance': 0.3,
                'mean_magnitude': 0.1
            }
            
            composite = sum(scores.get(metric, 0.0) * weight 
                           for metric, weight in weights.items())
            
            return composite
            
        except Exception as e:
            logger.warning(f"计算综合评分时出错: {e}")
            return 0.0


class SequenceAnalysisStrategy(BaseAnalysisStrategy):
    """序列特征分析策略"""
    
    def get_importance_metrics(self) -> List[str]:
        return ['temporal_importance', 'complexity_importance', 'entropy_importance', 'composite_score']
    
    def analyze(self, model: Any, features: List[str], batch: Dict[str, Any]) -> Dict[str, Any]:
        """分析序列特征"""
        importance_scores = {}
        
        for feature_name in features:
            if feature_name in batch:
                feature_data = batch[feature_name]
                
                try:
                    # 序列特征分析
                    scores = {
                        'temporal_importance': self._calculate_temporal_importance(feature_data),
                        'complexity_importance': self._calculate_complexity_importance(feature_data),
                        'entropy_importance': self._calculate_sequence_entropy(feature_data),
                        'length_variance': self._calculate_length_variance(feature_data),
                        'pattern_diversity': self._calculate_pattern_diversity(feature_data)
                    }
                    
                    # 综合评分
                    composite_score = self._calculate_composite_score(scores)
                    scores['composite_score'] = composite_score
                    
                    importance_scores[feature_name] = {
                        'metrics': scores,
                        'analysis_type': 'sequence',
                        'data_shape': feature_data.shape if hasattr(feature_data, 'shape') else None,
                        'sequence_type': self._detect_sequence_type(feature_name)
                    }
                    
                except Exception as e:
                    logger.warning(f"分析序列特征 {feature_name} 时出错: {e}")
                    importance_scores[feature_name] = {
                        'metrics': {'error': str(e)},
                        'analysis_type': 'sequence',
                        'error': True
                    }
        
        return importance_scores
    
    def _calculate_temporal_importance(self, data: torch.Tensor) -> float:
        """计算时序重要性"""
        if not torch.is_tensor(data):
            return 0.0
        
        try:
            data_cpu = data.cpu().float()
            
            # 检查数据有效性
            if torch.all(data_cpu == 0):
                logger.debug("时序数据全为零，返回0.0")
                return 0.0
            
            # 检查是否为常数序列
            if torch.var(data_cpu) < 1e-8:
                logger.debug("时序数据为常数序列，返回0.0")
                return 0.0
            
            # 计算时间序列的自相关性
            if len(data_cpu.shape) >= 2:
                # 计算每个序列的自相关性
                autocorr_scores = []
                for i in range(min(data_cpu.shape[0], 10)):  # 最多分析10个序列
                    seq = data_cpu[i]
                    if len(seq) > 1 and torch.var(seq) > 1e-8:  # 只分析有变化的序列
                        try:
                            # 计算滞后1的自相关性
                            seq_diff = seq[:-1] - seq[1:]
                            correlation = torch.corrcoef(torch.stack([seq[:-1], seq[1:]]))
                            if not torch.isnan(correlation).any():
                                autocorr_scores.append(abs(correlation[0, 1].item()))
                        except Exception as e:
                            logger.debug(f"序列 {i} 自相关计算失败: {e}")
                            autocorr_scores.append(0.0)
                
                result = np.mean(autocorr_scores) if autocorr_scores else 0.0
                # 确保返回有效值
                return float(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0))
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"计算时序重要性时出错: {e}")
            return 0.0
    
    def _calculate_complexity_importance(self, data: torch.Tensor) -> float:
        """计算复杂度重要性"""
        if not torch.is_tensor(data):
            return 0.0
        
        try:
            data_cpu = data.cpu().float()
            
            if len(data_cpu.shape) >= 2:
                # 计算序列的变化率
                complexity_scores = []
                for i in range(min(data_cpu.shape[0], 10)):
                    seq = data_cpu[i]
                    if len(seq) > 1:
                        # 计算差分的方差作为复杂度指标
                        diffs = torch.diff(seq)
                        complexity = torch.var(diffs).item()
                        complexity_scores.append(complexity)
                
                return np.mean(complexity_scores) if complexity_scores else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"计算复杂度重要性时出错: {e}")
            return 0.0
    
    def _calculate_sequence_entropy(self, data: torch.Tensor) -> float:
        """计算序列信息熵"""
        if not torch.is_tensor(data):
            return 0.0
        
        try:
            data_cpu = data.cpu().float()
            data_flat = data_cpu.flatten()
            
            # 离散化数据
            hist = torch.histc(data_flat, bins=20, min=torch.min(data_flat), max=torch.max(data_flat))
            hist = hist[hist > 0]
            
            # 计算熵
            prob = hist / torch.sum(hist)
            entropy = -torch.sum(prob * torch.log(prob + 1e-8))
            
            return float(entropy)
            
        except Exception as e:
            logger.warning(f"计算序列熵时出错: {e}")
            return 0.0
    
    def _calculate_length_variance(self, data: torch.Tensor) -> float:
        """计算长度方差"""
        if not torch.is_tensor(data):
            return 0.0
        
        try:
            # 对于序列数据，计算有效长度（非零元素）的方差
            data_cpu = data.cpu().float()
            
            if len(data_cpu.shape) >= 2:
                lengths = []
                for i in range(data_cpu.shape[0]):
                    seq = data_cpu[i]
                    # 计算非零元素的比例作为有效长度
                    valid_length = (seq != 0).float().mean().item()
                    lengths.append(valid_length)
                
                return np.var(lengths) if lengths else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"计算长度方差时出错: {e}")
            return 0.0
    
    def _calculate_pattern_diversity(self, data: torch.Tensor) -> float:
        """计算模式多样性"""
        if not torch.is_tensor(data):
            return 0.0
        
        try:
            data_cpu = data.cpu().float()
            
            # 简单的模式多样性：计算不同值的出现频率
            unique_values = len(torch.unique(data_cpu))
            total_values = data_cpu.numel()
            
            # 归一化的多样性
            diversity = unique_values / (total_values + 1e-8)
            
            return diversity
            
        except Exception as e:
            logger.warning(f"计算模式多样性时出错: {e}")
            return 0.0
    
    def _detect_sequence_type(self, feature_name: str) -> str:
        """检测序列类型"""
        if 'iat' in feature_name:
            return 'inter_arrival_time'
        elif 'payload' in feature_name:
            return 'payload_size'
        elif 'direction' in feature_name:
            return 'packet_direction'
        else:
            return 'generic_sequence'
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """计算综合评分"""
        try:
            weights = {
                'temporal_importance': 0.3,
                'complexity_importance': 0.3,
                'entropy_importance': 0.2,
                'pattern_diversity': 0.2
            }
            
            composite = sum(scores.get(metric, 0.0) * weight 
                           for metric, weight in weights.items())
            
            return composite
            
        except Exception as e:
            logger.warning(f"计算序列综合评分时出错: {e}")
            return 0.0


class TextAnalysisStrategy(BaseAnalysisStrategy):
    """文本特征分析策略"""
    
    def get_importance_metrics(self) -> List[str]:
        return ['length_importance', 'richness_importance', 'diversity_importance', 'composite_score']
    
    def analyze(self, model: Any, features: List[str], batch: Dict[str, Any]) -> Dict[str, Any]:
        """分析文本特征"""
        importance_scores = {}
        
        for feature_name in features:
            if feature_name in batch:
                feature_data = batch[feature_name]
                
                try:
                    # 文本特征分析
                    scores = {
                        'length_importance': self._calculate_length_importance(feature_data),
                        'richness_importance': self._calculate_richness_importance(feature_data),
                        'diversity_importance': self._calculate_diversity_importance(feature_data),
                        'non_empty_ratio': self._calculate_non_empty_ratio(feature_data),
                        'avg_word_count': self._calculate_avg_word_count(feature_data)
                    }
                    
                    # 综合评分
                    composite_score = self._calculate_composite_score(scores)
                    scores['composite_score'] = composite_score
                    
                    importance_scores[feature_name] = {
                        'metrics': scores,
                        'analysis_type': 'text',
                        'text_type': self._detect_text_type(feature_name),
                        'sample_count': len(feature_data) if isinstance(feature_data, (list, tuple)) else 1
                    }
                    
                except Exception as e:
                    logger.warning(f"分析文本特征 {feature_name} 时出错: {e}")
                    importance_scores[feature_name] = {
                        'metrics': {'error': str(e)},
                        'analysis_type': 'text',
                        'error': True
                    }
        
        return importance_scores
    
    def _analyze_text_data(self, data: Any) -> List[str]:
        """将输入数据转换为字符串列表"""
        if isinstance(data, torch.Tensor):
            # 如果是tensor，尝试转换为字符串列表
            if data.numel() == 1:
                return [str(data.item())]
            else:
                return [str(x.item()) for x in data.flatten()]
        elif isinstance(data, list):
            return [str(x) for x in data]
        elif isinstance(data, (tuple,)):
            return [str(x) for x in data]
        else:
            # 其他类型转为字符串
            return [str(data)]
    
    def _calculate_length_importance(self, data: Any) -> float:
        """计算长度重要性"""
        try:
            text_data = self._analyze_text_data(data)
            lengths = [len(text) for text in text_data if isinstance(text, str)]
            if not lengths:
                return 0.0
            
            # 长度的方差作为重要性指标
            return np.var(lengths)
            
        except Exception as e:
            logger.warning(f"计算长度重要性时出错: {e}")
            return 0.0
    
    def _calculate_richness_importance(self, data: Any) -> float:
        """计算丰富度重要性"""
        try:
            # 计算词汇丰富度（不同词汇的比例）
            text_data = self._analyze_text_data(data)
            all_words = []
            for text in text_data:
                if isinstance(text, str):
                    words = text.lower().split()
                    all_words.extend(words)
            
            if not all_words:
                return 0.0
            
            unique_words = set(all_words)
            richness = len(unique_words) / len(all_words)
            
            return richness
            
        except Exception as e:
            logger.warning(f"计算丰富度重要性时出错: {e}")
            return 0.0
    
    def _calculate_diversity_importance(self, data: Any) -> float:
        """计算多样性重要性"""
        try:
            # 计算文本内容的多样性（不同文本的比例）
            text_data = self._analyze_text_data(data)
            if not text_data:
                return 0.0
            
            unique_texts = len(set(text_data))
            diversity = unique_texts / len(text_data)
            
            return diversity
            
        except Exception as e:
            logger.warning(f"计算多样性重要性时出错: {e}")
            return 0.0
    
    def _calculate_non_empty_ratio(self, data: Any) -> float:
        """计算非空文本比例"""
        try:
            text_data = self._analyze_text_data(data)
            non_empty_count = sum(1 for text in text_data if isinstance(text, str) and text.strip())
            return non_empty_count / len(text_data) if text_data else 0.0
            
        except Exception as e:
            logger.warning(f"计算非空比例时出错: {e}")
            return 0.0
    
    def _calculate_avg_word_count(self, data: Any) -> float:
        """计算平均词数"""
        try:
            text_data = self._analyze_text_data(data)
            word_counts = []
            for text in text_data:
                if isinstance(text, str):
                    words = text.split()
                    word_counts.append(len(words))
            
            return np.mean(word_counts) if word_counts else 0.0
            
        except Exception as e:
            logger.warning(f"计算平均词数时出错: {e}")
            return 0.0
    
    def _detect_text_type(self, feature_name: str) -> str:
        """检测文本类型"""
        if 'ssl' in feature_name or 'server_name' in feature_name:
            return 'domain_name'
        elif 'dns' in feature_name or 'query' in feature_name:
            return 'dns_query'
        elif 'cert' in feature_name or 'subject' in feature_name:
            return 'certificate_subject'
        elif 'cert' in feature_name or 'issuer' in feature_name:
            return 'certificate_issuer'
        else:
            return 'generic_text'
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """计算综合评分"""
        try:
            weights = {
                'length_importance': 0.2,
                'richness_importance': 0.3,
                'diversity_importance': 0.3,
                'non_empty_ratio': 0.2
            }
            
            composite = sum(scores.get(metric, 0.0) * weight 
                           for metric, weight in weights.items())
            
            return composite
            
        except Exception as e:
            logger.warning(f"计算文本综合评分时出错: {e}")
            return 0.0


class CategoricalAnalysisStrategy(BaseAnalysisStrategy):
    """分类特征分析策略"""
    
    def get_importance_metrics(self) -> List[str]:
        return ['entropy_importance', 'balance_importance', 'diversity_importance', 'composite_score']
    
    def analyze(self, model: Any, features: List[str], batch: Dict[str, Any]) -> Dict[str, Any]:
        """分析分类特征"""
        importance_scores = {}
        
        for feature_name in features:
            if feature_name in batch:
                feature_data = batch[feature_name]
                
                try:
                    # 分类特征分析
                    scores = {
                        'entropy_importance': self._calculate_entropy_importance(feature_data),
                        'balance_importance': self._calculate_balance_importance(feature_data),
                        'diversity_importance': self._calculate_diversity_importance(feature_data),
                        'category_count': self._get_category_count(feature_data)
                    }
                    
                    # 综合评分
                    composite_score = self._calculate_composite_score(scores)
                    scores['composite_score'] = composite_score
                    
                    importance_scores[feature_name] = {
                        'metrics': scores,
                        'analysis_type': 'categorical',
                        'category_count': scores['category_count']
                    }
                    
                except Exception as e:
                    logger.warning(f"分析分类特征 {feature_name} 时出错: {e}")
                    importance_scores[feature_name] = {
                        'metrics': {'error': str(e)},
                        'analysis_type': 'categorical',
                        'error': True
                    }
        
        return importance_scores
    
    def _calculate_entropy_importance(self, data: torch.Tensor) -> float:
        """计算分类熵"""
        try:
            data_cpu = data.cpu().long()
            unique, counts = torch.unique(data_cpu, return_counts=True)
            
            # 计算概率分布
            prob = counts.float() / torch.sum(counts)
            
            # 计算熵
            entropy = -torch.sum(prob * torch.log(prob + 1e-8))
            
            return float(entropy)
            
        except Exception as e:
            logger.warning(f"计算分类熵时出错: {e}")
            return 0.0
    
    def _calculate_balance_importance(self, data: torch.Tensor) -> float:
        """计算平衡重要性（类别的均匀性）"""
        try:
            data_cpu = data.cpu().long()
            unique, counts = torch.unique(data_cpu, return_counts=True)
            
            # 计算基尼系数作为不平衡指标
            n = len(data_cpu)
            gini = 1.0 - torch.sum((counts / n) ** 2)
            
            return float(gini)
            
        except Exception as e:
            logger.warning(f"计算平衡重要性时出错: {e}")
            return 0.0
    
    def _calculate_diversity_importance(self, data: torch.Tensor) -> float:
        """计算多样性重要性"""
        try:
            data_cpu = data.cpu().long()
            unique_count = len(torch.unique(data_cpu))
            total_count = len(data_cpu)
            
            diversity = unique_count / total_count
            
            return diversity
            
        except Exception as e:
            logger.warning(f"计算分类多样性时出错: {e}")
            return 0.0
    
    def _get_category_count(self, data: torch.Tensor) -> int:
        """获取类别数量"""
        try:
            return len(torch.unique(data.cpu().long()))
        except:
            return 0
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """计算综合评分"""
        try:
            weights = {
                'entropy_importance': 0.4,
                'balance_importance': 0.3,
                'diversity_importance': 0.3
            }
            
            composite = sum(scores.get(metric, 0.0) * weight 
                           for metric, weight in weights.items())
            
            return composite
            
        except Exception as e:
            logger.warning(f"计算分类综合评分时出错: {e}")
            return 0.0


class AnalysisStrategyRegistry:
    """分析策略注册表"""
    
    def __init__(self):
        self.strategies = {}
        self._register_default_strategies()
    
    def register_strategy(self, feature_type: str, strategy: BaseAnalysisStrategy):
        """注册新的分析策略"""
        self.strategies[feature_type] = strategy
        logger.info(f"已注册分析策略: {feature_type}")
    
    def get_strategy(self, feature_type: str) -> BaseAnalysisStrategy:
        """获取分析策略"""
        return self.strategies.get(feature_type)
    
    def _register_default_strategies(self):
        """注册默认分析策略"""
        self.register_strategy('numeric', NumericAnalysisStrategy())
        self.register_strategy('sequence', SequenceAnalysisStrategy())
        self.register_strategy('text', TextAnalysisStrategy())
        self.register_strategy('categorical', CategoricalAnalysisStrategy())
    
    def list_available_strategies(self) -> List[str]:
        """列出所有可用的分析策略"""
        return list(self.strategies.keys())