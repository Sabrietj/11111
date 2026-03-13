"""
评估指标基础类和分类指标（基于PyTorch实现，禁止使用sklearn）
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseMetric(ABC):
    """指标基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    @abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        """更新指标"""
        pass
    
    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """计算指标值"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置指标"""
        pass

def calculate_auc_pr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    基于PyTorch计算AUC-PR（精确率-召回率曲线下面积）
    
    Args:
        predictions: 预测概率 [batch_size, num_classes] 或 [batch_size]
        targets: 真实标签 [batch_size]
    
    Returns:
        auc_pr: AUC-PR值
    """
    if predictions.dim() == 1:
        # 二分类情况
        predictions = torch.stack([1 - predictions, predictions], dim=1)
    
    num_classes = predictions.shape[1]
    auc_pr_values = []
    
    for class_idx in range(num_classes):
        # 获取当前类别的预测概率和真实标签
        class_probs = predictions[:, class_idx]
        class_targets = (targets == class_idx).float()
        
        # 排序预测概率
        sorted_indices = torch.argsort(class_probs, descending=True)
        sorted_probs = class_probs[sorted_indices]
        sorted_targets = class_targets[sorted_indices]
        
        # 计算精确率和召回率
        cum_tp = torch.cumsum(sorted_targets, dim=0)
        cum_fp = torch.cumsum(1 - sorted_targets, dim=0)
        
        precision = cum_tp / (cum_tp + cum_fp + 1e-8)
        recall = cum_tp / (torch.sum(sorted_targets) + 1e-8)
        
        # 计算AUC-PR（梯形法则）
        auc_pr = torch.trapz(precision, recall)
        # 确保auc_pr是标量
        if auc_pr.numel() > 1:
            auc_pr = auc_pr.mean()
        auc_pr_values.append(auc_pr.item())
    
    # 返回宏平均AUC-PR
    return float(np.mean(auc_pr_values))

class ClassificationMetrics(BaseMetric):
    """分类指标（基于PyTorch实现）"""
    
    def __init__(self, num_classes: int, compute_per_class: bool = True):
        super().__init__('classification')
        self.num_classes = num_classes
        self.compute_per_class = compute_per_class
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               probabilities: Optional[torch.Tensor] = None, **kwargs):
        """
        更新分类指标
        
        Args:
            predictions: 预测标签 [batch_size]
            targets: 真实标签 [batch_size]
            probabilities: 预测概率 [batch_size, num_classes]
        """
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
        
        if probabilities is not None:
            self.probabilities.append(probabilities.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """计算分类指标（基于PyTorch）"""
        if not self.predictions:
            return {}
        
        # 合并所有批次数据
        y_pred = torch.cat(self.predictions)
        y_true = torch.cat(self.targets)
        
        metrics = {}
        
        # 准确率
        accuracy = (y_pred == y_true).float().mean()
        metrics['accuracy'] = accuracy.item()
        
        # 宏平均精确率、召回率、F1分数
        precision_macro, recall_macro, f1_macro = self._compute_macro_metrics(y_pred, y_true)
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        
        # 加权平均精确率、召回率、F1分数
        precision_weighted, recall_weighted, f1_weighted = self._compute_weighted_metrics(y_pred, y_true)
        metrics['precision_weighted'] = precision_weighted
        metrics['recall_weighted'] = recall_weighted
        metrics['f1_weighted'] = f1_weighted
        
        # 每类指标
        if self.compute_per_class:
            for class_idx in range(self.num_classes):
                class_metrics = self._compute_class_metrics(y_pred, y_true, class_idx)
                metrics.update({
                    f'precision_class_{class_idx}': class_metrics['precision'],
                    f'recall_class_{class_idx}': class_metrics['recall'],
                    f'f1_class_{class_idx}': class_metrics['f1']
                })
        
        # AUC-PR（如果有概率）
        if self.probabilities:
            y_prob = torch.cat(self.probabilities)
            auc_pr = calculate_auc_pr(y_prob, y_true)
            metrics['auc_pr'] = auc_pr
        
        return metrics
        
    def _compute_macro_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[float, float, float]:
        """计算宏平均精确率、召回率和F1分数"""
        precision_list = []
        recall_list = []
        f1_list = []
        
        for class_idx in range(self.num_classes):
            # 计算当前类别的TP, FP, FN
            tp = ((y_pred == class_idx) & (y_true == class_idx)).sum().float()
            fp = ((y_pred == class_idx) & (y_true != class_idx)).sum().float()
            fn = ((y_pred != class_idx) & (y_true == class_idx)).sum().float()
            
            # 计算精确率、召回率和F1分数
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            precision_list.append(precision.item())
            recall_list.append(recall.item())
            f1_list.append(f1.item())
        
        # 计算宏平均
        precision_macro = sum(precision_list) / len(precision_list)
        recall_macro = sum(recall_list) / len(recall_list)
        f1_macro = sum(f1_list) / len(f1_list)
        
        return precision_macro, recall_macro, f1_macro
    
    def _compute_weighted_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[float, float, float]:
        """计算加权平均精确率、召回率和F1分数"""
        precision_list = []
        recall_list = []
        f1_list = []
        class_weights = []
        
        for class_idx in range(self.num_classes):
            # 计算当前类别的样本权重
            class_count = (y_true == class_idx).sum().float()
            class_weight = class_count / len(y_true)
            class_weights.append(class_weight.item())
            
            # 计算当前类别的TP, FP, FN
            tp = ((y_pred == class_idx) & (y_true == class_idx)).sum().float()
            fp = ((y_pred == class_idx) & (y_true != class_idx)).sum().float()
            fn = ((y_pred != class_idx) & (y_true == class_idx)).sum().float()
            
            # 计算精确率、召回率和F1分数
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            precision_list.append(precision.item())
            recall_list.append(recall.item())
            f1_list.append(f1.item())
        
        # 计算加权平均
        precision_weighted = sum(p * w for p, w in zip(precision_list, class_weights))
        recall_weighted = sum(r * w for r, w in zip(recall_list, class_weights))
        f1_weighted = sum(f * w for f, w in zip(f1_list, class_weights))
        
        return precision_weighted, recall_weighted, f1_weighted
    
    def _compute_class_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_idx: int) -> Dict[str, float]:
        """计算单个类别的指标"""
        # 计算当前类别的TP, FP, FN
        tp = ((y_pred == class_idx) & (y_true == class_idx)).sum().float()
        fp = ((y_pred == class_idx) & (y_true != class_idx)).sum().float()
        fn = ((y_pred != class_idx) & (y_true == class_idx)).sum().float()
        
        # 计算精确率、召回率和F1分数
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }

def calculate_classification_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                                    probabilities: Optional[torch.Tensor] = None,
                                    num_classes: int = 7) -> Dict[str, float]:
    """
    便捷函数：直接计算分类指标
    
    Args:
        predictions: 预测标签 [batch_size]
        targets: 真实标签 [batch_size]
        probabilities: 预测概率 [batch_size, num_classes]
        num_classes: 类别数量
    
    Returns:
        分类指标字典
    """
    metric = ClassificationMetrics(num_classes=num_classes)
    metric.update(predictions, targets, probabilities)
    result = metric.compute()
    # 确保总是返回字典，即使没有数据
    return result if result is not None else {}


class LossMetric(BaseMetric):
    """损失指标"""
    
    def __init__(self, name: str = 'loss'):
        super().__init__(name)
        self.reset()
    
    def reset(self):
        self.loss_values = []
        self.total_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               loss: Optional[torch.Tensor] = None, **kwargs):
        """更新损失指标"""
        if loss is not None:
            self.loss_values.append(loss.detach().cpu().item())
            self.total_samples += targets.size(0)
    
    def compute(self) -> Dict[str, float]:
        """计算平均损失"""
        if not self.loss_values:
            return {f'{self.name}': 0.0}
        
        avg_loss = np.mean(self.loss_values)
        return {f'{self.name}': avg_loss}

class AttentionWeightMetric(BaseMetric):
    """注意力权重指标（适配实际6个视图）"""
    
    def __init__(self):
        super().__init__('attention_weights')
        self.reset()
    
    def reset(self):
        # 适配实际的6个视图
        self.attention_weights = {
            'flow_statistics': [],
            'ssl_features': [], 
            'x509_features': [],
            'dns_features': [],
            'packet_length': [],
            'packet_iat': []
        }
    
    def update(self, attention_weights: Dict[str, torch.Tensor], **kwargs):
        """更新注意力权重"""
        for view_name, weights in attention_weights.items():
            if view_name in self.attention_weights:
                self.attention_weights[view_name].append(weights.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """计算平均注意力权重"""
        metrics = {}
        
        for view_name, weights_list in self.attention_weights.items():
            if weights_list:
                avg_weight = torch.mean(torch.cat(weights_list)).item()
                metrics[f'attention_weight_{view_name}'] = avg_weight
        
        return metrics