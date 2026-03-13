"""
损失函数注册器
支持多种损失函数的注册、组合和动态调度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import math
import logging

logger = logging.getLogger(__name__)

class BaseLoss(ABC):
    """损失函数基类"""
    
    def __init__(self, weight: float = 1.0, **kwargs):
        self.weight = weight
        self.kwargs = kwargs
    
    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, 
                **kwargs) -> torch.Tensor:
        """计算损失值"""
        pass
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 **kwargs) -> torch.Tensor:
        loss = self.compute(predictions, targets, **kwargs)
        return self.weight * loss

class CrossEntropyLoss(BaseLoss):
    """交叉熵损失"""
    
    def __init__(self, weight: float = 1.0, label_smoothing: float = 0.0, 
                 reduction: str = 'mean', **kwargs):
        super().__init__(weight, **kwargs)
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, 
                **kwargs) -> torch.Tensor:
        return F.cross_entropy(predictions, targets, 
                             label_smoothing=self.label_smoothing,
                             reduction=self.reduction)

class FocalLoss(BaseLoss):
    """Focal Loss - 处理类别不平衡"""
    
    def __init__(self, weight: float = 1.0, alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, reduction: str = 'mean', **kwargs):
        super().__init__(weight, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, 
                **kwargs) -> torch.Tensor:
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ReconstructionLoss(BaseLoss):
    """重构损失"""
    
    def __init__(self, weight: float = 1.0, loss_type: str = 'mse', 
                 reduction: str = 'mean', **kwargs):
        super().__init__(weight, **kwargs)
        self.loss_type = loss_type.lower()
        self.reduction = reduction
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, 
                **kwargs) -> torch.Tensor:
        if self.loss_type == 'mse':
            return F.mse_loss(predictions, targets, reduction=self.reduction)
        elif self.loss_type == 'mae':
            return F.l1_loss(predictions, targets, reduction=self.reduction)
        elif self.loss_type == 'huber':
            delta = self.kwargs.get('delta', 1.0)
            return F.huber_loss(predictions, targets, delta=delta, 
                              reduction=self.reduction)
        else:
            raise ValueError(f"未知的重构损失类型: {self.loss_type}")

class ContrastiveLoss(BaseLoss):
    """对比学习损失"""
    
    def __init__(self, weight: float = 1.0, temperature: float = 0.1, 
                 similarity_metric: str = 'cosine', **kwargs):
        super().__init__(weight, **kwargs)
        self.temperature = temperature
        self.similarity_metric = similarity_metric
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, 
                labels: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # predictions作为features1, targets作为features2
        features1, features2 = predictions, targets
        batch_size = features1.size(0)
        
        # 计算相似度矩阵
        if self.similarity_metric == 'cosine':
            features1_norm = F.normalize(features1, dim=1)
            features2_norm = F.normalize(features2, dim=1)
            similarity = torch.mm(features1_norm, features2_norm.t()) / self.temperature
        else:
            raise ValueError(f"未知的相似度度量: {self.similarity_metric}")
        
        # 构建标签矩阵
        if labels is None:
            # 自监督：对角线为正样本
            labels = torch.arange(batch_size, device=features1.device)
        
        # 计算对比损失
        loss = F.cross_entropy(similarity, labels)
        return loss

class MLMLoss(BaseLoss):
    """掩码语言模型损失"""
    
    def __init__(self, weight: float = 1.0, loss_type: str = 'mse', 
                 reduction: str = 'mean', **kwargs):
        super().__init__(weight, **kwargs)
        self.loss_type = loss_type
        self.reduction = reduction
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if mask is not None:
            # 只计算被掩码位置的损失
            predictions = predictions[mask]
            targets = targets[mask]
        
        if self.loss_type == 'mse':
            return F.mse_loss(predictions, targets, reduction=self.reduction)
        elif self.loss_type == 'mae':
            return F.l1_loss(predictions, targets, reduction=self.reduction)
        else:
            raise ValueError(f"未知的MLM损失类型: {self.loss_type}")

class RegularizationLoss(BaseLoss):
    """正则化损失"""
    
    def __init__(self, weight: float = 1.0, l1_weight: float = 0.0, 
                 l2_weight: float = 1e-4, **kwargs):
        super().__init__(weight, **kwargs)
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, 
                model: Optional[nn.Module] = None, **kwargs) -> torch.Tensor:
        if model is None:
            return torch.tensor(0.0, device=predictions.device)
        l1_loss = 0.0
        l2_loss = 0.0
        
        device = predictions.device
        for param in model.parameters():
            if param.requires_grad:
                if self.l1_weight > 0:
                    l1_loss += torch.sum(torch.abs(param))
                if self.l2_weight > 0:
                    l2_loss += torch.sum(param ** 2)
        
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        return total_loss.to(device) if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss, device=device)

class LossRegistry:
    """损失函数注册器"""
    
    def __init__(self):
        self._losses = {}
        self._register_default_losses()
    
    def _register_default_losses(self):
        """注册默认损失函数"""
        self.register('cross_entropy', CrossEntropyLoss)
        self.register('focal_loss', FocalLoss)
        self.register('reconstruction', ReconstructionLoss)
        self.register('contrastive', ContrastiveLoss)
        self.register('mlm', MLMLoss)
        self.register('regularization', RegularizationLoss)
    
    def register(self, name: str, loss_class: type):
        """注册损失函数"""
        self._losses[name] = loss_class
        logger.info(f"注册损失函数: {name}")
    
    def create_loss(self, name: str, **kwargs) -> BaseLoss:
        """创建损失函数实例"""
        if name not in self._losses:
            raise ValueError(f"未知的损失函数: {name}")
        
        return self._losses[name](**kwargs)
    
    def list_losses(self) -> List[str]:
        """列出所有注册的损失函数"""
        return list(self._losses.keys())

class CompositeLoss(nn.Module):
    """复合损失函数"""
    
    def __init__(self, loss_config: Dict[str, Any], loss_registry: LossRegistry):
        super().__init__()
        self.loss_registry = loss_registry
        self.primary_loss = None
        self.auxiliary_losses = []
        self.loss_weights = {}
        
        # 解析损失配置
        self._parse_loss_config(loss_config)
        
        # 权重调度器
        self.weight_scheduler = None
        if 'weight_scheduling' in loss_config:
            self.weight_scheduler = WeightScheduler(loss_config['weight_scheduling'])
    
    def _parse_loss_config(self, loss_config: Dict[str, Any]):
        """解析损失配置"""
        # 主损失
        if 'primary' in loss_config:
            primary_config = loss_config['primary']
            self.primary_loss = self.loss_registry.create_loss(
                primary_config['name'], **primary_config.get('params', {}))
            self.primary_loss.weight = primary_config.get('weight', 1.0)
        
        # 辅助损失
        if 'auxiliary' in loss_config:
            for aux_config in loss_config['auxiliary']:
                aux_loss = self.loss_registry.create_loss(
                    aux_config['name'], **aux_config.get('params', {}))
                aux_loss.weight = aux_config.get('weight', 1.0)
                # 使用setattr动态添加属性
                setattr(aux_loss, 'apply_to', aux_config.get('apply_to', []))
                self.auxiliary_losses.append(aux_loss)
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor], 
                model: Optional[nn.Module] = None, epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算复合损失
        
        Args:
            outputs: 模型输出字典
            targets: 目标值字典
            model: 模型（用于正则化损失）
            epoch: 当前轮数（用于权重调度）
            
        Returns:
            总损失和各项损失详情
        """
        total_loss = 0.0
        loss_details = {}
        
        # 更新权重调度
        if self.weight_scheduler:
            self.weight_scheduler.step(epoch)
        
        # 计算主损失
        if self.primary_loss and 'predictions' in outputs and 'labels' in targets:
            primary_loss_value = self.primary_loss(outputs['predictions'], targets['labels'])
            total_loss += primary_loss_value
            loss_details['primary_loss'] = primary_loss_value.item()
        
        # 计算辅助损失
        for i, aux_loss in enumerate(self.auxiliary_losses):
            aux_loss_value = self._compute_auxiliary_loss(aux_loss, outputs, targets, model)
            if aux_loss_value is not None:
                # 应用权重调度
                if self.weight_scheduler:
                    weight = self.weight_scheduler.get_weight(f"aux_{i}", aux_loss.weight)
                    aux_loss_value = weight * aux_loss_value
                
                total_loss += aux_loss_value
                loss_details[f'aux_loss_{i}'] = aux_loss_value.item() if isinstance(aux_loss_value, torch.Tensor) else float(aux_loss_value)
        
        loss_details['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss)
        return total_loss if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss), loss_details
    
    def _compute_auxiliary_loss(self, aux_loss: BaseLoss, outputs: Dict[str, torch.Tensor],
                               targets: Dict[str, torch.Tensor], model: Optional[nn.Module]) -> Optional[torch.Tensor]:
        """计算辅助损失"""
        loss_name = aux_loss.__class__.__name__
        
        if isinstance(aux_loss, ReconstructionLoss):
            # 重构损失
            apply_to = getattr(aux_loss, 'apply_to', [])
            if apply_to:
                total_loss = 0.0
                count = 0
                for view_name in apply_to:
                    if f'{view_name}_reconstructed' in outputs and view_name in targets:
                        loss_val = aux_loss(outputs[f'{view_name}_reconstructed'], 
                                          targets[view_name])
                        total_loss += loss_val
                        count += 1
                return total_loss / count if count > 0 else torch.tensor(0.0)
        
        elif isinstance(aux_loss, MLMLoss):
            # MLM损失
            apply_to = getattr(aux_loss, 'apply_to', [])
            if apply_to:
                total_loss = 0.0  
                count = 0
                for view_name in apply_to:
                    pred_key = f'{view_name}_mlm_predictions'
                    target_key = f'{view_name}_mlm_targets'
                    mask_key = f'{view_name}_mlm_mask'
                    
                    if pred_key in outputs and target_key in targets:
                        mask = targets.get(mask_key, None)
                        loss_val = aux_loss(outputs[pred_key], targets[target_key], mask=mask)
                        total_loss += loss_val
                        count += 1
                return total_loss / count if count > 0 else torch.tensor(0.0)
        
        elif isinstance(aux_loss, RegularizationLoss):
            # 正则化损失
            if model is not None and 'predictions' in outputs:
                # 创建dummy targets用于满足接口要求
                dummy_targets = torch.zeros_like(outputs['predictions'][:, 0])
                return aux_loss.compute(outputs['predictions'], dummy_targets, model=model)
        
        elif isinstance(aux_loss, ContrastiveLoss):
            # 对比损失
            if 'contrastive_features_1' in outputs and 'contrastive_features_2' in outputs:
                labels = targets.get('contrastive_labels', None)
                return aux_loss.compute(outputs['contrastive_features_1'], 
                                      outputs['contrastive_features_2'], labels=labels)
        
        return None

class WeightScheduler:
    """损失权重调度器"""
    
    def __init__(self, schedule_config: Dict[str, Any]):
        self.schedule_config = schedule_config
        self.current_epoch = 0
        self.warmup_config = schedule_config.get('warmup', {})
        self.decay_config = schedule_config.get('decay', {})
    
    def step(self, epoch: int):
        """更新当前轮数"""
        self.current_epoch = epoch
    
    def get_weight(self, loss_name: str, base_weight: float) -> float:
        """获取调度后的权重"""
        weight = base_weight
        
        # 预热调度
        if self.warmup_config.get('enabled', False):
            warmup_epochs = self.warmup_config.get('warmup_epochs', 5)
            if self.current_epoch < warmup_epochs:
                warmup_method = self.warmup_config.get('warmup_method', 'linear')
                progress = self.current_epoch / warmup_epochs
                
                if warmup_method == 'linear':
                    weight = base_weight * progress
                elif warmup_method == 'cosine':
                    weight = base_weight * (1 - math.cos(math.pi * progress)) / 2
        
        # 衰减调度
        if self.decay_config.get('enabled', False):
            decay_epochs = self.decay_config.get('decay_epochs', 10)
            if self.current_epoch > 0 and self.current_epoch % decay_epochs == 0:
                decay_rate = self.decay_config.get('decay_rate', 0.95)
                decay_method = self.decay_config.get('decay_method', 'exponential')
                
                if decay_method == 'exponential':
                    weight = weight * decay_rate
        
        return weight

# 全局损失注册器
loss_registry = LossRegistry()