import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import os, sys
import matplotlib.pyplot as plt


class MultiViewFusionFactory:
    """多视图融合策略工厂"""
    
    @staticmethod
    def create_fusion_layer(cfg: DictConfig, hidden_size: int, num_views: int):
        fusion_method = cfg.model.multiview.fusion.method
        
        if fusion_method == "cross_attention":
            return CrossAttentionFusion(
                hidden_size=hidden_size,
                num_heads=cfg.model.multiview.fusion.cross_attention_heads,
                dropout=cfg.model.multiview.fusion.cross_attention_dropout,
                num_views=num_views
            )
        elif fusion_method == "weighted_sum":
            return WeightedSumFusion(
                hidden_size=hidden_size,
                num_views=num_views,
                learnable_weights=cfg.model.multiview.fusion.weighted_sum.learnable_weights,
                initial_weights=cfg.model.multiview.fusion.weighted_sum.initial_weights
            )
        elif fusion_method == "concat":
            return ConcatFusion(
                hidden_size=hidden_size,
                num_views=num_views,
            )
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")

class WeightedSumFusion(nn.Module):
    """加权求和多视图融合"""
    
    def __init__(
        self, 
        hidden_size: int,
        num_views: int,
        learnable_weights: bool = True,
        initial_weights: List[float] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_views = num_views
        
        # 视图权重
        if learnable_weights:
            if initial_weights is not None and len(initial_weights) == num_views:
                self.view_weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
            else:
                self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)
        else:
            if initial_weights is not None and len(initial_weights) == num_views:
                self.register_buffer('view_weights', torch.tensor(initial_weights, dtype=torch.float32))
            else:
                self.register_buffer('view_weights', torch.ones(num_views) / num_views)
        
        # 可选的特征变换层
        self.view_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_views)
        ])
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, view_embeddings: List[torch.Tensor]) -> torch.Tensor:
        # 对每个视图进行投影变换
        projected_views = []
        for i, view in enumerate(view_embeddings):
            projected_view = self.view_projections[i](view)
            projected_views.append(projected_view)
        
        # 加权求和
        weights = F.softmax(self.view_weights, dim=0)
        fused = sum(weight * view for weight, view in zip(weights, projected_views))
        
        # 层归一化
        return self.layer_norm(fused)

class ConcatFusion(nn.Module):
    """简化的拼接多视图融合"""
    
    def __init__(
        self, 
        hidden_size: int,
        num_views: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_views = num_views
        
        # 拼接后的总维度 = hidden_size * num_views
        concat_dim = hidden_size * num_views
        
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, hidden_size),  # 直接投影到 hidden_size
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size)
        )

        logger.info(f"[ConcatFusion] 初始化：输入维度 = {concat_dim}, 输出维度 = {hidden_size}")
        
    def forward(self, view_embeddings: List[torch.Tensor]) -> torch.Tensor:
        # 拼接所有视图特征
        concatenated = torch.cat(view_embeddings, dim=1)

        # ---- 打印拼接后的维度 ----
        # logger.info(f"[ConcatFusion] 拼接后 concatenated.shape = {tuple(concatenated.shape)}")
                
        # 直接投影到目标维度
        fused = self.projection(concatenated)

        # ---- 打印投影输出维度 ----
        # logger.info(f"[ConcatFusion] 投影后 fused.shape = {tuple(fused.shape)}")
        
        return fused
    
class CrossAttentionFusion(nn.Module):
    """
    多视图交叉注意力融合层，支持获取注意力权重。
    与 MultiViewFusionFactory.create_fusion_layer 完全兼容。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        num_views: int = 3,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_views = num_views
        self.dropout = dropout

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} 必须能被 num_heads={num_heads} 整除")

        self.head_dim = hidden_size // num_heads

        # === 为每个视图构建 Q/K/V 映射 ===
        self.query_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_views)])
        self.key_proj   = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_views)])
        self.value_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_views)])

        # === 为每个视图的 查询视图 构建一个交叉注意力模块 ===
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_views)
        ])

        # === 输出融合（拼接所有 attended view） ===
        self.output_linear = nn.Linear(hidden_size * num_views, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        for proj_list in [self.query_proj, self.key_proj, self.value_proj]:
            for proj in proj_list:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.constant_(proj.bias, 0)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, 0)

    def forward(self, view_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        view_embeddings: List of [B, hidden_size]
        return: [B, hidden_size]
        """

        # 过滤掉 None（例如 disabled view）
        valid_views = [v for v in view_embeddings if v is not None]
        actual_num_views = len(valid_views)

        if actual_num_views == 0:
            raise ValueError("没有可用的视图输入！")

        batch_size = valid_views[0].size(0)

        attended_outputs = []

        for i in range(actual_num_views):
            # Query 为本视图
            q = self.query_proj[i](valid_views[i]).unsqueeze(1)

            # 其他视图作为 Key/Value
            other_indices = [j for j in range(actual_num_views) if j != i]

            k = torch.stack([self.key_proj[j](valid_views[j]) for j in other_indices], dim=1)
            v = torch.stack([self.value_proj[j](valid_views[j]) for j in other_indices], dim=1)

            out, _ = self.cross_attn[i](q, k, v, need_weights=False)
            attended_outputs.append(out.squeeze(1))

        # 拼接
        fused = torch.cat(attended_outputs, dim=1)

        # 投影到 hidden_size
        projected = self.output_linear(fused)

        # 残差（使用第一个视图）
        output = self.norm(valid_views[0] + self.dropout_layer(projected))

        return output

    def get_attention_weights(self, view_embeddings: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        返回每个视图作为 Query 时的 attention weight
        """
        valid_views = [v for v in view_embeddings if v is not None]
        actual_num_views = len(valid_views)

        weights = {}

        for i in range(actual_num_views):
            q = self.query_proj[i](valid_views[i]).unsqueeze(1)
            other_indices = [j for j in range(actual_num_views) if j != i]

            k = torch.stack([self.key_proj[j](valid_views[j]) for j in other_indices], dim=1)
            v = torch.stack([self.value_proj[j](valid_views[j]) for j in other_indices], dim=1)

            _, attn = self.cross_attn[i](q, k, v, need_weights=True)

            weights[f"view_{i}_attn"] = attn  # [B, 1, num_other]

        return weights
    