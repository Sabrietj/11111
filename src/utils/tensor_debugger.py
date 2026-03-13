import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import os, sys
import matplotlib.pyplot as plt

# 导入配置管理器和相关模块
try:
    import config_manager as ConfigManager
    from logging_config import setup_preset_logging
    # 使用统一的日志配置
    logger = setup_preset_logging(log_level=logging.INFO)    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目结构正确，所有依赖模块可用")
    sys.exit(1)


# ---------------- NaN 检查 ----------------
def _safe_check_nan_in_tensor(x: torch.Tensor, name: str, operation: str = "") -> torch.Tensor:
    """安全的NaN检查，不会导致训练崩溃"""    
    if torch.isnan(x).any():
        nan_count = torch.isnan(x).sum().item()
        logger.warning(f"⚠️ {name} 在 {operation} 后包含 {nan_count} 个NaN值")
        # 尝试修复：将NaN替换为0
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
    if torch.isinf(x).any():
        inf_count = torch.isinf(x).sum().item()
        logger.warning(f"⚠️ {name} 在 {operation} 后包含 {inf_count} 个Inf值")
        # 尝试修复：将Inf替换为有限大值
        x = torch.where(torch.isinf(x), torch.finfo(x.dtype).max * torch.ones_like(x), x)
        
    return x

def _debug_tensor_for_nan_inf_count(x: torch.Tensor, name: str):
    """调试张量信息"""
    stats = {
        "shape": x.shape,
        "min": x.min().item(),
        "max": x.max().item(), 
        "mean": x.mean().item() if x.dtype.is_floating_point else 0.0,
        "std": x.std(unbiased=False).item() if (x.dtype.is_floating_point and x.numel() >= 2) else 0.0,
        "nan_count": torch.isnan(x).sum().item(),
        "inf_count": torch.isinf(x).sum().item()
    }
    
    # 只有当有NaN或Inf时才输出警告
    if stats["nan_count"] > 0 or stats["inf_count"] > 0:
        msg = f"🔍🔍 {name}: " + " | ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                                        for k, v in stats.items()])
        logger.warning(msg)


def _check_input_dict_for_nan(batch: Dict[str, Any]):
    """更详细的输入数据检查"""        
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.isnan(value).any():
                nan_count = torch.isnan(value).sum().item()
                logger.error(f"🚨🚨🚨🚨 严重错误: 输入数据 {key} 包含 {nan_count} 个NaN值")
                raise ValueError(f"输入数据 {key} 包含NaN值，请检查数据预处理")
                
            if torch.isinf(value).any():
                inf_count = torch.isinf(value).sum().item()
                logger.error(f"🚨🚨🚨🚨 严重错误: 输入数据 {key} 包含 {inf_count} 个Inf值")
                raise ValueError(f"输入数据 {key} 包含Inf值，请检查数据预处理")
                
            # 只对数值类型（浮点数、整数）的张量进行统计计算，跳过布尔类型
            if value.numel() > 0 and value.dtype in [torch.float16, torch.float32, torch.float64, torch.int16, torch.int32, torch.int64]:
                try:
                    stats = {
                        'min': value.min().item(),
                        'max': value.max().item(),
                    }
                    
                    # 只有浮点类型才能计算mean和std
                    if value.dtype in [torch.float16, torch.float32, torch.float64]:
                        stats['mean'] = value.mean().item()
                        if value.numel() >= 2 and value.dtype.is_floating_point:
                            stats["std"] = value.std(unbiased=False).item()
                        else:
                            stats["std"] = 0.0
                    else:
                        # 对于整数类型，计算总和作为替代
                        stats['sum'] = value.sum().item()
                    
                    # logger.debug(f"输入数据 {key} 统计: {stats}")
                    
                except Exception as e:
                    # 如果统计计算失败，只记录基本信息
                    logger.debug(f"无法计算 {key} 的统计信息: {e}, dtype: {value.dtype}")


def _check_and_fix_numeric_tensor_for_nan_inf(name, tensor):
    if not torch.is_tensor(tensor):
        return False

    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    nan_cnt = nan_mask.sum().item()
    inf_cnt = inf_mask.sum().item()
    total = tensor.numel()

    if nan_cnt == 0 and inf_cnt == 0:
        return False

    ratio = (nan_cnt + inf_cnt) / total

    if ratio > 0.1:
        raise RuntimeError(
            f"[FATAL] numeric feature `{name}` NaN/Inf 比例过高 "
            f"({ratio:.2%}), shape={tuple(tensor.shape)}"
        )

    logger.warning(
        f"[FIX] `{name}` 含 NaN={nan_cnt}, Inf={inf_cnt}, "
        f"比例={ratio:.2%}，已置 0"
    )

    tensor[nan_mask | inf_mask] = 0.0
    return True