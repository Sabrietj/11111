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


def get_project_root(start_path: str | None = None) -> str:
    import os, subprocess

    if start_path is None:
        start_path = os.path.abspath(os.path.dirname(__file__))

    # ① 尝试通过 Git
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
        return root
    except Exception:
        pass

    # ② 尝试查找关键文件
    markers = ("pyproject.toml", "setup.py", "requirements.txt", ".git")
    cur = start_path
    while True:
        if any(os.path.exists(os.path.join(cur, m)) for m in markers):
            return cur
        parent = os.path.abspath(os.path.join(cur, os.pardir))
        if parent == cur:
            break
        cur = parent

    # ③ fallback：使用 VSCode 工作路径
    return os.environ.get("PWD", os.getcwd())