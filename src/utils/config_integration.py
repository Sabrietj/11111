"""
配置兼容性包装器
为现有代码提供透明的数据集切换功能
"""

import configparser
import os
from pathlib import Path
from typing import Dict, Any

from config_loader import load_config_with_dataset_switch

def get_config_parser(config_file: str = "config.cfg") -> configparser.ConfigParser:
    """
    获取处理后的配置解析器（兼容现有API）
    
    Args:
        config_file: 配置文件名
        
    Returns:
        ConfigParser对象
    """
    # 获取当前文件的目录，然后构建配置文件路径
    current_dir = Path(__file__).parent
    config_path = current_dir / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    return load_config_with_dataset_switch(str(config_path))

def get_config_section(section: str, config_file: str = "config.cfg") -> Dict[str, Any]:
    """
    获取配置段（兼容现有API）
    
    Args:
        section: 配置段名称
        config_file: 配置文件名
        
    Returns:
        配置段字典
    """
    config = get_config_parser(config_file)
    
    if section in config:
        return dict(config[section])
    else:
        raise KeyError(f"配置段不存在: {section}")

# 全局配置缓存
_cached_config = None

def get_global_config() -> configparser.ConfigParser:
    """
    获取全局配置（单例模式，提高性能）
    
    Returns:
        ConfigParser对象
    """
    global _cached_config
    if _cached_config is None:
        _cached_config = get_config_parser()
    return _cached_config

def refresh_config():
    """
    刷新配置缓存（当配置文件被修改时调用）
    """
    global _cached_config
    _cached_config = None