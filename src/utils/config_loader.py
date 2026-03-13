"""
配置加载器
支持YAML配置文件的继承、合并和验证
"""

import os
import yaml
import configparser
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import copy
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.loaded_configs = {}
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件，支持继承
        
        Args:
            config_path: 配置文件路径（相对于config_dir）
            
        Returns:
            配置字典
        """
        full_path = self.config_dir / config_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {full_path}")
        
        # 检查是否已经加载过
        if str(full_path) in self.loaded_configs:
            return copy.deepcopy(self.loaded_configs[str(full_path)])
        
        # 加载配置文件
        with open(full_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # 处理继承
        if '_base_' in config:
            base_configs = config['_base_']
            if isinstance(base_configs, str):
                base_configs = [base_configs]
            
            # 递归加载基础配置
            base_config = {}
            for base_path in base_configs:
                base_config.update(self.load_config(base_path))
            
            # 合并配置（当前配置覆盖基础配置）
            config = self._merge_configs(base_config, config)
            # 删除_base_字段
            config.pop('_base_', None)
        
        # 缓存配置
        self.loaded_configs[str(full_path)] = copy.deepcopy(config)
        
        logger.info(f"加载配置文件: {full_path}")
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        递归合并配置
        
        Args:
            base: 基础配置
            override: 覆盖配置
            
        Returns:
            合并后的配置
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                # 递归合并字典
                result[key] = self._merge_configs(result[key], value)
            else:
                # 直接覆盖
                result[key] = value
        
        return result
    
    def list_configs(self, pattern: str = "**/*.yaml") -> List[Path]:
        """
        列出所有配置文件
        
        Args:
            pattern: 文件匹配模式
            
        Returns:
            配置文件路径列表
        """
        config_files = list(self.config_dir.glob(pattern))
        return sorted(config_files)
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """
        保存配置文件
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        full_path = self.config_dir / config_path
        
        # 确保目录存在
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
        
        logger.info(f"配置已保存到: {full_path}")

# 创建全局配置加载器实例
config_loader = ConfigLoader()

def load_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """
    加载实验配置
    
    Args:
        experiment_name: 实验名称
        
    Returns:
        实验配置字典
    """
    if not experiment_name.endswith('.yaml'):
        experiment_name = f"experiments/{experiment_name}.yaml"
    
    return config_loader.load_config(experiment_name)

def load_base_config(config_type: str) -> Dict[str, Any]:
    """
    加载基础配置
    
    Args:
        config_type: 配置类型
        
    Returns:
        基础配置字典
    """
    config_path = f"base/{config_type}.yaml"
    return config_loader.load_config(config_path)

def load_loss_config(loss_type: str) -> Dict[str, Any]:
    """
    加载损失函数配置
    
    Args:
        loss_type: 损失函数类型
        
    Returns:
        损失函数配置字典
    """
    config_path = f"loss_functions/{loss_type}.yaml"
    return config_loader.load_config(config_path)

def load_config_with_dataset_switch(config_path: str) -> configparser.ConfigParser:
    """
    加载INI配置文件并处理数据集自动切换
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        处理后的ConfigParser对象
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    # 获取激活的数据集名称
    if 'GENERAL' in config and 'ACTIVE_DATASET' in config['GENERAL']:
        active_dataset = config['GENERAL']['ACTIVE_DATASET']
        
        # 检查是否存在对应的数据集配置段
        if active_dataset in config:
            dataset_config = config[active_dataset]
            
            # 自动填充PATH段
            if 'PATH' in config:
                for key in ['path_to_dataset', 'plot_data_path']:
                    if key in dataset_config:
                        config['PATH'][key] = dataset_config[key]
            
            # 自动填充SESSION段
            if 'SESSION' in config:
                for key in ['session_tuple_mode', 'session_label_id_map', 
                          'concurrent_flow_iat_threshold', 'sequential_flow_iat_threshold']:
                    if key in dataset_config and dataset_config[key] != 'AUTO_FILL':
                        config['SESSION'][key] = dataset_config[key]
            
            logger.info(f"已激活数据集配置: {active_dataset}")
        else:
            logger.warning(f"未找到数据集配置段: {active_dataset}")
    else:
        logger.warning("未配置ACTIVE_DATASET，使用默认配置")
    
    return config

def get_active_dataset_config(config_path: str) -> Dict[str, Any]:
    """
    获取当前激活的数据集配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        数据集配置字典
    """
    config = load_config_with_dataset_switch(config_path)
    
    # 从GENERAL段获取激活的数据集名称
    if 'GENERAL' in config and 'ACTIVE_DATASET' in config['GENERAL']:
        active_dataset = config['GENERAL']['ACTIVE_DATASET']
        
        if active_dataset in config:
            return dict(config[active_dataset])
    
    return {}