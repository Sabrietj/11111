"""
工具类模块
提供消融实验的通用工具函数
"""

import os
import sys
import yaml
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import hashlib

# 使用统一的日志配置
try:
    from utils.logging_config import setup_preset_logging
except ImportError:
    import logging
    def setup_preset_logging(log_level):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

import logging
logger = setup_preset_logging(log_level=logging.DEBUG)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


# 通用配置到模型配置的映射表
GENERAL_CONFIG_MAPPING = {}

def load_config_mapping(mapping_file_path: str = None) -> Dict[str, Any]:
    """
    从YAML文件加载配置映射表
    
    Args:
        mapping_file_path: 映射文件路径，默认使用当前目录下的config_mapping.yaml
        
    Returns:
        映射表字典
    """
    if mapping_file_path is None:
        # 默认映射文件路径
        current_dir = Path(__file__).parent
        mapping_file_path = current_dir / "config_mapping.yaml"
    
    logger.info(f"加载配置映射表: {mapping_file_path}")
    
    try:
        if Path(mapping_file_path).exists():
            with open(mapping_file_path, 'r', encoding='utf-8') as f:
                mapping_data = yaml.safe_load(f)
            
            # 简化的扁平化映射，只处理我们需要的配置项
            flat_mapping = {}
            
            def process_section(section_name: str, section_data: Dict[str, Any]):
                for key, value in section_data.items():
                    if isinstance(value, dict) and ('default' in value or any(k in ['flow_bert_multiview', 'flow_bert_multiview_ssl', 'flow_autoencoder'] for k in value.keys())):
                        flat_mapping[f"{section_name}.{key}"] = value
            
            # 处理各个配置段
            if 'data' in mapping_data:
                process_section('data', mapping_data['data'])
            
            if 'training' in mapping_data:
                process_section('training', mapping_data['training'])
            
            if 'model' in mapping_data:
                process_section('model', mapping_data['model'])
            
            global GENERAL_CONFIG_MAPPING
            GENERAL_CONFIG_MAPPING = flat_mapping
            logger.info(f"成功加载配置映射表: {mapping_file_path}")
            logger.debug(f"映射表内容: {GENERAL_CONFIG_MAPPING}")
            
        else:
            logger.warning(f"配置映射文件不存在: {mapping_file_path}，使用空映射表")
            
    except Exception as e:
        logger.error(f"加载配置映射表失败: {e}")
        logger.warning("使用空的配置映射表")


def get_model_config_mapping(exp_config_path: str, model_name: str) -> str:
    """
    根据模型名称获取对应的配置路径
    
    Args:
        exp_config_path: 实验配置中的路径（如 'data.data_path'）
        model_name: 模型名称
        
    Returns:
        模型配置中的对应路径
    """
    if exp_config_path not in GENERAL_CONFIG_MAPPING:
        logger.warning(f"未找到配置映射: {exp_config_path}")
        return exp_config_path
    
    mapping = GENERAL_CONFIG_MAPPING[exp_config_path]
    
    # 优先返回模型特定的映射
    if model_name in mapping:
        return mapping[model_name]
    
    # 其次返回默认映射
    if 'default' in mapping:
        return mapping['default']
    
    # 最后返回原始路径
    logger.warning(f"模型 {model_name} 在 {exp_config_path} 中无映射，使用原始路径")
    return exp_config_path


def extract_general_config(exp_config: Dict[str, Any], model_name: str = None) -> Dict[str, Any]:
    """
    从实验配置中提取通用配置参数，并转换为模型配置格式
    
    Args:
        exp_config: 完整的实验配置字典
        model_name: 模型名称，用于配置映射
        
    Returns:
        通用配置字典，包含可直接传递给训练脚本的参数
    """
    general_config = {}

    logger.info(f"提取通用配置（{model_name}): {exp_config}")
    
    # 数据相关配置
    if 'data' in exp_config:
        data_config = exp_config['data']
        
        # 数据路径映射
        if 'data_path' in data_config and model_name:
            model_path = get_model_config_mapping('data.data_path', model_name)
            general_config[model_path] = data_config['data_path']
            logger.info(f"数据路径映射: data.data_path -> {model_path} = {data_config['data_path']}")
        
        # 会话划分文件路径映射
        if 'session_split_path' in data_config and model_name:
            model_path = get_model_config_mapping('data.session_split_path', model_name)
            general_config[model_path] = data_config['session_split_path']
            logger.info(f"会话划分路径映射: data.session_split_path -> {model_path} = {data_config['session_split_path']}")
        
        # 批量大小和工作者（通用配置）
        if 'batch_size' in data_config:
            model_path = get_model_config_mapping('data.batch_size', model_name)
            general_config[model_path] = data_config['batch_size']
            
        if 'num_workers' in data_config:
            model_path = get_model_config_mapping('data.num_workers', model_name)
            general_config[model_path] = data_config['num_workers']
    
    # 训练相关配置
    if 'training' in exp_config:
        training_config = exp_config['training']
        
        # 最大训练轮数
        if 'max_epochs' in training_config:
            model_path = get_model_config_mapping('training.max_epochs', model_name)
            general_config[model_path] = training_config['max_epochs']
            logger.info(f"最大轮数映射: training.max_epochs -> {model_path} = {training_config['max_epochs']}")
        
        # 耐心参数（需要映射）
        if 'patience' in training_config and model_name:
            model_path = get_model_config_mapping('training.patience', model_name)
            general_config[model_path] = training_config['patience']
            logger.info(f"耐心参数映射: training.patience -> {model_path} = {training_config['patience']}")
        
        # 可选的学习率配置
        if 'learning_rate' in training_config:
            model_path = get_model_config_mapping('training.learning_rate', model_name)
            general_config[model_path] = training_config['learning_rate']
            
        if 'weight_decay' in training_config:
            model_path = get_model_config_mapping('training.weight_decay', model_name)
            general_config[model_path] = training_config['weight_decay']
    
    logger.info(f"提取到通用配置（{model_name}）: {general_config}")
    return general_config


def convert_to_hydra_overrides(general_config: Dict[str, Any]) -> List[str]:
    """
    将通用配置转换为Hydra覆盖参数格式
    
    Args:
        general_config: 通用配置字典
        
    Returns:
        Hydra覆盖参数列表
    """
    overrides = []
    
    for param_path, value in general_config.items():
        # 转换为Hydra格式: param_path=value
        overrides.append(f"{param_path}={value}")
    
    logger.info(f"生成的Hydra覆盖参数: {overrides}")
    return overrides


def add_config_mapping(exp_config_path: str, model_mappings: Dict[str, str], default_mapping: str = None):
    """
    动态添加配置映射（用于运行时扩展）
    
    Args:
        exp_config_path: 实验配置路径（如 'data.data_path'）
        model_mappings: 模型特定映射字典
        default_mapping: 默认映射路径
        
    Example:
        add_config_mapping(
            'data.custom_param',
            {'model_a': 'model_a.config.custom_param'},
            'general.custom_param'
        )
    """
    global GENERAL_CONFIG_MAPPING
    
    mapping_entry = model_mappings.copy()
    if default_mapping:
        mapping_entry['default'] = default_mapping
    
    GENERAL_CONFIG_MAPPING[exp_config_path] = mapping_entry
    logger.info(f"添加新配置映射: {exp_config_path} -> {mapping_entry}")


def print_available_mappings():
    """
    打印所有可用的配置映射（用于调试）
    """
    logger.info("=== 可用的配置映射 ===")
    for exp_path, mappings in GENERAL_CONFIG_MAPPING.items():
        logger.info(f"{exp_path}:")
        for model, target in mappings.items():
            logger.info(f"  {model}: {target}")
    logger.info("=================")


def save_yaml_config(config: Dict[str, Any], config_path: str):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    try:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"Successfully saved config to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        raise


def load_json_file(json_path: str) -> Dict[str, Any]:
    """
    加载JSON文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        数据字典
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {json_path}: {e}")
        raise


def save_json_file(data: Dict[str, Any], json_path: str, indent: int = 2):
    """
    保存数据到JSON文件
    
    Args:
        data: 数据字典
        json_path: JSON文件路径
        indent: 缩进空格数
    """
    try:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Failed to save JSON to {json_path}: {e}")
        raise


def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    验证配置模式
    
    Args:
        config: 要验证的配置
        schema: 模式定义
        
    Returns:
        错误列表
    """
    errors = []
    
    def _validate_recursive(data: Dict, schema_part: Dict, path: str = ""):
        for key, expected_type in schema_part.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in data:
                if isinstance(expected_type, dict) and 'required' in expected_type:
                    if expected_type['required']:
                        errors.append(f"Missing required field: {current_path}")
                continue
            
            value = data[key]
            
            if isinstance(expected_type, dict):
                # 嵌套对象验证
                if 'type' in expected_type:
                    expected_type_name = expected_type['type']
                    if not _check_type(value, expected_type_name):
                        errors.append(f"Type mismatch at {current_path}: expected {expected_type_name}, got {type(value).__name__}")
                
                if 'properties' in expected_type:
                    _validate_recursive(value, expected_type['properties'], current_path)
            else:
                # 简单类型验证
                if not _check_type(value, expected_type):
                    errors.append(f"Type mismatch at {current_path}: expected {expected_type.__name__}, got {type(value).__name__}")
    
    def _check_type(value, expected_type):
        """检查类型匹配"""
        if expected_type == str:
            return isinstance(value, str)
        elif expected_type == int:
            return isinstance(value, int)
        elif expected_type == float:
            return isinstance(value, (int, float))
        elif expected_type == bool:
            return isinstance(value, bool)
        elif expected_type == list:
            return isinstance(value, list)
        elif expected_type == dict:
            return isinstance(value, dict)
        else:
            return True
    
    _validate_recursive(config, schema)
    return errors


def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """
    计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法 (md5, sha1, sha256)
        
    Returns:
        文件哈希值
    """
    hash_func = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        raise


def create_directory_with_permissions(directory_path: str, mode: int = 0o755):
    """
    创建目录并设置权限
    
    Args:
        directory_path: 目录路径
        mode: 权限模式
    """
    try:
        dir_path = Path(directory_path)
        dir_path.mkdir(parents=True, exist_ok=True, mode=mode)
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise


def check_dependencies(required_packages: List[str]) -> Dict[str, bool]:
    """
    检查依赖包是否可用
    
    Args:
        required_packages: 必需的包列表
        
    Returns:
        包可用性字典
    """
    dependencies = {}
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
            logger.warning(f"Missing dependency: {package}")
    
    return dependencies


def format_duration(seconds: float) -> str:
    """
    格式化时间长度
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
        
    Returns:
        格式化的文件大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_experiment_config_schema() -> Dict[str, Any]:
    """
    获取实验配置模式定义
    
    Returns:
        配置模式字典
    """
    return {
        "data": {
            "type": "object",
            "required": True,
            "properties": {
                "data_path": {"type": str, "required": True},
                "batch_size": {"type": int, "required": True},
                "num_workers": {"type": int, "required": True}
            }
        },
        "training": {
            "type": "object",
            "required": True,
            "properties": {
                "max_epochs": {"type": int, "required": True},
                "patience": {"type": int, "required": True}
            }
        },
        "experiment": {
            "type": "object",
            "required": True,
            "properties": {
                "model_name": {"type": str, "required": True},
                "type": {"type": str, "required": True},
                "enable": {"type": dict, "required": True}
            }
        },
        "ablation_variants": {
            "type": list,
            "required": True,
            "properties": {
                # "ablation_id": {"type": str, "required": True},
                "name": {"type": str, "required": True},
                "description": {"type": str, "required": True},
                "type": {"type": str, "required": True},
                "section": {"type": str, "required": True},
                "config": {"type": dict, "required": True},
                "model_name": {"type": str, "required": False},
                "baseline": {"type": bool, "required": False}
            }
        }
    }


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置字典
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    import copy
    merged = copy.deepcopy(base_config)
    
    def _merge_recursive(base: Dict, override: Dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _merge_recursive(base[key], value)
            else:
                base[key] = value
    
    _merge_recursive(merged, override_config)
    return merged


def ensure_path_exists(path: Union[str, Path]):
    """
    确保路径存在
    
    Args:
        path: 文件或目录路径
    """
    path = Path(path)
    if path.suffix:  # 文件路径
        path.parent.mkdir(parents=True, exist_ok=True)
    else:  # 目录路径
        path.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """
    获取当前时间戳字符串
    
    Returns:
        时间戳字符串
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除非法字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        清理后的文件名
    """
    import re
    # 移除或替换非法字符
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除多余的空格和点
    sanitized = re.sub(r'\s+', '_', sanitized.strip(' .'))
    # 限制长度
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized


def get_available_memory() -> float:
    """
    获取可用内存（GB）
    
    Returns:
        可用内存大小
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)
    except ImportError:
        logger.warning("psutil not available, cannot get memory info")
        return 0.0


def check_disk_space(path: str, min_space_gb: float = 1.0) -> bool:
    """
    检查磁盘空间是否足够
    
    Args:
        path: 路径
        min_space_gb: 最小所需空间（GB）
        
    Returns:
        是否有足够空间
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        return free_gb >= min_space_gb
    except Exception as e:
        logger.warning(f"Failed to check disk space: {e}")
        return True


class ExperimentTimer:
    """实验计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.checkpoints = {}
    
    def start(self):
        """开始计时"""
        self.start_time = datetime.now()
        logger.debug(f"Timer started at {self.start_time}")
    
    def stop(self):
        """停止计时"""
        self.end_time = datetime.now()
        logger.debug(f"Timer stopped at {self.end_time}")
    
    def checkpoint(self, name: str):
        """记录检查点"""
        if self.start_time:
            self.checkpoints[name] = datetime.now()
            logger.debug(f"Checkpoint '{name}' recorded at {self.checkpoints[name]}")
    
    def get_duration(self) -> Optional[float]:
        """获取总持续时间（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def get_checkpoint_duration(self, checkpoint_name: str) -> Optional[float]:
        """获取从开始到指定检查点的持续时间"""
        if self.start_time and checkpoint_name in self.checkpoints:
            return (self.checkpoints[checkpoint_name] - self.start_time).total_seconds()
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """获取计时摘要"""
        summary = {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration': self.get_duration(),
            'checkpoints': {}
        }
        
        if self.start_time:
            for name, checkpoint_time in self.checkpoints.items():
                summary['checkpoints'][name] = {
                    'time': checkpoint_time.isoformat(),
                    'duration_from_start': (checkpoint_time - self.start_time).total_seconds()
                }
        
        return summary


# 在模块导入时自动加载配置映射表
load_config_mapping()