# -*- coding: utf-8 -*-
"""
统一日志配置
提供预设的日志配置
"""

import logging
import sys
from pathlib import Path

from typing import Dict, Any

# 使用原生 ANSI 颜色代码，避免外部依赖
import sys

# 检查是否为终端输出
IS_TTY = sys.stdout.isatty()

# 定义 ANSI 颜色代码
class Fore:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

class Back:
    BLACK = '\033[40m'
    RED = '\033[41m'
    GREEN = '\033[42m'
    YELLOW = '\033[43m'
    BLUE = '\033[44m'
    MAGENTA = '\033[45m'
    CYAN = '\033[46m'
    WHITE = '\033[47m'
    RESET = '\033[0m'

class Style:
    RESET_ALL = '\033[0m'
    BRIGHT = '\033[1m'
    DIM = '\033[2m'
    NORMAL = '\033[22m'


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # 颜色映射
    COLOR_MAP = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Back.WHITE
    }
    
    def __init__(self, fmt=None, datefmt=None, style='%', colored=True):
        super().__init__(fmt, datefmt, style)
        self.colored = colored
    
    def format(self, record):
        if self.colored and IS_TTY:
            # 根据日志级别设置颜色
            color = self.COLOR_MAP.get(record.levelno, '')
            reset = Style.RESET_ALL
            
            # 修改日志级别名称和消息格式，添加颜色
            record.levelname = f"{color}{record.levelname}{reset}"
            record.msg = f"{color}{record.msg}{reset}"
        
        return super().format(record)
# 默认配置
DEFAULT_LOG_LEVEL = logging.DEBUG
DEFAULT_LOG_FORMAT = '[%(asctime)s %(filename)s,%(lineno)d] [%(levelname)s] %(message)s'

def setup_preset_logging(log_level: int = logging.INFO, 
                        log_format: str = None,
                        log_file: str = None) -> logging.Logger:
    """
    设置预设的日志配置
    
    Args:
        log_level: 日志级别
        log_format: 日志格式
        log_file: 日志文件路径
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT
    
    # 创建根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 控制框架内部日志级别，减少调试信息
    logging.getLogger('hydra').setLevel(logging.ERROR)
    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # 控制更底层的调试信息
    logging.getLogger('hydra._internal').setLevel(logging.ERROR)
    logging.getLogger('hydra._internal.utils').setLevel(logging.ERROR)
    logging.getLogger('hydra.utils').setLevel(logging.ERROR)
    logging.getLogger('omegaconf').setLevel(logging.ERROR)
    
    # 控制JobRuntime相关日志
    logging.getLogger('joblib').setLevel(logging.WARNING)
    logging.getLogger('distributed').setLevel(logging.WARNING)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 创建带颜色的格式化器
    formatter = ColoredFormatter(log_format, colored=True)
    console_handler.setFormatter(formatter)
    
    # 添加控制台处理器
    root_logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        root_logger.addHandler(file_handler)
        
        # 记录日志文件位置
        root_logger.info(f"日志文件保存到: {log_file}")
    
    # 记录日志配置完成
    # root_logger.info(msg=f"日志配置完成 - 级别: {logging.getLevelName(log_level)}")
    
    return root_logger


def setup_experiment_logging(experiment_name: str, 
                           log_level: int = logging.DEBUG,
                           log_dir: str = "logs") -> logging.Logger:
    """
    设置实验专用的日志配置
    
    Args:
        experiment_name: 实验名称
        log_level: 日志级别
        log_dir: 日志目录
        
    Returns:
        logging.Logger: 实验专用日志器
    """
    
    # 创建实验专用日志器
    experiment_logger = logging.getLogger(experiment_name)
    experiment_logger.setLevel(log_level)
    
    # 如果实验日志器已经有处理器，直接返回
    if experiment_logger.handlers:
        return experiment_logger
    
    # 日志格式
    log_format = f'[%(asctime)s-{experiment_name},%(lineno)d] %(message)s'
    
    # 创建日志文件路径
    log_path = Path(log_dir) / f"{experiment_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # 创建文件格式化器（普通格式）
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    
    # 添加文件处理器
    experiment_logger.addHandler(file_handler)
    
    # 同时添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 创建带颜色的格式化器
    console_formatter = ColoredFormatter(log_format, colored=True)
    console_handler.setFormatter(console_formatter)
    
    experiment_logger.addHandler(console_handler)
    
    # 记录实验日志设置
    experiment_logger.info(f"实验日志配置完成 - 文件: {log_path}")
    
    return experiment_logger


def setup_module_logging(module_name: str, 
                        log_level: int = logging.DEBUG) -> logging.Logger:
    """
    设置模块专用的日志配置
    
    Args:
        module_name: 模块名称
        log_level: 日志级别
        
    Returns:
        logging.Logger: 模块专用日志器
    """
    
    # 创建模块专用日志器
    module_logger = logging.getLogger(module_name)
    module_logger.setLevel(log_level)
    
    # 如果模块日志器已经有处理器，直接返回
    if module_logger.handlers:
        return module_logger
    
    # 日志格式
    log_format: str = f'[%(asctime)s-{module_name},%(lineno)d] %(message)s'
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 创建带颜色的格式化器
    formatter = ColoredFormatter(log_format, colored=True)
    console_handler.setFormatter(formatter)
    
    # 添加控制台处理器
    module_logger.addHandler(console_handler)
    
    return module_logger


def get_logging_config_summary(logger: logging.Logger) -> Dict[str, Any]:
    """
    获取日志配置摘要
    
    Args:
        logger: 日志器
        
    Returns:
        Dict[str, Any]: 日志配置摘要
    """
    
    summary = {
        'logger_name': logger.name,
        'log_level': logging.getLevelName(logger.level),
        'handlers': []
    }
    
    for handler in logger.handlers:
        handler_info = {
            'handler_type': type(handler).__name__,
            'handler_level': logging.getLevelName(handler.level)
        }
        
        if hasattr(handler, 'baseFilename'):
            handler_info['log_file'] = handler.baseFilename
        
        summary['handlers'].append(handler_info)
    
    return summary


def validate_logging_configuration(logger: logging.Logger) -> bool:
    """
    验证日志配置
    
    Args:
        logger: 日志器
        
    Returns:
        bool: 配置是否有效
    """
    
    # 检查是否有处理器
    if not logger.handlers:
        print("警告: 日志器没有处理器")
        return False
    
    # 检查日志级别
    if logger.level == logging.NOTSET:
        print("警告: 日志级别未设置")
        return False
    
    # 检查处理器级别
    for handler in logger.handlers:
        if handler.level == logging.NOTSET:
            print(f"警告: 处理器 {type(handler).__name__} 级别未设置")
            return False
    
    return True




# 预配置的日志器
PRESET_LOGGERS = {
    'experiment': setup_experiment_logging,
    'module': setup_module_logging,
    'root': setup_preset_logging
}