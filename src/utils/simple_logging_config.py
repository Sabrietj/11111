# -*- coding: utf-8 -*-
"""
简化日志颜色配置
避免外部依赖，使用原生 ANSI 颜色
"""

import logging
import sys
from pathlib import Path

class SimpleColoredFormatter(logging.Formatter):
    """简化版带颜色的日志格式化器"""
    
    # ANSI 颜色映射
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[31;1m' # 红色加粗
    }
    RESET = '\033[0m'
    
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.is_tty = sys.stdout.isatty()
    
    def format(self, record):
        if self.is_tty:
            # 只对级别名称着色，保持消息内容不变
            level_color = self.COLORS.get(record.levelname, '')
            level_name = record.levelname
            record.levelname = f"{level_color}{level_name}{self.RESET}"
        
        return super().format(record)

def setup_simple_logging(log_level=logging.INFO, log_file=None):
    """设置简化版日志配置"""
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 日志格式
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = SimpleColoredFormatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger