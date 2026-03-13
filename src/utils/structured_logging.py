# -*- coding: utf-8 -*-
"""
结构化日志记录
支持 JSON 格式和上下文信息
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional

class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def __init__(self, mode='text', include_context=True):
        """
        Args:
            mode: 'text' 或 'json'
            include_context: 是否包含上下文信息
        """
        super().__init__()
        self.mode = mode
        self.include_context = include_context
    
    def format(self, record):
        if self.mode == 'json':
            return self.format_json(record)
        else:
            return self.format_text(record)
    
    def format_json(self, record):
        """JSON 格式日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        if self.include_context and hasattr(record, 'context'):
            log_entry['context'] = record.context
        
        return json.dumps(log_entry, ensure_ascii=False)
    
    def format_text(self, record):
        """文本格式日志（带颜色）"""
        # ANSI 颜色代码
        colors = {
            'DEBUG': '\033[36m',
            'INFO': '\033[32m', 
            'WARNING': '\033[33m',
            'ERROR': '\033[31m',
            'CRITICAL': '\033[31;1m'
        }
        reset = '\033[0m'
        
        level_color = colors.get(record.levelname, '')
        
        # 基础格式
        base_format = f"[{record.asctime}] [{level_color}{record.levelname}{reset}] {record.module}:{record.funcName}:{record.lineno} - {record.getMessage()}"
        
        # 添加上下文信息
        if self.include_context and hasattr(record, 'context'):
            context_str = ' '.join([f"{k}={v}" for k, v in record.context.items()])
            base_format += f" | {context_str}"
        
        return base_format

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name, level=logging.INFO, mode='text', log_file=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 清除现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 设置格式化器
        formatter = StructuredFormatter(mode=mode)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（可选）
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message, context=None):
        self._log_with_context(logging.DEBUG, message, context)
    
    def info(self, message, context=None):
        self._log_with_context(logging.INFO, message, context)
    
    def warning(self, message, context=None):
        self._log_with_context(logging.WARNING, message, context)
    
    def error(self, message, context=None):
        self._log_with_context(logging.ERROR, message, context)
    
    def critical(self, message, context=None):
        self._log_with_context(logging.CRITICAL, message, context)
    
    def _log_with_context(self, level, message, context):
        """带上下文信息的日志记录"""
        extra = {'context': context} if context else {}
        self.logger.log(level, message, extra=extra)

# 快速使用函数
def get_structured_logger(name, level=logging.INFO, mode='text', log_file=None):
    """获取结构化日志记录器"""
    return StructuredLogger(name, level, mode, log_file)

# 示例用法
if __name__ == "__main__":
    logger = get_structured_logger("test", mode='text')
    
    logger.info("系统启动", {"version": "1.0.0", "env": "production"})
    logger.debug("调试信息", {"process_id": 1234})
    logger.warning("警告信息", {"threshold": 0.8})
    logger.error("错误信息", {"error_code": "E001"})