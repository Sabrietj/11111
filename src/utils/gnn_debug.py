"""
统一配置管理版本 - 从 config.cfg 读取日志配置
"""

import os
import logging
import inspect
import configparser
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


def find_config_file() -> str:
    """查找配置文件"""
    # 在当前目录和上级目录查找 config.cfg
    current_dir = Path(__file__).parent
    possible_paths = [
        current_dir / "config.cfg",
        current_dir.parent / "config.cfg",
        current_dir.parent.parent / "config.cfg"
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            return str(config_path)
    
    # 如果找不到配置文件，返回默认路径
    return str(current_dir / "config.cfg")


def load_log_config(config_path: str | None = None) -> Dict[str, Any]:
    """从config.cfg加载日志配置"""
    if config_path is None:
        config_path = find_config_file()
    
    config = configparser.ConfigParser()
    
    # 设置默认配置
    default_config = {
        'log_level': 'DEBUG',
        'log_format': '[%(asctime)s.%(msecs)03d] [%(levelname)s][%(filename)s:%(lineno)d] - %(message)s',
        'log_to_console': True,
        'log_to_file': True,
        'log_file_name': 'debug.log',
        'log_dir': 'logs',
        'max_file_size': 10485760,  # 10MB
        'backup_count': 5,
        'enable_log_rotation': True,
        'enable_timestamp_naming': True,      # 新增：启用时间戳命名
        'timestamp_format': '%Y%m%d_%H%M%S',  # 新增：时间戳格式
        'keep_daily_logs': 7                   # 新增：保留最近7天的日志
    }
    
    # 尝试读取配置文件
    try:
        # 禁用插值以避免 % 字符被误解析
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path, encoding='utf-8')
        
        # 获取配置值，如果不存在则使用默认值
        log_config = {}
        for key, default_value in default_config.items():
            if config.has_option('LOG', key):
                if key in ['log_to_console', 'log_to_file', 'enable_log_rotation']:
                    try:
                        log_config[key] = config.getboolean('LOG', key)
                    except ValueError:
                        log_config[key] = default_value
                elif key in ['max_file_size', 'backup_count']:
                    try:
                        log_config[key] = config.getint('LOG', key)
                    except ValueError:
                        log_config[key] = default_value
                else:
                    value = config.get('LOG', key)
                    # 处理转义的 % 字符
                    if key == 'log_format':
                        value = value.replace('%%', '%')
                    log_config[key] = value
            else:
                log_config[key] = default_value
        
        return log_config
        
    except Exception as e:
        # 如果读取配置文件失败，使用默认配置
        print(f"警告: 无法读取配置文件 {config_path}, 使用默认配置: {e}")
        return default_config


# ===========================================================================
# GNN 统一日志系统 - 重构版本，从 config.cfg 读取配置
# ===========================================================================

import logging.handlers
import sys
import shutil
from datetime import datetime  # 确保datetime模块被导入


class TimestampedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """支持时间戳命名的日志轮转处理器"""
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False, 
                 timestamp_format='%Y%m%d_%H%M%S', enable_timestamp=True):
        
        self.timestamp_format = timestamp_format
        self.enable_timestamp = enable_timestamp
        self.original_filename = filename
        
        # 生成带时间戳的文件名
        if self.enable_timestamp:
            filename = self._generate_timestamped_filename()
        
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
    
    def _generate_timestamped_filename(self):
        """生成带时间戳的文件名"""
        timestamp = datetime.now().strftime(self.timestamp_format)
        base_name, ext = os.path.splitext(self.original_filename)
        return f"{base_name}_{timestamp}{ext}"
    
    def doRollover(self):
        """重写轮转方法，支持时间戳命名"""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if self.backupCount > 0:
            # 清理旧日志文件
            self._cleanup_old_logs()
            
            # 生成新的带时间戳的文件名
            if self.enable_timestamp:
                new_filename = self._generate_timestamped_filename()
                
                # 如果当前文件存在，则轮转它
                if os.path.exists(self.baseFilename):
                    for i in range(self.backupCount - 1, 0, -1):
                        sfn = f"{self.baseFilename}.{i}"
                        dfn = f"{self.baseFilename}.{i+1}"
                        if os.path.exists(sfn):
                            if os.path.exists(dfn):
                                os.remove(dfn)
                            os.rename(sfn, dfn)
                    
                    # 将当前文件重命名为.1
                    dfn = f"{self.baseFilename}.1"
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(self.baseFilename, dfn)
                    
                    # 更新基础文件名
                    self.baseFilename = new_filename
        
        # 创建新的日志文件
        if not self.delay:
            self.stream = self._open()
    
    def _cleanup_old_logs(self):
        """清理过期日志文件"""
        try:
            log_dir = os.path.dirname(self.baseFilename)
            if not os.path.exists(log_dir):
                return
                
            # 获取所有日志文件
            base_name = os.path.basename(self.original_filename).split('.')[0]
            log_files = []
            
            for file in os.listdir(log_dir):
                if file.startswith(base_name) and file.endswith('.log'):
                    file_path = os.path.join(log_dir, file)
                    log_files.append((file_path, os.path.getmtime(file_path)))
            
            # 按修改时间排序，保留最新的文件
            log_files.sort(key=lambda x: x[1], reverse=True)
            
            # 保留最近的文件，删除其他
            for i, (file_path, _) in enumerate(log_files[self.backupCount:], start=self.backupCount):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
                    
        except Exception:
            # 清理失败不影响主流程
            pass


class GNLogger:
    """统一的GNN日志管理器"""
    
    _instance: 'GNLogger | None' = None
    _initialized: bool = False
    logger: 'logging.Logger'
    log_config: 'Dict[str, Any]'
    
    def __new__(cls, config_path: str | None = None):
        if cls._instance is None:
            cls._instance = super(GNLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str | None = None):
        if not self._initialized:
            self.logger = logging.getLogger("")
            self.log_config = load_log_config(config_path)
            self._setup_logger()
            self._initialized = True
    
    def _setup_logger(self):
        """设置日志器"""
        # logger已经在__init__中初始化
        self.logger.setLevel(logging.DEBUG)
        # 禁止传播到root logger，避免重复输出
        self.logger.propagate = False
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 设置日志格式
        formatter = logging.Formatter(
            fmt=self.log_config['log_format'],
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 控制台输出
        if self.log_config['log_to_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._get_log_level())
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件输出 - 使用新的时间戳命名处理器
        if self.log_config['log_to_file']:
            log_file = Path(self.log_config['log_dir']) / self.log_config['log_file_name']
            # 创建日志目录（如果不存在）
            log_dir = log_file.parent
            os.makedirs(log_dir, exist_ok=True)
            
            # 获取时间戳命名配置
            enable_timestamp = self.log_config.get('enable_timestamp_naming', True)
            timestamp_format = self.log_config.get('timestamp_format', '%Y%m%d_%H%M%S')
            
            if self.log_config['enable_log_rotation']:
                # 使用带时间戳的轮转处理器
                file_handler = TimestampedRotatingFileHandler(
                    log_file,
                    maxBytes=self.log_config['max_file_size'],
                    backupCount=self.log_config['backup_count'],
                    encoding='utf-8',
                    enable_timestamp=enable_timestamp,
                    timestamp_format=timestamp_format
                )
            else:
                # 不使用轮转，但支持时间戳命名
                if enable_timestamp:
                    # 生成带时间戳的文件名
                    timestamp = datetime.now().strftime(timestamp_format)
                    base_name, ext = os.path.splitext(log_file)
                    timestamped_file = f"{base_name}_{timestamp}{ext}"
                    log_file = Path(timestamped_file)
                
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
            
            file_handler.setLevel(self._get_log_level())
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # 记录使用的日志文件名
            if enable_timestamp:
                actual_filename = file_handler.baseFilename if hasattr(file_handler, 'baseFilename') else log_file
                self.logger.info(f"日志文件已创建: {actual_filename}")
    
    def _get_log_level(self) -> int:
        """获取日志级别"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return level_map.get(self.log_config['log_level'].upper(), logging.DEBUG)
    
    def _get_caller_info(self):
        """获取调用者信息（文件名和行号）"""
        # 获取调用栈，跳过GNLogger内部调用
        stack = inspect.stack()
        for i, frame_info in enumerate(stack):
            # 跳过utils.py文件本身和logging模块的调用
            if 'gnn_debug.py' not in frame_info.filename and 'logging' not in frame_info.filename:
                # 获取相对路径的文件名
                filename = os.path.basename(frame_info.filename)
                return filename, frame_info.lineno
        # 如果找不到调用者，返回默认值
        return 'unknown', 0
    
    def debug(self, message: str, end: str = '\n', flush: bool = False):
        """DEBUG级别日志"""
        if self._get_log_level() <= logging.DEBUG:
            filename, lineno = self._get_caller_info()
            # 处理end参数
            if end != '\n':
                message = message + end
            # 创建自定义LogRecord，设置正确的文件名和行号
            record = self.logger.makeRecord(
                self.logger.name, logging.DEBUG, filename, lineno, message, (), None
            )
            self.logger.handle(record)
            # 处理flush参数
            if flush:
                for handler in self.logger.handlers:
                    handler.flush()
    
    def info(self, message: str, end: str = '\n', flush: bool = False):
        """INFO级别日志"""
        if self._get_log_level() <= logging.INFO:
            filename, lineno = self._get_caller_info()
            if end != '\n':
                message = message + end
            record = self.logger.makeRecord(
                self.logger.name, logging.INFO, filename, lineno, message, (), None
            )
            self.logger.handle(record)
            if flush:
                for handler in self.logger.handlers:
                    handler.flush()
    
    def warning(self, message: str, end: str = '\n', flush: bool = False):
        """WARNING级别日志"""
        if self._get_log_level() <= logging.WARNING:
            filename, lineno = self._get_caller_info()
            if end != '\n':
                message = message + end
            record = self.logger.makeRecord(
                self.logger.name, logging.WARNING, filename, lineno, message, (), None
            )
            self.logger.handle(record)
            if flush:
                for handler in self.logger.handlers:
                    handler.flush()
    
    def error(self, message: str, end: str = '\n', flush: bool = False):
        """ERROR级别日志"""
        if self._get_log_level() <= logging.ERROR:
            filename, lineno = self._get_caller_info()
            if end != '\n':
                message = message + end
            record = self.logger.makeRecord(
                self.logger.name, logging.ERROR, filename, lineno, message, (), None
            )
            self.logger.handle(record)
            if flush:
                for handler in self.logger.handlers:
                    handler.flush()
    
    def critical(self, message: str, end: str = '\n', flush: bool = False):
        """CRITICAL级别日志"""
        if self._get_log_level() <= logging.CRITICAL:
            filename, lineno = self._get_caller_info()
            if end != '\n':
                message = message + end
            record = self.logger.makeRecord(
                self.logger.name, logging.CRITICAL, filename, lineno, message, (), None
            )
            self.logger.handle(record)
            if flush:
                for handler in self.logger.handlers:
                    handler.flush()


def gnn_log(level: str, message: str, end: str = '\n', flush: bool = False):
    """
    统一的GNN日志函数
    
    Args:
        level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        message: 日志消息
        end: 结尾字符，默认为换行符
        flush: 是否立即刷新缓冲区
    """
    logger = GNLogger()
    if level == "DEBUG":
        logger.debug(message, end, flush)
    elif level == "INFO":
        logger.info(message, end, flush)
    elif level == "WARNING":
        logger.warning(message, end, flush)
    elif level == "ERROR":
        logger.error(message, end, flush)
    elif level == "CRITICAL":
        logger.critical(message, end, flush)


def gnn_debug(message: str, end: str = '\n', flush: bool = False):
    """DEBUG级别便捷函数"""
    gnn_log("DEBUG", message, end, flush)


def gnn_info(message: str, end: str = '\n', flush: bool = False):
    """INFO级别便捷函数"""
    gnn_log("INFO", message, end, flush)


def gnn_warning(message: str, end: str = '\n', flush: bool = False):
    """WARNING级别便捷函数"""
    gnn_log("WARNING", message, end, flush)


def gnn_error(message: str, end: str = '\n', flush: bool = False):
    """ERROR级别便捷函数"""
    gnn_log("ERROR", message, end, flush)


def gnn_critical(message: str, end: str = '\n', flush: bool = False):
    """CRITICAL级别便捷函数"""
    gnn_log("CRITICAL", message, end, flush)


# 测试函数
def test_logging():
    """测试日志功能"""
    gnn_debug("这是DEBUG级别的测试消息")
    gnn_info("这是INFO级别的测试消息")
    gnn_warning("这是WARNING级别的测试消息")
    gnn_error("这是ERROR级别的测试消息")
    gnn_critical("这是CRITICAL级别的测试消息")


if __name__ == "__main__":
    test_logging()