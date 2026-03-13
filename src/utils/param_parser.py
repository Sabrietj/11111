import os
import re
from typing import Dict, Any, Optional
from configparser import ConfigParser

class ParamParser:
    """参数解析器，支持脚本参数 > 配置文件 > 默认参数的优先级"""
    
    def __init__(self, config_path: str = "src_qyf/utils/config.cfg"):
        self.config = ConfigParser()
        self.config.read(config_path)
    
    def get_all_params(self, 
                      mode: str,
                      script_args: Optional[Dict[str, Any]] = None,
                      config_section: Optional[str] = None) -> Dict[str, Any]:
        """
        获取所有参数，优先级：脚本参数 > 配置文件 > 默认参数
        
        Args:
            mode: 训练模式 (pretrain, supervised, end2end)
            script_args: 脚本传入的参数
            config_section: 配置文件中特定的section
        """
        params = {}
        
        # 1. 从默认参数配置获取基础值
        default_params = self._get_default_params(mode)

        params.update(default_params)

        # 2. 用配置文件参数覆盖
        if config_section and self.config.has_section(config_section):
            config_params = self._get_config_params(config_section)
            params.update(config_params)
  
        # 3. 用脚本参数覆盖（最高优先级）
        if script_args:
            params.update(script_args)
  
        # 4. 特殊处理trainset路径解析
        if 'trainset' in params and isinstance(params['trainset'], str):
            params['trainset'] = self._resolve_path(params['trainset'])
        

        return params
    
    def _get_default_params(self, mode: str) -> Dict[str, Any]:
        """从默认参数配置获取参数"""
        params = {}
        if self.config.has_section('default_params'):
            for key, value in self.config.items('default_params'):
                if key.startswith(f"{mode}."):
                    param_name = key.split('.', 1)[1]
                    params[param_name] = self._parse_config_value(value)
        return params
    
    def _get_config_params(self, config_section: str) -> Dict[str, Any]:
        """从配置文件特定section获取参数"""
        params = {}
        if self.config.has_section(config_section):
            for key, value in self.config.items(config_section):
                params[key] = self._parse_config_value(value)
        return params
    
    def _resolve_path(self, path_template: str) -> str:
        """解析路径模板中的变量"""
        # 支持${SECTION:KEY}格式的变量替换
        pattern = r'\$\{([^:]+):([^}]+)\}'
        
        def replace_var(match):
            section = match.group(1)
            key = match.group(2)
            if self.config.has_section(section) and self.config.has_option(section, key):
                return self.config.get(section, key)
            return match.group(0)  # 找不到配置时返回原字符串
        
        resolved_path = re.sub(pattern, replace_var, path_template)
        
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(resolved_path):
            # 获取当前工作目录或配置中的根目录
            if self.config.has_section('PATH') and self.config.has_option('PATH', 'data_root'):
                data_root = self.config.get('PATH', 'data_root')
                resolved_path = os.path.join(data_root, resolved_path)
        
        return resolved_path
    
    def _parse_config_value(self, value: str) -> Any:
        """解析配置文件中的值"""
        # 处理路径变量
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            return self._resolve_path(value)
        
        # 布尔值处理
        if isinstance(value, str) and value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 数字处理
        if isinstance(value, str):
            try:
                # 先尝试直接转换为int（支持科学计数法）
                return int(value)
                
            except ValueError:
                try:
                    # 如果int失败，尝试float
                    return float(value)
                except ValueError:
                    pass
        
        # 字符串处理
        return value