"""
消融实验配置转换模块
负责将消融实验配置转换为模型训练所需的参数覆盖配置
"""

import yaml
import copy
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AblationConfigConverter:
    """消融实验配置转换器"""
    
    def __init__(self):
        # 加载外部配置映射文件
        self.config_mapping = self._load_config_mapping()
        
        # 特征消融配置映射: exp_config特征路径 -> 模型配置路径
        self.feature_mapping = {
            'flow_bert_multiview': {
                'enabled_features.sequence_features': 'data.sequence_features',
                'enabled_features.domain_name_embedding_features': 'data.domain_name_embedding_features',
                'enabled_features.text_features': 'data.text_features'
            },
            'flow_bert_multiview_ssl': {
                'enabled_features.sequence_features': 'data.sequence_features',
                'enabled_features.domain_name_embedding_features': 'data.domain_name_embedding_features',
                'enabled_features.text_features': 'data.text_features'
            }
        }
        
        # 融合消融配置映射
        self.fusion_mapping = {
            'flow_bert_multiview': {
                'fusion.method': 'model.fusion.method'
            },
            'flow_bert_multiview_ssl': {
                'fusion.method': 'model.fusion.method'
            }
        }
        
       

    def _load_config_mapping(self) -> Dict[str, Any]:
        """
        加载外部配置映射文件
        
        Returns:
            配置映射字典
        """
        try:
            current_dir = Path(__file__).parent
            mapping_file_path = current_dir / "config_mapping.yaml"
            
            if mapping_file_path.exists():
                with open(mapping_file_path, 'r', encoding='utf-8') as f:
                    mapping_data = yaml.safe_load(f)
                logger.info(f"成功加载配置映射文件: {mapping_file_path}")
                return mapping_data or {}
            else:
                logger.warning(f"配置映射文件不存在: {mapping_file_path}")
                return {}
        except Exception as e:
            logger.error(f"加载配置映射文件失败: {e}")
            return {}

    def _get_config_path(self, section: str, key: str, model_name: str) -> Optional[str]:
        """
        从配置映射中获取配置路径
        
        Args:
            section: 配置节名 (如 'multiview', 'fusion')
            key: 配置键名
            model_name: 模型名称
            
        Returns:
            配置路径字符串，如果未找到返回None
        """
        try:
            section_mapping = self.config_mapping.get(section, {})
            key_mapping = section_mapping.get(key, {})
            
            # 优先使用模型特定映射
            if model_name in key_mapping:
                return key_mapping[model_name]
            # 其次使用默认映射
            elif 'default' in key_mapping:
                return key_mapping['default']
            else:
                logger.warning(f"未找到配置路径映射: {section}.{key} for {model_name}")
                return None
        except Exception as e:
            logger.error(f"获取配置路径失败: {e}")
            return None

    def convert_feature_ablation(self, 
                                ablation_config: Dict[str, Any], 
                                model_name: str) -> Dict[str, Any]:
        """
        转换特征消融配置
        
        Args:
            ablation_config: 消融实验配置
            model_name: 模型名称
            
        Returns:
            参数覆盖配置
        """
        override_config = {}
        enabled_features = ablation_config.get('enabled_features', {})
        
        # 直接处理特征使能开关，映射到对应的enabled字段
        feature_mappings = [
            ('sequence_features', 'multiview'),
            ('domain_name_embedding_features', 'multiview'),
            ('text_features', 'multiview')
        ]
        
        for feature_name, section in feature_mappings:
            if feature_name in enabled_features:
                is_enabled = enabled_features[feature_name]
                param_path = self._get_config_path(section, feature_name, model_name)
                
                if not param_path and model_name in self.feature_mapping:
                    mapping = self.feature_mapping[model_name]
                    param_path = mapping.get(f'enabled_features.{feature_name}')
                
                if param_path:
                    # 根据启用状态设置enabled字段
                    override_config[f"{param_path}.enabled"] = is_enabled
                    logger.info(f"特征配置: {feature_name}={is_enabled} -> {param_path}.enabled={is_enabled}")
        
        logger.info(f"convert_feature_ablation[{ablation_config.get('name', 'unknown')}] override_config: {override_config}")
        return override_config

    def convert_fusion_ablation(self, 
                               ablation_config: Dict[str, Any], 
                               model_name: str) -> Dict[str, Any]:
        """
        转换融合消融配置
        
        Args:
            ablation_config: 消融实验配置
            model_name: 模型名称
            
        Returns:
            参数覆盖配置
        """
        override_config = {}
        method = ablation_config.get('method', 'concat')
        
        # 优先使用配置映射
        param_path = self._get_config_path('fusion', 'method', model_name)
        
        # 如果配置映射中没有，回退到硬编码映射
        if not param_path and model_name in self.fusion_mapping:
            mapping = self.fusion_mapping[model_name]
            for config_path, path in mapping.items():
                param_path = path
                break
        
        if param_path:
            override_config[param_path] = method
            logger.info(f"融合配置: method={method} -> {param_path}")
        else:
            logger.warning(f"未找到融合方法映射 for {model_name}")
        
        logger.info(f"convert_fusion_ablation[{ablation_config.get('name', 'unknown')}] override_config: {override_config}")
        return override_config

    def convert_loss_ablation(self, 
                             ablation_config: Dict[str, Any], 
                             model_name: str) -> Dict[str, Any]:
        """
        转换损失消融配置
        
        Args:
            ablation_config: 消融实验配置
            model_name: 模型名称
            
        Returns:
            参数覆盖配置
        """
        override_config = {}
        
        # if model_name not in self.loss_mapping:
        #     logger.warning(f"Model {model_name} not supported for loss ablation")
        #     return override_config
        
        # mapping = self.loss_mapping[model_name]
        
        # # 根据损失类型设置配置
        # loss_type = ablation_config.get('type', 'prediction')
        
        # if loss_type == 'prediction':
        #     # 仅预测损失
        #     override_config['model.ssl.enabled'] = False
        # elif loss_type == 'prediction+reconstruction':
        #     # 预测损失 + 数值特征恢复损失
        #     override_config['model.ssl.enabled'] = True
        #     override_config['model.ssl.reconstruction_loss'] = True
        #     override_config['model.ssl.mlm_loss'] = False
        # elif loss_type == 'prediction+reconstruction+mlm':
        #     # 预测损失 + 数值特征恢复损失 + MLM损失
        #     override_config['model.ssl.enabled'] = True
        #     override_config['model.ssl.reconstruction_loss'] = True
        #     override_config['model.ssl.mlm_loss'] = True
        
        logger.info(f"convert_loss_ablation[{ablation_config.get('name', 'unknown')}] override_config: {override_config}")
        return override_config

    def convert_ablation_config(self, 
                               ablation_variant: Dict[str, Any], 
                               model_name: str) -> Dict[str, Any]:
        """
        转换消融实验配置为参数覆盖配置
        
        Args:
            ablation_variant: 单个消融实验变体配置
            model_name: 模型名称
            
        Returns:
            完整的参数覆盖配置
        """
        ablation_type = ablation_variant.get('type')
        config = ablation_variant.get('config', {})

        # logger.info(f"convert_ablation_config[{ablation_variant.get('name', 'unknown')}] ablation_type: {ablation_type}")
        
        if ablation_type == 'feature_ablation':
            return self.convert_feature_ablation(config, model_name)
        elif ablation_type == 'fusion_ablation':
            # 融合消融：method 在根级别，传递整个 variant
            return self.convert_fusion_ablation(ablation_variant, model_name)
        elif ablation_type == 'loss_ablation':
            return self.convert_loss_ablation(config, model_name)
        else:
            logger.warning(f"Unknown ablation type: {ablation_type}")
            return {}

    def generate_hydra_overrides(self, 
                                override_config: Dict[str, Any],
                                model_name: str | None = None,
                                base_config_path: str | None = None,
                                general_overrides: List[str] | None = None) -> List[str]:
        """
        生成参数覆盖列表
        
        Args:
            override_config: 参数覆盖配置
            model_name: 模型名称
            base_config_path: 基础配置文件路径
            general_overrides: 通用配置覆盖列表
            
        Returns:
            完整的覆盖参数列表
        """
        overrides = []
        
        # 1. 添加通用配置覆盖（优先级最高）
        if general_overrides:
            overrides.extend(general_overrides)
            logger.info(f"添加通用覆盖参数: {general_overrides}")
        
        # 2. 判断是否为Python配置文件
        # is_python_config = base_config_path and base_config_path.endswith('.py')

        # TODO: 暂时不支持Python配置文件，因为需要修改代码，暂时不支持.默认是yaml文件做配置文件
        # 后续需要再补充
        is_python_config = False

        # 3. 处理消融变体配置
        for param_path, value in override_config.items():
            if is_python_config:
                # Python配置文件格式
                if value is None:
                    # 对于Python配置文件，使用null语法
                    key = param_path.replace('.', '_')
                    overrides.append(f"{key}=null")
                else:
                    # 转换为命令行参数格式
                    key = param_path.replace('.', '_')
                    # 转换布尔值为小写
                    if isinstance(value, bool):
                        value = str(value).lower()
                    overrides.append(f"{key}={value}")
            else:
                # Hydra配置文件格式
                if value is None:
                    # 删除参数，使用不带+前缀的null语法
                    overrides.append(f"{param_path}=null")
                else:
                    # 设置参数值，转换布尔值为小写
                    if isinstance(value, bool):
                        value = str(value).lower()
                    overrides.append(f"{param_path}={value}")
        
        logger.info(f"最终Hydra覆盖参数列表: {overrides}")
        return overrides

    def create_temp_config_file(self, 
                               override_config: Dict[str, Any], 
                               experiment_name: str) -> str:
        """
        创建临时配置文件
        
        Args:
            override_config: 参数覆盖配置
            experiment_name: 实验名称
            
        Returns:
            临时配置文件路径
        """
        import tempfile
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / f"ablation_config_{experiment_name}_{timestamp}.yaml"
        
        with open(temp_file, 'w') as f:
            yaml.dump(override_config, f, default_flow_style=False)
        
        return str(temp_file)

    def validate_override_config(self, 
                                override_config: Dict[str, Any], 
                                base_config_path: str) -> List[str]:
        """
        验证覆盖配置的有效性
        
        Args:
            override_config: 参数覆盖配置
            base_config_path: 基础配置文件路径
            
        Returns:
            错误列表
        """
        errors = []
        
        # 首先检查配置文件是否存在
        if not Path(base_config_path).exists():
            errors.append(f"Config file not found: {base_config_path}")
            return errors
        
        # 对于Python配置文件，跳过YAML验证
        if base_config_path.endswith('.py'):
            # Python配置文件无法直接验证内容，只做基本检查
            logger.info(f"Skipping YAML validation for Python config: {base_config_path}")
            return errors
        
        try:
            # 加载YAML基础配置
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
        except Exception as e:
            errors.append(f"Failed to load base YAML config: {e}")
            return errors
        
        for param_path, value in override_config.items():
            try:
                # 检查参数路径是否存在
                if not self._check_parameter_exists(base_config, param_path):
                    errors.append(f"Invalid parameter path: {param_path}")
                    continue
                
                # 检查参数类型匹配
                if value is not None and not self._check_parameter_type(base_config, param_path, value):
                    errors.append(f"Type mismatch for {param_path}")
                    
            except Exception as e:
                errors.append(f"Validation error for {param_path}: {e}")
        
        return errors

    def _check_parameter_exists(self, config: Dict[str, Any], param_path: str) -> bool:
        """检查参数路径是否存在"""
        keys = param_path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]
        
        return True

    def _check_parameter_type(self, 
                             config: Dict[str, Any], 
                             param_path: str, 
                             value: Any) -> bool:
        """检查参数类型是否匹配（支持类型转换）"""
        try:
            keys = param_path.split('.')
            current = config
            
            # 导航到父级
            for key in keys[:-1]:
                current = current[key]
            
            # 检查类型
            final_key = keys[-1]
            if final_key in current:
                existing_value = current[final_key]
                existing_type = type(existing_value)
                
                # 如果值为None，总是允许（表示删除或禁用）
                if value is None:
                    return True
                
                # 如果类型完全匹配，直接返回True
                if isinstance(value, existing_type):
                    return True
                
                # 处理字符串到布尔值的转换
                if existing_type is bool:
                    if isinstance(value, str):
                        return value.lower() in ('true', 'false', 'yes', 'no', '1', '0')
                    elif isinstance(value, (int, float)):
                        return value in (0, 1)
                    return False
                
                # 处理字符串到数字的转换
                if existing_type in (int, float):
                    if isinstance(value, str):
                        try:
                            existing_type(value)
                            return True
                        except ValueError:
                            return False
                
                # 处理None值到复杂类型的转换
                if existing_type in (dict, list):
                    if value is None:
                        return True
                
                # 如果不支持转换，返回False
                return False
            
            return True  # 如果参数不存在，认为类型匹配
        except:
            return True