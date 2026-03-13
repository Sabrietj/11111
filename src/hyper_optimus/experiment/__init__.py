"""
消融实验脚本包
提供统一的消融实验执行框架
"""

from .config_converter import AblationConfigConverter
from .wandb_integration import (
    WandBIntegration, 
    RealTimeMetricsCollector, 
    MetricsAggregator,
    BatchWandBUploader,
    manual_upload_ablation_results
)
from .experiment_executor import ExperimentExecutor
from .variant_identifier import VariantIdentifier
from .result_generator import ResultGenerator, generate_final_results, validate_experiment_structure




from .exp_utils import (
    load_yaml_config, 
    save_yaml_config,
    load_json_file,
    save_json_file,
    validate_config_schema,
    get_experiment_config_schema,
    format_duration,
    format_file_size,
    ExperimentTimer
)
from .exceptions import (
    AblationExperimentError,
    ConfigurationError,
    ModelNotFoundError,
    ConfigValidationError,
    TrainingScriptError,
    ExperimentExecutionError,
    WandBIntegrationError,
    MetricsCollectionError,
    DependencyError,
    ResourceError,
    TimeoutError,
    ValidationError
)

# CLI工具入口
from .cli_tools import main as cli_main

__version__ = "1.0.0"
__author__ = "Ablation Experiment Framework"

__all__ = [
    # 核心组件
    "AblationConfigConverter",
    "WandBIntegration", 
    "RealTimeMetricsCollector",
    "MetricsAggregator",
    "BatchWandBUploader",
    "ExperimentExecutor",
    "VariantIdentifier",
    "ResultGenerator",
    
    # CLI工具
    "cli_main",
    
    # 上传和生成功能
    "manual_upload_ablation_results",
    "generate_final_results",
    "validate_experiment_structure",
    
    # 工具函数
    "setup_logging",
    "load_yaml_config",
    "save_yaml_config", 
    "load_json_file",
    "save_json_file",
    "validate_config_schema",
    "get_experiment_config_schema",
    "format_duration",
    "format_file_size",
    "ExperimentTimer",
    
    # 异常类
    "AblationExperimentError",
    "ConfigurationError", 
    "ModelNotFoundError",
    "ConfigValidationError",
    "TrainingScriptError",
    "ExperimentExecutionError",
    "WandBIntegrationError",
    "MetricsCollectionError",
    "DependencyError",
    "ResourceError",
    "TimeoutError",
    "ValidationError"
]