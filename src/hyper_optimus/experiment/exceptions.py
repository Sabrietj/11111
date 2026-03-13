"""
自定义异常模块
定义消融实验的特定异常类型
"""


class AblationExperimentError(Exception):
    """消融实验基础异常类"""
    pass


class ConfigurationError(AblationExperimentError):
    """配置错误"""
    pass


class ModelNotFoundError(AblationExperimentError):
    """模型未找到错误"""
    pass


class ConfigValidationError(AblationExperimentError):
    """配置验证错误"""
    pass


class TrainingScriptError(AblationExperimentError):
    """训练脚本错误"""
    pass


class ExperimentExecutionError(AblationExperimentError):
    """实验执行错误"""
    pass


class WandBIntegrationError(AblationExperimentError):
    """W&B集成错误"""
    pass


class MetricsCollectionError(AblationExperimentError):
    """指标采集错误"""
    pass


class DependencyError(AblationExperimentError):
    """依赖错误"""
    pass


class ResourceError(AblationExperimentError):
    """资源错误（内存、磁盘空间等）"""
    pass


class TimeoutError(AblationExperimentError):
    """超时错误"""
    pass


class ValidationError(AblationExperimentError):
    """验证错误"""
    pass