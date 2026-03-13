#!/usr/bin/env python3
"""
消融实验主入口脚本
执行完整的消融实验套件，包括配置解析、实验执行和结果收集
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf


# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 使用统一的日志配置
from utils.logging_config import setup_preset_logging
import logging
logger = setup_preset_logging(log_level=logging.DEBUG)

logger.info(f"sys.path: {sys.path}")

# 获取根日志器并设置为DEBUG级别
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# 确保所有处理器也都是DEBUG级别
for handler in root_logger.handlers:
    handler.setLevel(logging.DEBUG)

    

# 导入消融实验框架
from hyper_optimus.experiment.config_converter import AblationConfigConverter
from hyper_optimus.experiment.wandb_integration import WandBIntegration, RealTimeMetricsCollector, MetricsAggregator
from hyper_optimus.experiment.experiment_executor import ExperimentExecutor, BatchExperimentExecutor
from hyper_optimus.experiment.variant_identifier import VariantIdentifier
from hyper_optimus.experiment.exp_utils import (
    load_yaml_config, 
    save_yaml_config,
    load_json_file,
    save_json_file,
    validate_config_schema,
    get_experiment_config_schema,
    format_duration,
    format_file_size,
    ExperimentTimer,
    check_dependencies
)
from hyper_optimus.experiment.exceptions import (
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


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="执行多视图网络流量分类消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            示例用法:
            python run_ablation_exp.py --config src/hyper_optimus/configs/exp_config.yaml
            python run_ablation_exp.py --config exp_config.yaml --parallel 2
            python run_ablation_exp.py --config exp_config.yaml --output-dir ./results --debug
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='消融实验配置文件路径'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./ablation_results',
        help='实验结果输出目录 (默认: ./ablation_results)'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=1,
        help='并行执行的实验数量 (默认: 1)'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='日志文件路径 (可选)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='启用调试模式 (等价于 --log-level DEBUG)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅验证配置，不执行实验'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='检查依赖包是否满足要求'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='multiview-ablation-studies',
        help='W&B项目名称 (默认: multiview-ablation-studies)'
    )
    
    parser.add_argument(
        '--wandb-entity',
        type=str,
        help='W&B实体名称 (可选)'
    )
    
    parser.add_argument(
        '--enable-delayed-report',
        action='store_true',
        help='启用延迟报告生成模式'
    )
    
    parser.add_argument(
        '--batch-wandb-upload',
        action='store_true',
        help='启用批量W&B上传（配合延迟报告使用）'
    )
    
    parser.add_argument(
        '--manual-upload',
        type=str,
        help='手动上传指定结果文件到W&B'
    )
    
    parser.add_argument(
        '--auto-assign-ids',
        action='store_true',
        default=True,
        help='自动分配变体ID (默认: 启用)'
    )
    
    parser.add_argument(
        '--no-auto-assign-ids',
        action='store_true',
        help='禁用自动分配变体ID'
    )
    
    parser.add_argument(
        '--id-type',
        type=str,
        choices=['standard', 'readable', 'semantic'],
        default='semantic',
        help='变体ID生成类型 (默认: semantic)'
    )
    
    return parser.parse_args()


def check_required_dependencies():
    """检查必需的依赖包"""
    required_packages = [
        'yaml',
        'wandb', 
        'torch',
        'pytorch_lightning',
        'transformers',
        'hydra',
        'omegaconf'
    ]
    
    logging.info("Checking dependencies...")
    dependencies = check_dependencies(required_packages)
    
    missing_packages = [pkg for pkg, available in dependencies.items() if not available]
    
    if missing_packages:
        error_msg = f"Missing required packages: {', '.join(missing_packages)}"
        logging.error(error_msg)
        logging.error("Please install missing packages using: pip install " + " ".join(missing_packages))
        raise DependencyError(error_msg)
    
    logging.info("All required dependencies are available")
    return True


def validate_experiment_config(config_path: str, auto_assign_ids: bool = True, id_type: str = 'semantic') -> dict:
    """
    验证实验配置并自动分配变体ID
    
    Args:
        config_path: 配置文件路径
        auto_assign_ids: 是否自动分配变体ID
        id_type: ID类型 ('standard', 'readable', 'semantic')
        
    Returns:
        验证后的配置字典
        
    Raises:
        ConfigurationError: 配置错误
        ConfigValidationError: 配置验证错误
    """
    try:
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        # 加载配置
        logging.info(f"Loading configuration from: {config_path}")
        config = load_yaml_config(config_path)
        
        if not config:
            raise ConfigurationError("Configuration file is empty or invalid")
        
        logging.info(f"config: \n{OmegaConf.to_yaml(config)}")

        # 检查消融实验变体
        ablation_variants = config.get('ablation_variants', {})
        if not ablation_variants:
            raise ConfigValidationError("No ablation variants found in configuration")
        
        logging.info(f"Found {len(ablation_variants)} ablation variants")
        
        # 初始化变体标识器
        identifier = VariantIdentifier()
        
        # 验证变体ID格式
        validation_errors = identifier.validate_variant_ids(ablation_variants)
        if validation_errors:
            error_msg = "Variant ID validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ConfigValidationError(error_msg)
        
        # 自动分配ID（如果启用）
        if auto_assign_ids:
            logging.info(f"Auto-assigning variant IDs using {id_type} method...")
            updated_variants = identifier.assign_variant_ids(ablation_variants, id_type)
            config['ablation_variants'] = updated_variants
            
            # 打印ID映射
            logging.info("Variant ID mapping:")
            for new_id, variant in updated_variants.items():
                original_id = variant.get('original_id', new_id)
                logging.info(f"  {original_id} -> {new_id}")
        
        # 生成变体摘要
        summary = identifier.get_variant_summary(config['ablation_variants'])
        logging.info(f"Variant summary: {summary}")
        
        # 验证必需字段
        for variant_id, variant in config['ablation_variants'].items():
            if not isinstance(variant, dict):
                raise ConfigValidationError(f"Variant {variant_id} must be a dictionary")
            
            if not variant.get('name'):
                raise ConfigValidationError(f"Missing 'name' in variant: {variant_id}")
            
            if not variant.get('description'):
                raise ConfigValidationError(f"Missing 'description' in variant: {variant_id}")
            
            if not variant.get('type'):
                raise ConfigValidationError(f"Missing 'type' in variant: {variant_id}")
            
            variant_type = variant.get('type')
            
            # 根据类型验证特定字段
            if variant_type == 'feature_ablation':
                if not variant.get('config'):
                    raise ConfigValidationError(f"Missing 'config' in variant: {variant_id}")
                if 'enabled_features' not in variant.get('config', {}):
                    raise ConfigValidationError(f"Missing 'enabled_features' in config for feature_ablation variant: {variant_id}")
            elif variant_type == 'fusion_ablation':
                # 融合消融可以直接有method字段，也可以有config.method
                if variant.get('method'):
                    # 直接有method字段
                    pass
                elif variant.get('config', {}).get('method'):
                    # 在config中有method字段
                    pass
                else:
                    raise ConfigValidationError(f"Missing 'method' in variant or config for fusion_ablation variant: {variant_id}")
            elif variant_type == 'loss_ablation':
                if not variant.get('model_name'):
                    raise ConfigValidationError(f"Missing 'model_name' in loss_ablation variant: {variant_id}")
        
        logging.info("Configuration validation passed")
        return config
        
    except Exception as e:
        if isinstance(e, (ConfigurationError, ConfigValidationError)):
            raise
        else:
            raise ConfigurationError(f"Failed to validate configuration: {e}")


def setup_experiment_environment(args, config):
    """
    设置实验环境
    
    Args:
        args: 命令行参数
        config: 实验配置
        
    Returns:
        实验执行器
    """
    # 设置工作目录为项目根目录，而不是当前 ablation_experiment 目录
    workspace_root = Path(__file__).parent.parent.parent
    
    # 创建输出目录
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载实验配置（包含通用配置）
    logging.info(f"Loading configuration from: {args.config}")
    config = load_yaml_config(args.config)
    
    # 初始化批量实验执行器
    executor = BatchExperimentExecutor(
        workspace_root=str(workspace_root),
        max_parallel_jobs=args.parallel,
        exp_config=config  # 传递完整实验配置
    )
    
    logging.info(f"Workspace root: {workspace_root}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Parallel jobs: {args.parallel}")
    
    return executor, str(output_dir)


def print_experiment_summary(config, output_dir):
    """
    打印实验摘要
    
    Args:
        config: 实验配置
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("消融实验执行摘要")
    print("="*60)
    
    # 基本信息
    model_name = config.get('experiment', {}).get('model_name', 'unknown')
    experiment_type = config.get('experiment', {}).get('type', 'unknown')
    ablation_variants = config.get('ablation_variants', {})
    
    print(f"模型名称: {model_name}")
    print(f"实验类型: {experiment_type}")
    print(f"消融变体数量: {len(ablation_variants)}")
    print(f"输出目录: {output_dir}")
    
    # 变体列表
    print("\n消融实验变体:")
    print("-" * 120)
    print(f"{'ID':<25} {'原始ID':<8} {'类型':<25} {'名称':<32} {'基线':<6} {'描述'}")
    print("-" * 120)
    
    for variant_id, variant in ablation_variants.items():
        original_id = variant.get('original_id', variant_id)[:12]
        variant_type = variant.get('type', 'N/A')[:25]
        name = variant.get('name', 'N/A')[:40]
        baseline = '是' if variant.get('baseline', False) else '否'
        description = variant.get('description', 'N/A')[:50]
        
        print(f"{variant_id:<25} {original_id:<8} {variant_type:<25} {name:<32} {baseline:<6} {description}")
    
    print("="*60)
    print()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志
    log_level = 'DEBUG' if args.debug else args.log_level
    
    # 记录启动信息
    logging.info("Starting ablation experiment execution")
    logging.info(f"Arguments: {vars(args)}")
    
    # 检查依赖
    if args.check_deps:
        try:
            # check_required_dependencies()
            logging.info("Dependency check completed successfully")
            return 0
        except DependencyError as e:
            logging.error(f"Dependency check failed: {e}")
            return 1
    
    # 手动上传模式
    if args.manual_upload:
        try:
            from wandb_integration import manual_upload_ablation_results
            
            print(f"开始手动上传结果文件: {args.manual_upload}")
            success = manual_upload_ablation_results(
                results_file=args.manual_upload,
                project_name=args.wandb_project,
                entity=args.wandb_entity
            )
            
            if success:
                print("手动上传完成")
                return 0
            else:
                print("手动上传失败")
                return 1
                
        except Exception as e:
            print(f"手动上传错误: {e}")
            return 1
    
    try:
        # 确定是否自动分配ID
        auto_assign_ids = args.auto_assign_ids and not args.no_auto_assign_ids
        
        # 验证配置
        config = validate_experiment_config(
            args.config, 
            auto_assign_ids=auto_assign_ids, 
            id_type=args.id_type
        )
        
        # 如果是dry-run，仅验证配置
        if args.dry_run:
            print("Configuration validation passed. Dry-run mode, no experiments executed.")
            return 0
        
        # 检查依赖
        # check_required_dependencies()
        
        # 设置实验环境
        executor, output_dir = setup_experiment_environment(args, config)
        
        # 配置延迟报告和批量上传
        if args.enable_delayed_report:
            executor.exp_config.setdefault('experiment', {})['enable_delayed_report'] = True
        
        if args.batch_wandb_upload:
            executor.exp_config.setdefault('experiment', {})['batch_wandb_upload'] = True
        
        # 打印实验摘要
        print_experiment_summary(config, output_dir)
        
        # 默认自动执行，确认执行
        if args.debug:
            response = input("是否开始执行消融实验? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("实验执行已取消")
                return 0
        
        # 启动计时器
        timer = ExperimentTimer()
        timer.start()
        
        # 执行消融实验套件
        logging.info("Starting ablation experiment suite...")
        results, experiment_suite_dir = executor.execute_ablation_suite(
            experiment_config=config,
            base_output_dir=output_dir
        )
        
        # 停止计时器
        timer.stop()
        
        # 生成执行报告
        print("\n" + "="*60)
        print("实验执行完成")
        print("="*60)
        
        total_experiments = len(results)
        successful_experiments = sum(1 for r in results if r.get('status') == 'completed')
        failed_experiments = total_experiments - successful_experiments
        
        print(f"总实验数: {total_experiments}")
        print(f"成功实验: {successful_experiments}")
        print(f"失败实验: {failed_experiments}")
        print(f"执行时间: {format_duration(timer.get_duration() or 0)}")
        print(f"结果目录: {experiment_suite_dir}")
        
        if failed_experiments > 0:
            print("\n失败的实验:")
            for result in results:
                if result.get('status') == 'failed':
                    error = result.get('error', 'unknown error')
        
        # 保存执行摘要到实验套件目录
        summary_file = Path(experiment_suite_dir) / 'execution_summary.yaml'
        summary_data = {
            'execution_time': timer.get_summary(),
            'experiment_summary': {
                'total_experiments': total_experiments,
                'successful_experiments': successful_experiments,
                'failed_experiments': failed_experiments
            },
            'config_file': args.config,
            'command_args': vars(args),
            'results': results
        }
        
        from exp_utils import save_yaml_config
        save_yaml_config(summary_data, str(summary_file))
        
        print(f"\n执行摘要已保存到: {summary_file}")
        
        return 0 if failed_experiments == 0 else 1
        
    except KeyboardInterrupt:
        logging.info("Experiment execution interrupted by user")
        print("\n实验执行被用户中断")
        return 1
        
    except AblationExperimentError as e:
        logging.error(f"Ablation experiment error: {e}")
        print(f"\n消融实验错误: {e}")
        return 1
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n未预期的错误: {e}")
        print("请查看日志文件获取详细信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())