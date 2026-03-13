#!/usr/bin/env python3
"""
消融实验CLI工具统一接口
提供上传、生成结果等命令行功能
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# 添加项目路径（确保模块导入正常）
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入框架组件
try:
    from .experiment_executor import ExperimentExecutor
    from .wandb_integration import WandBIntegration, BatchWandBUploader, manual_upload_ablation_results
    from .exp_utils import load_yaml_config, save_yaml_config, format_duration, ExperimentTimer
    from .exceptions import AblationExperimentError, ConfigurationError, WandBIntegrationError
except ImportError:
    from experiment_executor import ExperimentExecutor
    from wandb_integration import WandBIntegration, BatchWandBUploader, manual_upload_ablation_results
    from exp_utils import load_yaml_config, save_yaml_config, format_duration, ExperimentTimer
    from exceptions import AblationExperimentError, ConfigurationError, WandBIntegrationError

# 导入日志配置
try:
    from utils.logging_config import setup_preset_logging
except ImportError:
    import logging
    def setup_preset_logging(log_level):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

import logging
logger = setup_preset_logging(log_level=logging.INFO)


def find_latest_results():
    """查找最新的消融实验结果"""
    ablation_results_dir = project_root / 'ablation_results'
    
    if not ablation_results_dir.exists():
        logger.error("消融实验结果目录不存在")
        return None
    
    suite_dirs = [d for d in ablation_results_dir.iterdir() 
                if d.is_dir() and d.name.startswith('suite_')]
    
    if not suite_dirs:
        logger.error("未找到任何消融实验套件目录")
        return None
    
    latest_dir = max(suite_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"找到最新实验目录: {latest_dir}")
    return latest_dir


def validate_experiment_dir(results_dir: Path):
    """验证实验目录结构"""
    if not results_dir.exists():
        return False, "实验目录不存在"
    
    config_file = results_dir / 'experiment_config.yaml'
    if not config_file.exists():
        return False, "缺少 experiment_config.yaml 文件"
    
    sub_dirs = [d for d in results_dir.iterdir() 
               if d.is_dir() and not d.name.startswith('.')]
    if not sub_dirs:
        return False, "未找到子实验目录"
    
    return True, "目录结构有效"


def load_experiment_config(results_dir: Path) -> Dict[str, Any]:
    """加载实验配置"""
    config_file = results_dir / 'experiment_config.yaml'
    return load_yaml_config(str(config_file))

def parse_experiment_results(results_dir: Path, experiment_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """解析实验结果"""
    try:
        ablation_variants = experiment_config.get('ablation_variants', {})
        
        logger.info(f"找到 {len(ablation_variants)} 个消融变体配置")
        
        # 创建执行器实例
        executor = ExperimentExecutor(
            workspace_root=str(project_root)
        )
        
        results = []
        
        # 处理每个变体
        for variant_id, variant_config in ablation_variants.items():
            logger.info(f"处理变体: {variant_id}")
            
            # 查找对应的实验目录
            variant_dir = None
            for sub_dir in results_dir.iterdir():
                if sub_dir.is_dir() and variant_id in sub_dir.name:
                    variant_dir = sub_dir
                    break
            
            if not variant_dir:
                logger.warning(f"未找到变体 {variant_id} 的实验目录")
                continue
            
            log_file = variant_dir / 'training.log'
            if not log_file.exists():
                logger.warning(f"未找到变体 {variant_id} 的训练日志")
                continue
            
            try:
                # 只解析训练日志，不执行新的训练
                parsed_data = executor._parse_log_files(str(variant_dir))
                variant_result = {
                    'variant_id': variant_id,
                    'ablation_config': variant_config,
                    'log_file': str(log_file),
                    'status': 'completed' if parsed_data.get('test_results') else 'failed',
                    'test_results': parsed_data.get('test_results', {}),
                    'final_results': parsed_data.get('final_results', {}),
                    'epoch_metrics': parsed_data.get('epoch_metrics', []),
                    'learning_rates': parsed_data.get('learning_rates', []),
                    'duration': parsed_data.get('duration', 0),
                    'return_code': 0
                }
                results.append(variant_result)
                logger.info(f"成功解析变体 {variant_id}")
                
            except Exception as e:
                logger.error(f"解析变体 {variant_id} 失败: {e}")
                results.append({
                    'ablation_id': variant_id,
                    'ablation_config': variant_config,
                    'log_file': str(log_file),
                    'status': 'failed',
                    'error': f"解析失败: {e}",
                    'return_code': 1,
                    'duration': 0
                })
        
        logger.info(f"成功解析 {len(results)} 个变体结果")
        return results
        
    except Exception as e:
        logger.error(f"解析实验结果失败: {e}")
        return None


def generate_final_results(results_dir: Path, output_file: Optional[Path] = None):
    """生成 final_results.json"""
    if output_file is None:
        output_file = results_dir / 'final_results.json'
    
    # 验证目录
    is_valid, message = validate_experiment_dir(results_dir)
    if not is_valid:
        logger.error(f"目录验证失败: {message}")
        return False
    
    # 解析结果
    experiment_config = load_experiment_config(results_dir)
    results = parse_experiment_results(results_dir, experiment_config)
    if results is None:
        logger.error("解析实验结果失败")
        return False
    
    if not results:
        logger.error("没有找到任何实验结果")
        return False
    
    try:
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"结果已保存到: {output_file}")
        
        # 生成实验报告
        try:
            # 导入BatchExperimentExecutor
            from .experiment_executor import BatchExperimentExecutor
            
            # 创建批量实验执行器实例
            batch_executor = BatchExperimentExecutor(
                workspace_root=str(project_root),
                exp_config={}  # 使用默认配置，不启用延迟报告
            )
            
            # 生成实验报告
            batch_executor._generate_experiment_report(results, results_dir)
            logger.info(f"实验报告已生成: {results_dir / 'experiment_report.md'}")
            
        except Exception as e:
            logger.error(f"生成实验报告失败: {e}")
            import traceback
            traceback.print_exc()
            # 不影响主流程，只记录错误
        
        # 统计信息
        total = len(results)
        completed = sum(1 for r in results if r.get('status') == 'completed')
        failed = total - completed
        
        print(f"\n{'='*50}")
        print("实验结果生成完成")
        print(f"{'='*50}")
        print(f"总实验数: {total}")
        print(f"成功实验: {completed}")
        print(f"失败实验: {failed}")
        print(f"结果文件: {output_file}")
        
        if failed > 0:
            print(f"\n失败的实验:")
            for result in results:
                if result.get('status') == 'failed':
                    error = result.get('error', 'unknown error')
                    ablation_id = result.get('ablation_id', 'unknown')
                    print(f"  - {ablation_id}: {error}")
        
        return True
        
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        return False


def upload_ablation_results(results_source: str, 
                        project_name: Optional[str] = None,
                        entity: Optional[str] = None,
                        dry_run: bool = False):
    """上传消融实验结果到W&B"""
    
    # 确定结果文件路径
    results_path = Path(results_source)
    
    if results_path.is_file():
        results_file = results_path
    elif results_path.is_dir():
        results_file = results_path / 'final_results.json'
        
        # 如果文件不存在，尝试生成
        if not results_file.exists():
            print("结果文件不存在，尝试生成...")
            if not generate_final_results(results_path):
                print("生成结果文件失败")
                return False
    else:
        print(f"错误: 结果路径不存在或无效: {results_source}")
        return False
    
    if not results_file.exists():
        print(f"错误: 结果文件不存在: {results_file}")
        return False
    
    print(f"使用结果文件: {results_file}")
    
    # 验证文件格式
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if not isinstance(results, list):
            print("错误: 结果文件必须包含实验结果列表")
            return False
        
        print(f"验证通过: {len(results)} 个实验结果")
        
    except json.JSONDecodeError as e:
        print(f"错误: JSON格式无效: {e}")
        return False
    except Exception as e:
        print(f"错误: 无法读取结果文件: {e}")
        return False
    
    if dry_run:
        print("验证完成 (dry-run模式，未上传)")
        return True
    
    # 执行上传
    try:
        print("开始上传到W&B...")
        success = manual_upload_ablation_results(
            results_file=str(results_file),
            project_name=project_name,
            entity=entity
        )
        
        if success:
            print("上传成功!")
            return True
        else:
            print("上传失败!")
            return False
            
    except Exception as e:
        print(f"上传过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def cmd_generate(args):
    """生成结果命令"""
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.latest:
        results_dir = find_latest_results()
        if not results_dir:
            print("错误: 未找到任何消融实验目录")
            return 1
    else:
        print("错误: 请指定 --results-dir 或 --latest")
        return 1
    
    if args.validate_only:
        is_valid, message = validate_experiment_dir(results_dir)
        if is_valid:
            print(f"验证通过: {results_dir}")
            return 0
        else:
            print(f"验证失败: {message}")
            return 1
    
    success = generate_final_results(results_dir, args.output_file)
    return 0 if success else 1


def cmd_upload(args):
    """上传命令"""
    if args.results_file:
        results_source = args.results_file
    elif args.results_dir:
        results_source = args.results_dir
    elif args.latest:
        latest_dir = find_latest_results()
        if not latest_dir:
            print("错误: 未找到任何消融实验目录")
            return 1
        results_source = str(latest_dir)
    else:
        print("错误: 请指定 --results-file, --results-dir 或 --latest")
        return 1
    
    success = upload_ablation_results(
        results_source=results_source,
        project_name=args.project,
        entity=args.entity,
        dry_run=args.dry_run
    )
    
    return 0 if success else 1


def cmd_run(args):
    """运行实验命令"""
    # 复用现有的 run_ablation_exp.py 逻辑
    try:
        from .run_ablation_exp import main as run_main
    except ImportError:
        from run_ablation_exp import main as run_main
    
    # 设置sys.argv以适配现有脚本
    sys.argv = ['run_ablation_exp.py']
    
    if args.config:
        sys.argv.extend(['--config', args.config])
    if args.enable_delayed_report:
        sys.argv.append('--enable-delayed-report')
    if args.batch_wandb_upload:
        sys.argv.append('--batch-wandb-upload')
    if args.wandb_project:
        sys.argv.extend(['--wandb-project', args.wandb_project])
    if args.wandb_entity:
        sys.argv.extend(['--wandb-entity', args.wandb_entity])
    
    return run_main()


def create_parser():
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        description='消融实验CLI工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成最新实验的结果文件
  python -m hyper_optimus.experiment generate --latest
  
  # 生成指定目录的结果文件
  python -m hyper_optimus.experiment generate --results-dir ablation_results/suite_20251127_123456
  
  # 上传最新实验结果
  python -m hyper_optimus.experiment upload --latest
  
  # 上传指定目录结果
  python -m hyper_optimus.experiment upload --results-dir ablation_results/suite_20251127_123456
  
  # 验证目录结构
  python -m hyper_optimus.experiment generate --results-dir suite_XXX --validate-only
  
  # 运行消融实验
  python -m hyper_optimus.experiment run --config exp_config.yaml --enable-delayed-report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # generate 命令
    gen_parser = subparsers.add_parser('generate', help='生成实验结果文件')
    gen_group = gen_parser.add_mutually_exclusive_group(required=True)
    gen_group.add_argument('--results-dir', '-d', type=str, help='实验结果目录路径')
    gen_group.add_argument('--latest', '-l', action='store_true', help='处理最新的消融实验结果')
    gen_parser.add_argument('--output-file', '-o', type=str, help='输出文件路径')
    gen_parser.add_argument('--validate-only', action='store_true', help='仅验证目录结构，不生成文件')
    gen_parser.set_defaults(func=cmd_generate)
    
    # upload 命令
    upload_parser = subparsers.add_parser('upload', help='上传实验结果到W&B')
    upload_group = upload_parser.add_mutually_exclusive_group(required=True)
    upload_group.add_argument('--results-file', '-f', type=str, help='结果文件路径')
    upload_group.add_argument('--results-dir', '-d', type=str, help='实验结果目录路径')
    upload_group.add_argument('--latest', '-l', action='store_true', help='上传最新的实验结果')
    upload_parser.add_argument('--project', '-p', type=str, help='W&B项目名称')
    upload_parser.add_argument('--entity', '-e', type=str, help='W&B实体名称')
    upload_parser.add_argument('--dry-run', action='store_true', help='仅验证文件格式，不上传')
    upload_parser.set_defaults(func=cmd_upload)
    
    # run 命令
    run_parser = subparsers.add_parser('run', help='运行消融实验')
    run_parser.add_argument('--config', '-c', type=str, required=True, help='实验配置文件')
    run_parser.add_argument('--enable-delayed-report', action='store_true', help='启用延迟报告生成模式')
    run_parser.add_argument('--batch-wandb-upload', action='store_true', help='启用批量W&B上传')
    run_parser.add_argument('--wandb-project', type=str, help='W&B项目名称')
    run_parser.add_argument('--wandb-entity', type=str, help='W&B实体名称')
    run_parser.set_defaults(func=cmd_run)
    
    return parser


def main():
    """统一CLI入口点"""
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        return 1
    except AblationExperimentError as e:
        print(f"消融实验错误: {e}")
        return 1
    except Exception as e:
        print(f"未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())