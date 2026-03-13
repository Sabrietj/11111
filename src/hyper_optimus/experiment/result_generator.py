#!/usr/bin/env python3
"""
实验结果生成器
专门用于从原始日志生成 final_results.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .experiment_executor import ExperimentExecutor, BatchExperimentExecutor
from .exp_utils import load_yaml_config
from .exceptions import AblationExperimentError

# 导入日志配置
try:
    from utils.logging_config import setup_preset_logging
except ImportError:
    import logging
    def setup_preset_logging(log_level):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

logger = setup_preset_logging(log_level=logging.INFO)


class ResultGenerator:
    """实验结果生成器"""
    
    def __init__(self, workspace_root: str = None):
        """初始化结果生成器
        
        Args:
            workspace_root: 工作空间根目录
        """
        if workspace_root is None:
            # 自动检测项目根目录
            current_file = Path(__file__)
            workspace_root = current_file.parent.parent.parent
        
        self.workspace_root = Path(workspace_root)
        self.executor = ExperimentExecutor(workspace_root=str(self.workspace_root))
    
    def validate_experiment_structure(self, results_dir: Path) -> Tuple[bool, str]:
        """验证实验目录结构
        
        Args:
            results_dir: 实验结果目录
            
        Returns:
            (是否有效, 错误信息)
        """
        if not results_dir.exists():
            return False, "实验目录不存在"
        
        # 检查配置文件
        config_file = results_dir / 'experiment_config.yaml'
        if not config_file.exists():
            return False, "缺少 experiment_config.yaml 文件"
        
        # 检查子实验目录
        sub_dirs = [d for d in results_dir.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
        if not sub_dirs:
            return False, "未找到子实验目录"
        
        return True, "目录结构有效"
    
    def load_experiment_config(self, results_dir: Path) -> Dict[str, Any]:
        """加载实验配置
        
        Args:
            results_dir: 实验结果目录
            
        Returns:
            实验配置字典
        """
        config_file = results_dir / 'experiment_config.yaml'
        
        try:
            config = load_yaml_config(str(config_file))
            logger.debug(f"成功加载实验配置: {config_file}")
            return config
        except Exception as e:
            logger.error(f"加载实验配置失败: {e}")
            raise AblationExperimentError(f"无法加载实验配置: {e}")
    
    def find_variant_directory(self, results_dir: Path, variant_id: str) -> Optional[Path]:
        """查找变体对应的实验目录
        
        Args:
            results_dir: 实验结果根目录
            variant_id: 变体ID
            
        Returns:
            变体目录路径或None
        """
        # 精确匹配
        for sub_dir in results_dir.iterdir():
            if sub_dir.is_dir() and variant_id in sub_dir.name:
                return sub_dir
        
        # 模糊匹配（容错）
        for sub_dir in results_dir.iterdir():
            if sub_dir.is_dir() and sub_dir.name.startswith(variant_id):
                logger.warning(f"使用模糊匹配: {variant_id} -> {sub_dir.name}")
                return sub_dir
        
        return None
    
    def parse_variant_result(self, 
                          results_dir: Path, 
                          variant_id: str, 
                          variant_config: Dict[str, Any]) -> Dict[str, Any]:
        """解析单个变体的结果
        
        Args:
            results_dir: 实验结果根目录
            variant_id: 变体ID
            variant_config: 变体配置
            
        Returns:
            变体结果字典
        """
        # 查找变体目录
        variant_dir = self.find_variant_directory(results_dir, variant_id)
        if not variant_dir:
            logger.warning(f"未找到变体 {variant_id} 的实验目录")
            return self._create_failed_result(variant_id, variant_config, "未找到实验目录")
        
        # 查找训练日志
        log_file = variant_dir / 'training.log'
        if not log_file.exists():
            logger.warning(f"未找到变体 {variant_id} 的训练日志")
            return self._create_failed_result(variant_id, variant_config, "未找到训练日志", str(log_file))
        
        try:
            # 加载实验配置
            experiment_config = self.load_experiment_config(results_dir)
            
            # 只解析训练日志，不执行新的训练
            parsed_data = self.executor._parse_log_files(str(variant_dir))
            variant_result = {
                'variant_id': variant_id,
                'ablation_config': variant_config,
                'log_file': str(log_file),
                'status': 'completed' if parsed_data.get('test_results') else 'failed',
                'test_results': parsed_data.get('test_results', {}),
                'final_results': parsed_data.get('final_results', {}),
                'epoch_metrics': parsed_data.get('epoch_metrics', []),
                'learning_rates': parsed_data.get('learning_rates', []),
                'parsed_metrics': parsed_data.get('parsed_metrics', {}),
                'return_code': 0
            }
            
            logger.info(f"成功解析变体 {variant_id}")
            return variant_result
            
        except Exception as e:
            logger.error(f"解析变体 {variant_id} 失败: {e}")
            return self._create_failed_result(variant_id, variant_config, f"解析失败: {e}", str(log_file))
    
    def _create_failed_result(self, 
                           variant_id: str, 
                           variant_config: Dict[str, Any], 
                           error_message: str, 
                           log_file: Optional[str] = None) -> Dict[str, Any]:
        """创建失败结果字典
        
        Args:
            variant_id: 变体ID
            variant_config: 变体配置
            error_message: 错误信息
            log_file: 日志文件路径
            
        Returns:
            失败结果字典
        """
        result = {
            'ablation_id': variant_id,
            'ablation_config': variant_config,
            'status': 'failed',
            'return_code': 1,
            'error': error_message,
            'parsed_metrics': {}
        }
        
        if log_file:
            result['log_file'] = log_file
        
        return result
    
    def generate_results(self, results_dir: Path) -> List[Dict[str, Any]]:
        """生成实验结果列表
        
        Args:
            results_dir: 实验结果目录
            
        Returns:
            实验结果列表
        """
        logger.info(f"开始生成实验结果: {results_dir}")
        
        # 验证目录结构
        is_valid, message = self.validate_experiment_structure(results_dir)
        if not is_valid:
            raise AblationExperimentError(f"目录验证失败: {message}")
        
        # 加载实验配置
        experiment_config = self.load_experiment_config(results_dir)
        ablation_variants = experiment_config.get('ablation_variants', {})
        
        if not ablation_variants:
            raise AblationExperimentError("实验配置中未找到消融变体")
        
        logger.info(f"找到 {len(ablation_variants)} 个消融变体配置")
        
        # 解析每个变体
        results = []
        for variant_id, variant_config in ablation_variants.items():
            logger.info(f"处理变体: {variant_id}")
            
            variant_result = self.parse_variant_result(results_dir, variant_id, variant_config)
            results.append(variant_result)
        
        logger.info(f"成功生成 {len(results)} 个变体结果")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: Path) -> bool:
        """保存结果到文件
        
        Args:
            results: 实验结果列表
            output_file: 输出文件路径
            
        Returns:
            是否保存成功
        """
        try:
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"结果已保存到: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            return False
    
    def generate_and_save(self, results_dir: Path, output_file: Optional[Path] = None) -> bool:
        """生成并保存实验结果
        
        Args:
            results_dir: 实验结果目录
            output_file: 输出文件路径（可选）
            
        Returns:
            是否成功
        """
        if output_file is None:
            output_file = results_dir / 'final_results.json'
        
        try:
            # 生成结果
            results = self.generate_results(results_dir)
            
            if not results:
                logger.error("没有生成任何实验结果")
                return False
            
            # 保存结果
            success = self.save_results(results, output_file)
            
            if success:
                # 打印统计信息
                self._print_summary(results, output_file)
                
                # 生成实验报告
                self._generate_experiment_report(results, results_dir)
            
            return success
            
        except Exception as e:
            logger.error(f"生成和保存结果失败: {e}")
            return False
    
    def _generate_experiment_report(self, results: List[Dict[str, Any]], results_dir: Path):
        """生成实验报告
        
        Args:
            results: 实验结果列表
            results_dir: 实验结果目录
        """
        try:
            # 创建批量实验执行器实例
            batch_executor = BatchExperimentExecutor(
                workspace_root=str(self.workspace_root),
                exp_config={}  # 使用默认配置，不启用延迟报告
            )
            
            # 生成实验报告
            batch_executor._generate_experiment_report(results, results_dir)
            logger.info(f"实验报告已生成: {results_dir / 'experiment_report.md'}")
            
        except Exception as e:
            logger.error(f"生成实验报告失败: {e}")
            # 不影响主流程，只记录错误

    def _print_summary(self, results: List[Dict[str, Any]], output_file: Path):
        """打印结果摘要
        
        Args:
            results: 实验结果列表
            output_file: 输出文件路径
        """
        total = len(results)
        completed = sum(1 for r in results if r.get('status') == 'completed')
        failed = total - completed
        
        print(f"\n{'='*60}")
        print("实验结果生成完成")
        print(f"{'='*60}")
        print(f"总实验数: {total}")
        print(f"成功实验: {completed}")
        print(f"失败实验: {failed}")
        print(f"结果文件: {output_file}")
        print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示失败实验详情
        if failed > 0:
            print(f"\n失败的实验:")
            for result in results:
                if result.get('status') == 'failed':
                    error = result.get('error', 'unknown error')
                    ablation_id = result.get('ablation_id', 'unknown')
                    print(f"  - {ablation_id}: {error}")
        
        print(f"{'='*60}")


# 便捷函数，供CLI和外部调用
def generate_final_results(results_dir: str, 
                       output_file: Optional[str] = None,
                       workspace_root: Optional[str] = None) -> bool:
    """便捷函数：生成实验结果
    
    Args:
        results_dir: 实验结果目录
        output_file: 输出文件路径
        workspace_root: 工作空间根目录
        
    Returns:
        是否成功
    """
    generator = ResultGenerator(workspace_root)
    results_path = Path(results_dir)
    output_path = Path(output_file) if output_file else None
    
    return generator.generate_and_save(results_path, output_path)


def validate_experiment_structure(results_dir: str, 
                              workspace_root: Optional[str] = None) -> Tuple[bool, str]:
    """便捷函数：验证实验目录结构
    
    Args:
        results_dir: 实验结果目录
        workspace_root: 工作空间根目录
        
    Returns:
        (是否有效, 错误信息)
    """
    generator = ResultGenerator(workspace_root)
    return generator.validate_experiment_structure(Path(results_dir))