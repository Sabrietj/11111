"""
实验执行器模块 - 重构版本
负责执行单个消融实验，包括训练脚本调用和结果收集
"""

from hyper_optimus.experiment.wandb_integration import WandBIntegration
from math import log
import subprocess
import os
import sys
import traceback
import yaml
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from omegaconf import OmegaConf

import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 运行时导入
try:
    from .wandb_integration import WandBIntegration
except ImportError:
    WandBIntegration = None

# 单独导入wandb用于图像处理
try:
    import wandb
except ImportError:
    wandb = None

from .config_converter import AblationConfigConverter

# 使用统一的日志配置
try:
    from utils.logging_config import setup_preset_logging
except ImportError:
    import logging
    def setup_preset_logging(log_level):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

import logging
logger = setup_preset_logging(logging.DEBUG)

# TensorBoard相关导入
try:
    from tensorboard.backend.event_processing import event_file_loader
    TENSORBOARD_IMPORT_AVAILABLE = True
except ImportError:
    TENSORBOARD_IMPORT_AVAILABLE = False
    logger.warning("TensorBoard not available, TB log parsing disabled")

class ExperimentExecutor:
    """实验执行器 - 重构版本"""
    
    def __init__(self, 
                 workspace_root: str,
                 wandb_project: str = "multiview-ablation-studies",
                 wandb_entity: Optional[str] = None,
                 exp_config: Optional[Dict[str, Any]] = None):
        self.workspace_root = Path(workspace_root)
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.exp_config = exp_config or {}
        
        # 初始化组件
        self.config_converter = AblationConfigConverter()
        if WandBIntegration is not None:
            self.wandb_integration: 'WandBIntegration' = WandBIntegration(wandb_project, wandb_entity)
        else:
            self.wandb_integration = None
            print("Warning: WandBIntegration not available, W&B logging disabled")
        
        # 模型训练脚本路径映射
        self.model_script_mapping = {
            'flow_autoencoder': 'models/flow_autoencoder/train.py',
            'flow_bert_multiview': 'models/flow_bert_multiview/train.py',
            'flow_bert_multiview_ssl': 'models/flow_bert_multiview_ssl/train.py',
            'flow_bert_multiview_ssl_mlm': 'models/flow_bert_multiview_ssl/train.py',
            'bert_iat': 'models/bert_iat/main.py',
            'fsnet': 'models/fsnet/main.py',
            'appscanner': 'models/appscanner/appscanner_main_model.py'
        }

    def _parse_epoch_patterns(self, content: str, epoch_dict: dict):
        """统一的epoch模式解析函数，减少重复代码"""
        # 定义epoch解析模式
        epoch_patterns = {
            # PyTorch Lightning 进度条格式 - 只匹配epoch完成时的100%状态
            'pytorch_lightning': r'Epoch\s+(\d+):\s*100%\|[^|]*\|\s*\d+/\d+\s*\[[^\]]*,\s*train_accuracy=([\d.]+),\s*train_precision=([\d.]+),\s*train_recall=([\d.]+),\s*train_f1=([\d.]+),\s*val_accuracy=([\d.]+),\s*val_precision=([\d.]+),\s*val_recall=([\d.]+),\s*val_f1=([\d.]+)',
            # 测试结果模式 - 匹配测试阶段的指标
            'test_results': r'test_accuracy=([\d.]+).*test_precision=([\d.]+).*test_recall=([\d.]+).*test_f1=([\d.]+)',
            'format1': r'Epoch\s*(\d+).*100%.*train_loss:\s*([\d.]+).*train_acc(?:uracy)?:\s*([\d.]+).*train_f1:\s*([\d.]+).*val_loss:\s*([\d.]+).*val_acc(?:uracy)?:\s*([\d.]+).*val_f1:\s*([\d.]+).*(?:lr|learning_rate):\s*([\d.e-]+)',
            'format2': r'Epoch\s*(\d+).*100%.*train/loss[=\s]\s*([\d.]+).*train/acc(?:uracy)?[=\s]\s*([\d.]+).*train/f1[=\s]\s*([\d.]+).*val/loss[=\s]\s*([\d.]+).*val/acc(?:accuracy)?[=\s]\s*([\d.]+).*val/f1[=\s]\s*([\d.]+).*(?:lr|learning_rate)[=\s]\s*([\d.e-]+)',
            'format3': r'Epoch\s*(\d+).*100%.*train_recon_loss[=\s]\s*([\d.]+).*train_class_loss[=\s]\s*([\d.]+).*train_total_loss[=\s]\s*([\d.]+).*train_accuracy[=\s]\s*([\d.]+).*val_recon_loss[=\s]\s*([\d.]+).*val_class_loss[=\s]\s*([\d.]+).*val_total_loss[=\s]\s*([\d.]+).*val_accuracy[=\s]\s*([\d.]+).*learning_rate[=\s]\s*([\d.e-]+)',
            'format4': r'Epoch\s*(\d+).*100%.*train_classification_loss[=\s]\s*([\d.]+).*train_total_loss[=\s]\s*([\d.]+).*train_accuracy[=\s]\s*([\d.]+).*train_precision[=\s]\s*([\d.]+).*train_recall[=\s]\s*([\d.]+).*train_f1[=\s]\s*([\d.]+).*val_classification_loss[=\s]\s*([\d.]+).*val_accuracy[=\s]\s*([\d.]+).*val_precision[=\s]\s*([\d.]+).*val_recall[=\s]\s*([\d.]+).*val_f1[=\s]\s*([\d.]+)',
            'format5': r'Epoch\s*(\d+).*100%.*cpu[=%\s]\s*([\d.]+).*memory[=%\s]\s*([\d.]+).*gpu[=%\s]\s*([\d.]+)'
        }
        
        # 应用所有模式进行解析
        for pattern_name, pattern in epoch_patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                try:
                    if pattern_name == 'format5':
                        # format5是系统指标，需要特殊处理
                        epoch, cpu_usage, memory_usage, gpu_usage = match.groups()
                        epoch_num = int(epoch)
                        
                        # 如果epoch已存在，添加系统指标；否则创建新条目
                        if epoch_num in epoch_dict:
                            epoch_dict[epoch_num].update({
                                'cpu_percent': float(cpu_usage),
                                'memory_usage': float(memory_usage),
                                'gpu_utilization': float(gpu_usage)
                            })
                        else:
                            epoch_dict[epoch_num] = {
                                'epoch': epoch_num,
                                'cpu_percent': float(cpu_usage),
                                'memory_usage': float(memory_usage),
                                'gpu_utilization': float(gpu_usage)
                            }
                    else:
                        # 其他格式正常处理
                        epoch_data = self._parse_epoch_match(match, pattern_name)
                        if epoch_data:
                            epoch_num = epoch_data['epoch']
                            epoch_dict[epoch_num] = epoch_data
                except (ValueError, IndexError) as e:
                    logger.debug(f"解析epoch匹配失败 ({pattern_name}): {e}")

    def _parse_epoch_match(self, match, pattern_name: str):
        """解析单个epoch匹配结果"""
        groups = match.groups()
        
        if pattern_name == 'pytorch_lightning':
            epoch, train_acc, train_prec, train_rec, train_f1, val_acc, val_prec, val_rec, val_f1 = groups
            return {
                'epoch': int(epoch),
                'train_accuracy': float(train_acc),
                'train_precision': float(train_prec),
                'train_recall': float(train_rec),
                'train_f1': float(train_f1),
                'val_accuracy': float(val_acc),
                'val_precision': float(val_prec),
                'val_recall': float(val_rec),
                'val_f1': float(val_f1)
            }
        
        elif pattern_name == 'test_results':
            test_acc, test_prec, test_rec, test_f1 = groups
            return {
                'epoch': -1,  # 使用-1表示测试结果不属于特定epoch
                'test_accuracy': float(test_acc),
                'test_precision': float(test_prec),
                'test_recall': float(test_rec),
                'test_f1': float(test_f1)
            }
        
        elif pattern_name in ['format1', 'format2']:
            epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, lr = groups
            return {
                'epoch': int(epoch),
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'train_f1': float(train_f1),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'val_f1': float(val_f1),
                'learning_rate': float(lr)
            }
        
        elif pattern_name == 'format3':
            epoch, train_recon_loss, train_class_loss, train_total_loss, train_accuracy, val_recon_loss, val_class_loss, val_total_loss, val_accuracy, lr = groups
            return {
                'epoch': int(epoch),
                'train_recon_loss': float(train_recon_loss),
                'train_class_loss': float(train_class_loss),
                'train_total_loss': float(train_total_loss),
                'train_accuracy': float(train_accuracy),
                'val_recon_loss': float(val_recon_loss),
                'val_class_loss': float(val_class_loss),
                'val_total_loss': float(val_total_loss),
                'val_accuracy': float(val_accuracy),
                'learning_rate': float(lr)
            }
        
        elif pattern_name == 'format4':
            epoch, train_class_loss, train_total_loss, train_acc, train_prec, train_rec, train_f1, val_class_loss, val_acc, val_prec, val_rec, val_f1 = groups
            return {
                'epoch': int(epoch),
                'train_classification_loss': float(train_class_loss),
                'train_total_loss': float(train_total_loss),
                'train_accuracy': float(train_acc),
                'train_precision': float(train_prec),
                'train_recall': float(train_rec),
                'train_f1': float(train_f1),
                'val_classification_loss': float(val_class_loss),
                'val_accuracy': float(val_acc),
                'val_precision': float(val_prec),
                'val_recall': float(val_rec),
                'val_f1': float(val_f1)
            }
        

        
        return None

    def _extract_training_duration(self, content: str) -> float:
        """
        从日志内容中提取训练时长
        
        Args:
            content: 日志文件内容
            
        Returns:
            训练时长（秒），如果无法提取则返回0
            1) 直接在代码中记录训练进程启动到退出的时长
            2) 或者查看日志文件的时间戳
        """
        try:
            # 定义时间戳模式，支持多种格式
            timestamp_patterns = [
                # 格式1: [2025-11-27 13:11:27,757]
                r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]',
                # 格式2: [2025-11-27 13:22:33]
                r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]',
                # 格式3: 2025-11-27 13:11:27,757
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})',
                # 格式4: 2025-11-27 13:22:33
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
            ]
            
            timestamps = []
            
            # 尝试所有模式来提取时间戳
            for pattern in timestamp_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    try:
                        # 标准化时间格式，去掉毫秒部分
                        timestamp_str = match.replace(',', '.').split('.')[0]
                        # 解析为datetime对象
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        # logger.info(f"提取到时间戳: {timestamp}")   

                        timestamps.append(timestamp)
                    except ValueError:
                        continue
            
            # 去重并排序
            if timestamps:
                timestamps = sorted(set(timestamps))
                start_time = timestamps[0]
                end_time = timestamps[-1]
                
                # 计算时长（秒）
                duration = (end_time - start_time).total_seconds()
                
                logger.info(f"训练时长计算: 开始时间 {start_time}, 结束时间 {end_time}, 时长 {duration:.1f}秒")
                return duration
            
        except Exception as e:
            logger.debug(f"提取训练时长失败: {e}")
        
        return 0.0

    def _parse_log_files(self, output_dir: str) -> Dict:
        """从输出文件解析训练指标"""
        epoch_metrics = []
        final_results = {}
        test_results = {}
        learning_rates = []
        epoch_dict = {}  # 初始化 epoch_dict
        training_duration = 0.0  # 训练时长

        logger.info(f"解析训练日志文件: {output_dir}")
        
        # 查找可能的日志文件
        log_patterns = [
            Path(output_dir) / "training.log",
        ]
        
        log_files = []
        for pattern in log_patterns:
            if pattern.parent.exists():
                log_files.extend(pattern.parent.glob(pattern.name))
        logger.info(f"找到日志文件: {log_files}")
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                logger.info(f"解析日志文件: {log_file}")
                
                # 提取训练时长
                file_duration = self._extract_training_duration(content)
                if file_duration > 0:
                    training_duration = file_duration
                
                # 使用统一的解析函数来减少重复代码，但保留原始功能
                self._parse_epoch_patterns(content, epoch_dict)
                
                # 将字典转换为列表，按epoch排序，确保每个epoch只有最终状态
                if epoch_dict:
                    epoch_metrics.extend(sorted(epoch_dict.values(), key=lambda x: x['epoch']))
                    # 提取学习率（如果存在）
                    for metrics in epoch_dict.values():
                        if 'learning_rate' in metrics:
                            learning_rates.append(metrics['learning_rate'])
                    logger.info(f"解析日志文件获取epoch_metrics: {epoch_metrics}")
                
                # 解析测试集指标 - 扩展支持更多测试指标
                test_patterns = [
                    r'Test[^:]*:\s*accuracy[^:]*:\s*([\d.]+)',
                    r'Test[^:]*:\s*f1[^:]*:\s*([\d.]+)',
                    r'Test[^:]*:\s*precision[^:]*:\s*([\d.]+)',
                    r'Test[^:]*:\s*recall[^:]*:\s*([\d.]+)',
                    r'Test[^:]*:\s*auc[^:]*:\s*([\d.]+)',
                    r'Test[^:]*:\s*loss[^:]*:\s*([\d.]+)',
                    r'test_acc(?:uracy)?[=\s]\s*([\d.]+)',
                    r'test_f1[=\s]\s*([\d.]+)',
                    r'test_precision[=\s]\s*([\d.]+)',
                    r'test_recall[=\s]\s*([\d.]+)',
                    r'test_auc[=\s]\s*([\d.]+)',
                    r'test_loss[=\s]\s*([\d.]+)'
                ]
                
                # 新的表格格式测试指标解析
                table_test_patterns = [
                    r'│\s*test_avg_precision\s*│\s*([\d.]+)\s*│',
                    r'│\s*test_classification_loss\s*│\s*([\d.]+)\s*│', 
                    r'│\s*test_f1\s*│\s*([\d.]+)\s*│',
                    r'│\s*test_precision\s*│\s*([\d.]+)\s*│',
                    r'│\s*test_recall\s*│\s*([\d.]+)\s*│',
                    r'│\s*test_roc_auc\s*│\s*([\d.]+)\s*│'
                ]
                
                # 通用表格行匹配 - 匹配任意 test_* 指标
                general_table_pattern = r'│\s*(test_[\w_]+)\s*│\s*([\d.]+)\s*│'
                
                # 解析分类报告格式 - PyTorch Lightning Autoencoder格式
                classification_report_patterns = [
                    # 整体准确率格式: accuracy                           1.00     19250
                    r'accuracy\s+([\d.]+)\s+\d+',
                    # macro avg格式: macro avg       1.00      1.00      1.00     19250
                    r'macro\s+avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+',
                    # weighted avg格式: weighted avg       1.00      1.00      1.00     19250
                    r'weighted\s+avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+',
                    # 各类别指标格式: 正常       1.00      1.00      1.00     18553
                    r'(正常|恶意|benign|malicious)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+'
                ]
                
                for pattern in classification_report_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(match) == 1:  # accuracy
                            test_results['test_accuracy'] = float(match[0])
                        elif len(match) == 3:  # macro/weighted avg
                            precision, recall, f1 = match
                            if 'macro' in pattern:
                                test_results['test_macro_precision'] = float(precision)
                                test_results['test_macro_recall'] = float(recall)
                                test_results['test_macro_f1'] = float(f1)
                            elif 'weighted' in pattern:
                                test_results['test_weighted_precision'] = float(precision)
                                test_results['test_weighted_recall'] = float(recall)
                                test_results['test_weighted_f1'] = float(f1)
                        elif len(match) == 4:  # class-specific
                            class_name, precision, recall, f1 = match
                            if '正常' in class_name or 'benign' in class_name.lower():
                                test_results['test_benign_precision'] = float(precision)
                                test_results['test_benign_recall'] = float(recall)
                                test_results['test_benign_f1'] = float(f1)
                            elif '恶意' in class_name or 'malicious' in class_name.lower():
                                test_results['test_malicious_precision'] = float(precision)
                                test_results['test_malicious_recall'] = float(recall)
                                test_results['test_malicious_f1'] = float(f1)
                
                # 解析表格格式的测试指标
                for pattern in table_test_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # 从模式中提取指标名称
                        if 'test_avg_precision' in pattern:
                            test_results['test_avg_precision'] = float(match[0])
                        elif 'test_classification_loss' in pattern:
                            test_results['test_classification_loss'] = float(match[0])
                        elif 'test_f1' in pattern:
                            test_results['test_f1'] = float(match[0])
                        elif 'test_precision' in pattern:
                            test_results['test_precision'] = float(match[0])
                        elif 'test_recall' in pattern:
                            test_results['test_recall'] = float(match[0])
                        elif 'test_roc_auc' in pattern:
                            test_results['test_roc_auc'] = float(match[0])
                
                # 使用通用模式匹配所有test_*指标
                table_matches = re.findall(general_table_pattern, content, re.IGNORECASE)
                for metric_name, value in table_matches:
                    test_results[metric_name] = float(value)
                
                for pattern in test_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        metric_name = pattern.split(':')[0].split('[')[0] if ':' in pattern else pattern.split('=')[0].split('[')[0]
                        metric_name = metric_name.replace('Test', '').replace('test_', '').strip().lower()
                        test_results[f'test_{metric_name}'] = float(match)
                
                # 解析最终准确率和F1分数
                acc_patterns = [
                    r'(?:final\s*)?(?:train|val|test)[^:]*accuracy[^:]*:\s*([\d.]+)',
                    r'(?:final\s*)?(?:train|val|test)[^:]*acc[^:]*:\s*([\d.]+)'
                ]
                for pattern in acc_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        final_results['accuracy'] = float(match)
                
                f1_patterns = [
                    r'(?:final\s*)?(?:train|val|test)[^:]*f1[^:]*:\s*([\d.]+)',
                    r'(?:final\s*)?(?:train|val|test)[^:]*f1[^:]*:\s*([\d.]+)'
                ]
                for pattern in f1_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        final_results['f1_score'] = float(match)
                
                # 解析其他业务指标
                business_patterns = [
                    r'(?:throughput|吞吐量)[^:]*:\s*([\d.]+)',
                    r'(?:latency|延迟)[^:]*:\s*([\d.]+)',
                    r'(?:inference_time|推理时间)[^:]*:\s*([\d.]+)',
                    r'(?:model_size|模型大小)[^:]*:\s*([\d.]+)',
                    r'(?:memory_usage|内存使用)[^:]*:\s*([\d.]+)'
                ]
                
                # 解析混淆矩阵格式 - [[TP FN] [FP TN]]
                confusion_matrix_pattern = r'\[\[([\d]+)\s+([\d]+)\]\s+\[([\d]+)\s+([\d]+)\]\]'
                cm_matches = re.findall(confusion_matrix_pattern, content)
                for match in cm_matches:
                    tn, fp, fn, tp = match  # 注意：实际格式可能是 [[TN FP] [FN TP]] 或 [[TP FN] [FP TN]]
                    test_results['confusion_matrix'] = {
                        'true_negatives': int(tn),
                        'false_positives': int(fp), 
                        'false_negatives': int(fn),
                        'true_positives': int(tp)
                    }
                
                # 解析GPU内存使用
                gpu_memory_pattern = r'GPU内存使用[^\d]*([\d.]+)\s*GB'
                gpu_matches = re.findall(gpu_memory_pattern, content)
                for match in gpu_matches:
                    test_results['gpu_memory_gb'] = float(match)
                
                for pattern in business_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        metric_name = pattern.split(':')[0].split('(')[0].strip().lower()
                        final_results[metric_name] = float(match)
                
                # 解析独立的系统资源指标（非epoch绑定）
                system_patterns = [
                    # CPU使用率模式
                    r'CPU[使用率]*(?:使用)*[:\s]*([0-9.]+)%?',
                    r'cpu[使用率]*(?:usage)*[:\s]*([0-9.]+)%?',
                    r'cpu_percent[=\s]*([0-9.]+)',
                    # 内存使用率模式
                    r'Memory[使用]*[:\s]*([0-9.]+)%?',
                    r'memory[使用]*(?:usage)*[:\s]*([0-9.]+)%?',
                    r'memory_percent[=\s]*([0-9.]+)',
                    r'Memory[使用]*[:\s]*([0-9.]+)\s*GB',
                    r'memory[使用]*(?:used)*[:\s]*([0-9.]+)\s*GB',
                    # GPU使用率模式
                    r'GPU[使用率]*(?:使用)*[:\s]*([0-9.]+)%?',
                    r'gpu[使用率]*(?:usage)*[:\s]*([0-9.]+)%?',
                    r'gpu_utilization[=\s]*([0-9.]+)',
                    r'GPU[内存]*(?:memory)*[:\s]*([0-9.]+)\s*GB',
                    r'gpu[内存]*(?:used)*[:\s]*([0-9.]+)\s*GB',
                ]
                
                # 将解析到的系统指标添加到final_results中（作为总体系统指标）
                for pattern in system_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        value = float(matches[-1])  # 取最后一个值（最可能是最终状态）
                        
                        if 'cpu' in pattern.lower():
                            final_results['system_cpu_percent'] = value
                        elif 'memory' in pattern.lower() and 'gb' in pattern.lower():
                            final_results['system_memory_used_gb'] = value
                        elif 'memory' in pattern.lower():
                            final_results['system_memory_percent'] = value
                        elif 'gpu' in pattern.lower() and 'gb' in pattern.lower():
                            final_results['system_gpu_memory_used_gb'] = value
                        elif 'gpu' in pattern.lower():
                            final_results['system_gpu_utilization'] = value
                        
            except Exception as e:
                logger.debug(f"解析日志文件 {log_file} 失败: {e}")
        

        return {
            'epoch_metrics': epoch_metrics,
            'final_results': final_results,
            'test_results': test_results,
            'learning_rates': learning_rates,
            'duration': training_duration  # 添加训练时长
        }
    
    def _upload_to_wandb(self, wandb_run, parsed_data: Dict):
        """上传解析的数据到W&B"""
        if not wandb_run:
            logger.warning("W&B运行对象为空，无法上传数据")
            return
        
        logger.info(f"==== 上传解析的数据到W&B: \n\n {parsed_data}\n")
        try:
            # 上传epoch指标 - 扩展支持更多指标
            for i, metric in enumerate(parsed_data['epoch_metrics']):
                log_data = {}  # 移除epoch字段，改用step参数
                
                # 训练指标 - 统一使用层级结构命名
                if 'train_loss' in metric:
                    log_data['train/loss'] = metric['train_loss']
                if 'train_acc' in metric:
                    log_data['train/accuracy'] = metric['train_acc']
                if 'train_f1' in metric:
                    log_data['train/f1'] = metric['train_f1']
                if 'train_precision' in metric:
                    log_data['train/precision'] = metric['train_precision']
                if 'train_recall' in metric:
                    log_data['train/recall'] = metric['train_recall']
                if 'train_accuracy' in metric:
                    log_data['train/accuracy'] = metric['train_accuracy']
                if 'train_classification_loss' in metric:
                    log_data['train/classification_loss'] = metric['train_classification_loss']
                if 'train_total_loss' in metric:
                    log_data['train/total_loss'] = metric['train_total_loss']
                if 'train_recon_loss' in metric:
                    log_data['train/reconstruction_loss'] = metric['train_recon_loss']
                if 'train_class_loss' in metric:
                    log_data['train/classification_loss'] = metric['train_class_loss']
                
                # 验证指标 - 统一使用层级结构命名
                if 'val_loss' in metric:
                    log_data['val/loss'] = metric['val_loss']
                if 'val_acc' in metric:
                    log_data['val/accuracy'] = metric['val_acc']
                if 'val_f1' in metric:
                    log_data['val/f1'] = metric['val_f1']
                if 'val_precision' in metric:
                    log_data['val/precision'] = metric['val_precision']
                if 'val_recall' in metric:
                    log_data['val/recall'] = metric['val_recall']
                if 'val_accuracy' in metric:
                    log_data['val/accuracy'] = metric['val_accuracy']
                if 'val_classification_loss' in metric:
                    log_data['val/classification_loss'] = metric['val_classification_loss']
                if 'val_total_loss' in metric:
                    log_data['val/total_loss'] = metric['val_total_loss']
                if 'val_recon_loss' in metric:
                    log_data['val/reconstruction_loss'] = metric['val_recon_loss']
                if 'val_class_loss' in metric:
                    log_data['val/classification_loss'] = metric['val_class_loss']
                if 'val_auc' in metric:
                    log_data['val/auc'] = metric['val_auc']
                
                # 学习率指标
                if 'learning_rate' in metric:
                    log_data['train/learning_rate'] = metric['learning_rate']
                
                # 尝试从metric中解析系统指标，而不是实时采集
                log_data.update(self._extract_system_metrics_from_epoch(metric))
                
                # 只有当log_data不为空时才上传
                if log_data:
                    # 使用step参数记录epoch，解决"epoch not monotonically increasing"警告
                    wandb_run.log(log_data, step=metric['epoch'])
                    logger.info(f"上传epoch指标: {json.dumps(log_data, indent=2)}")
                else:
                    logger.info(f"Epoch {metric['epoch']} 无有效指标，跳过上传")
            
            # 上传测试结果
            test_results = parsed_data.get('test_results', {})
            if test_results:
                wandb_run.summary.update(test_results)
                # logger.info(f"上传测试指标: {list(test_results.keys())}")
                logger.info(f"上传epoch指标: {json.dumps(test_results, indent=2)}")

            
            # 上传最终结果
            final_results = parsed_data.get('final_results', {})
            if final_results:
                wandb_run.summary.update(final_results)
                # logger.info(f"上传最终结果: {list(final_results.keys())}")
                logger.info(f"上传最终结果: {json.dumps(final_results, indent=2)}")
            
            # 上传系统资源汇总
            system_summary = self._get_system_summary()
            if system_summary:
                wandb_run.summary.update(system_summary)
                logger.info(f"上传系统资源汇总: {json.dumps(system_summary, indent=2)}")
            
            logger.info(f"成功上传 {len(parsed_data['epoch_metrics'])} 个epoch指标到W&B")

            
        except Exception as e:
            logger.error(f"上传W&B失败: {e}")
    
    def _extract_system_metrics_from_epoch(self, metric: Dict) -> Dict:
        """
        从epoch指标中提取系统资源指标
        优先从解析的metric中获取，如果不存在则返回空字典
        这样可以避免实时采集系统指标导致的不准确问题
        """
        system_metrics = {}
        
        # 从metric中提取系统资源指标（如果日志中包含的话）
        system_field_mapping = {
            'cpu_percent': 'system/cpu_percent',
            'cpu_usage': 'system/cpu_percent',
            'memory_used': 'system/memory_used_gb',
            'memory_usage': 'system/memory_percent',
            'gpu_utilization': 'system/gpu_utilization',
            'gpu_memory_used': 'system/gpu_memory_used',
            'gpu_memory_usage': 'system/gpu_memory_percent',
            'system_cpu': 'system/cpu_percent',
            'system_memory': 'system/memory_percent',
            'system_gpu': 'system/gpu_utilization'
        }
        
        for source_field, target_field in system_field_mapping.items():
            if source_field in metric:
                try:
                    value = float(metric[source_field])
                    if 'memory' in target_field and value > 100:  # 假设是MB或GB值，需要转换
                        # 如果值大于100，假设是MB，转换为GB
                        if value > 1000:
                            value = value / (1024**3)
                        else:
                            value = value / 1024  # MB to GB
                    system_metrics[target_field] = value
                except (ValueError, TypeError):
                    logger.debug(f"无法解析系统指标 {source_field}: {metric[source_field]}")
        
        return system_metrics
    
    def _get_system_metrics(self, enable_realtime: bool = False):
        """
        获取系统资源指标
        
        Args:
            enable_realtime: 是否启用实时采集（默认False，避免不准确数据）
            
        Returns:
            系统指标字典
        """
        # 如果禁用实时采集，直接返回空字典
        if not enable_realtime:
            return {}
            
        try:
            import psutil
            import time
            
            # CPU使用率 - 使用非阻塞方式
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            
            # GPU使用情况（如果可用）
            gpu_metrics = {}
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # 使用第一个GPU
                    gpu_metrics = {
                        'system/gpu_utilization': gpu.load * 100,
                        'system/gpu_memory_used': gpu.memoryUsed,
                        'system/gpu_memory_total': gpu.memoryTotal,
                        'system/gpu_temp': gpu.temperature if hasattr(gpu, 'temperature') else 0,
                        'system/gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0
                    }
            except ImportError:
                logger.debug("GPUtil未安装，跳过GPU指标采集")
                pass
            except Exception as e:
                logger.debug(f"获取GPU指标失败: {e}")
            
            system_metrics = {
                'system/cpu_percent': cpu_percent,
                'system/memory_used_gb': memory.used / (1024**3),
                'system/memory_total_gb': memory.total / (1024**3),
                'system/memory_percent': memory.percent,
                'system/timestamp': time.time(),
                **gpu_metrics
            }
            
            return system_metrics
            
        except ImportError:
            logger.debug("psutil未安装，跳过系统资源监控")
            return {}
        except Exception as e:
            logger.debug(f"获取系统指标失败: {e}")
            return {}
    
    def _get_system_summary(self):
        """获取系统资源汇总信息"""
        try:
            import psutil
            import platform
            
            # 系统基本信息
            system_info = {
                'system/cpu_count': psutil.cpu_count(),
                'system/cpu_count_logical': psutil.cpu_count(logical=True),
                'system/memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'system/disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'system/platform': platform.platform(),
                'system/python_version': platform.python_version()
            }
            
            # GPU信息
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    system_info.update({
                        'system/gpu_name': gpu.name,
                        'system/gpu_memory_total': gpu.memoryTotal,
                        'system/gpu_driver': gpu.driver if hasattr(gpu, 'driver') else 'unknown'
                    })
            except:
                pass
            
            return system_info
            
        except ImportError:
            return {}
        except Exception as e:
            logger.debug(f"获取系统汇总信息失败: {e}")
            return {}

    def _execute_training_simplified(self, 
                                   training_command: List[str],
                                   experiment_context: Dict) -> Dict:
        """简化训练执行 - 直接输出到控制台"""
        execution_result = {}
        
        try:
            # 创建输出目录 - 从命令参数中提取hydra.run.dir的路径
            output_dir = None
            for arg in training_command:
                if arg.startswith('hydra.run.dir='):
                    output_dir = Path(arg.split('=', 1)[1])
                    break
                elif arg == '--output_dir':
                    # Python配置文件格式
                    output_dir = Path(training_command[training_command.index('--output_dir') + 1])
                    break
            
            if output_dir is None:
                raise ValueError("Output directory not found in training command")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 启动训练进程 - 使用项目根目录作为工作目录
            project_root = self.workspace_root  # 项目根目录

            # logger.info(f"启动进程 运行training_command: {' '.join(training_command)}")
            logger.info(f"\n\n================= 开始模型[{experiment_context.get('experiment_name')}-{experiment_context.get('variant_id')}]训练... ... ======================")
            
            # 计算训练时间
            duration = 0

            # 创建日志文件以捕获训练输出
            log_file = output_dir / 'training.log'
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=== TRAINING COMMAND ===\n")
                f.write(' '.join(training_command))
                f.write("\n\n=== TRAINING OUTPUT ===\n")
                f.flush()
                
                import time
                #计算训练时间
                start_time = time.time()    

                # 启动训练进程
                with subprocess.Popen(
                    training_command,
                    cwd=str(project_root),  # 使用项目根目录作为工作目录
                    stdout=f,  # 重定向stdout到文件
                    stderr=subprocess.STDOUT,  # 重定向stderr到同一文件
                    env={**os.environ, 'PYTHONUNBUFFERED': '1'},  # 强制Python无缓冲输出
                    text=True,
                    bufsize=1  # 行缓冲
                ) as process:
                    # 等待进程完成
                    return_code = process.wait()

                    end_time = time.time()
                    duration = end_time - start_time
                    logger.info(f"模型[{experiment_context.get('experiment_name')}-{experiment_context.get('variant_id')}]训练完成，耗时: {duration:.2f}秒")
                    
                    logger.info(f"===== 训练进程结束，返回码: {return_code}")
                    
                    # 写入返回码信息
                    f.write(f"\n\n=== PROCESS RETURN CODE ===\n")
                    f.write(str(return_code))
                    f.write(f"\n\n=== TRAINING COMPLETED ===\n")
            
            # 实验结束后处理数据
            parsed_data = self._parse_log_files(str(output_dir))
            
            execution_result.update({
                'return_code': return_code,
                'stdout': f"训练命令: {' '.join(training_command)}\n返回码: {return_code}",
                'stderr': '',
                'log_file': str(log_file),
                'parsed_metrics': parsed_data,   # 解析的指标数据 epoch_metrics, final_results, test_results,learning_rates,duration,
                'duration': duration 
            })
            
            if return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code, training_command, execution_result['stdout'], execution_result['stderr']
                )
            
        except Exception as e:
            execution_result['execution_error'] = str(e)
            logger.exception(f"执行训练时出错: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return execution_result
    
    def _build_training_command(self, experiment_config: Dict, variant_id: str, variant: Dict, output_dir: str) -> List[str]:
        """
        构建训练命令 - 使用 config_converter 进行消融配置转换
        
        Args:
            experiment_config: 实验配置（包含 variant 和其他信息）
            output_dir: 输出目录
            
        Returns:
            训练命令列表
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # 1. 提取关键配置信息
            # 优先从variant中获取model_name，如果不存在则从experiment_config中获取
            model_name = variant.get('model_name') or experiment_config.get("experiment", {}).get('model_name', 'unknown')

            logger.info(f"model_name: {model_name} - variant_id: {variant_id}  ")
            
            # 获取训练脚本路径，并转换为绝对路径
            script_path = self.model_script_mapping.get(model_name)
            if not script_path:
                raise ValueError(f"Unknown model name: {model_name}")

            
            # 构建完整的脚本路径
            script_full_path = str(self.workspace_root / script_path)
            if not Path(script_full_path).exists():
                raise FileNotFoundError(f"Training script not found: {script_full_path}")
            
            logger.info(f"script_full_path for [{model_name}]: {script_full_path}")

            # 2. 使用 config_converter 转换消融配置
            override_config = {}
            if variant and 'type' in variant:
                override_config = self.config_converter.convert_ablation_config(variant, model_name)
                logger.debug(f"转换消融配置 {variant_id}: {override_config}")
            
            # 3. 构建通用配置覆盖（从 experiment_config 中的基础配置）
            general_overrides = []
            
            # 处理 data 配置
            data_config = experiment_config.get('data', {})
            for key, value in data_config.items():
                param_path = self.config_converter._get_config_path('data', key, model_name)
                if param_path:
                    general_overrides.append(f"{param_path}={value}")
            logger.debug(f"data_config for [{model_name}]: {data_config}")
            
            # 处理 training 配置  
            training_config = experiment_config.get('training', {})
            for key, value in training_config.items():
                param_path = self.config_converter._get_config_path('training', key, model_name)
                if param_path:
                    general_overrides.append(f"{param_path}={value}")
            logger.debug(f"training_config for [{model_name}]: {training_config}")
            
            # 4. 生成 Hydra 参数覆盖列表
            hydra_overrides = self.config_converter.generate_hydra_overrides(
                override_config=override_config,
                model_name=model_name,
                base_config_path=script_full_path,  # 用于判断是否为 Python 配置文件
                general_overrides=general_overrides
            )
            logger.debug(f"hydra_overrides for [{model_name}]: {hydra_overrides}")  
            
            # 5. 构建完整命令
            command = ['python', script_full_path]
            command.extend(hydra_overrides)
            command.append(f"hydra.run.dir={output_dir}")
            
            # 6. 添加TensorBoard日志目录配置
            tb_log_dir = f"{output_dir}/tensorboard"
            
            # 使用logging配置（模型期望的配置路径）
            command.append(f"logging.save_dir={tb_log_dir}")
            command.append(f"logging.name=tensorboard")
            command.append(f"logging.version={os.path.basename(output_dir)}")
            
            # 同时添加logger配置作为备用
            command.append("+logger.name=tensorboard")
            command.append("+logger.save_dir=" + tb_log_dir)
            command.append("+logger.version=" + os.path.basename(output_dir))
            
            # 7. 添加SHAP分析输出路径配置 - 输出到实验变体目录下的shap_results
            variant_shap_dir = Path(output_dir) / "shap_results"
            try:
                variant_shap_dir.mkdir(parents=True, exist_ok=True)
                command.append(f"shap.output_dir={variant_shap_dir}")
                # command.append(f"shap.analysis_frequency=1") # 每1个epoch分析一次, Debug模式

                # command.append("+shap.save_plots=true")  # 默认保存SHAP图形
                logger.info(f"SHAP结果将保存到变体目录: {variant_shap_dir}")
            except Exception as e:
                logger.warning(f"无法创建SHAP目录 {variant_shap_dir}: {e}")
                logger.info("将使用默认tensorboard路径保存SHAP结果")
            
            # 从配置中获取日志间隔设置
            log_interval = self.exp_config.get('training', {}).get('log_every_n_steps', 50)
            flush_interval = self.exp_config.get('training', {}).get('flush_logs_every_n_steps', 100)
            val_check_interval = self.exp_config.get('training', {}).get('val_check_interval', 1.0)
            
            # 获取显示配置
            enable_progress_bar = self.exp_config.get('display', {}).get('enable_progress_bar', False)
            enable_model_summary = self.exp_config.get('display', {}).get('enable_model_summary', False)
            progress_bar_refresh_rate = self.exp_config.get('display', {}).get('progress_bar_refresh_rate', 0)
            
            command.append(f"training.log_every_n_steps={log_interval}")
            command.append(f"+training.val_check_interval={val_check_interval}")  # 每个epoch都验证
            command.append(f"+training.flush_logs_every_n_steps={flush_interval}")
            
            # 确保记录所有指标
            command.append(f"+trainer.log_every_n_steps={log_interval}")
            command.append(f"+trainer.flush_logs_every_n_steps={flush_interval}")
            command.append("+trainer.enable_checkpointing=False")  # 禁用检查点
            command.append(f"+trainer.enable_progress_bar={str(enable_progress_bar).lower()}")  # 进度条控制
            command.append("+trainer.logger=True")
            command.append("+trainer.default_root_dir=" + output_dir)
            
            # 额外的显示控制设置
            command.append(f"+trainer.enable_model_summary={str(enable_model_summary).lower()}")  # 模型总结控制
            command.append(f"+trainer.progress_bar_refresh_rate={progress_bar_refresh_rate}")  # 进度条刷新频率
            
            # 确保验证指标被记录
            command.append("+trainer.check_val_every_n_epoch=1")
            command.append("+trainer.val_check_interval=1.0")
            
            # 确保learning rate被记录
            command.append("+trainer.log_gpu_memory=None")
            command.append("+track_grad_norm=2.0")

            
            logger.info(f"++++++++++++ 构建训练命令 [{variant_id}]:++++++++++++++++\n\n {' '.join(command)}\n\n")
            return command
            
        except Exception as e:
            logger.error(f"构建训练命令失败: {e}")
            raise
    
    # 处理SHAP分析结果 
    def _process_shap_results(self, output_dir: str, variant_id: str, wandb_run):
        """
        处理SHAP分析结果
        
        Args:
            output_dir: 训练输出目录
            variant_id: 变体ID
            wandb_run: W&B运行对象
        """
        logger.info(f"处理{variant_id}的SHAP分析结果...")
        
        # 查找SHAP相关文件 - 首先尝试新的shap_results目录
        shap_dir = Path(output_dir) / "shap_results"
        if not shap_dir.exists():
            logger.error(f"未找到SHAP分析目录: {shap_dir}")
            return
        
        # 收集SHAP图像文件
        shap_images = []
        for file_path in shap_dir.glob("*.png"):
            shap_images.append(str(file_path))
            logger.info(f"找到SHAP图像: {file_path}")
        
        # 收集SHAP数据文件
        shap_data_files = []
        for file_path in shap_dir.glob("*.json"):
            shap_data_files.append(str(file_path))
            logger.info(f"找到SHAP数据文件: {file_path}")
        
        # 上传SHAP结果到W&B
        if wandb_run and shap_images and wandb is not None:
            # 上传SHAP图像
            for img_path in shap_images:
                img_name = Path(img_path).stem
                wandb_run.log({
                    f"shap/{img_name}": wandb.Image(img_path)
                })
                logger.info(f"已上传SHAP图像到W&B: {img_name}")
        elif wandb_run and shap_images and wandb is None:
            logger.warning("wandb模块不可用，无法上传SHAP图像")
        
        # 上传SHAP数据
        if wandb_run and shap_data_files:
            for data_path in shap_data_files:
                data_name = Path(data_path).stem
                try:
                    with open(data_path, 'r') as f:
                        shap_data = json.load(f)
                    wandb_run.log({
                        f"shap/{data_name}": shap_data
                    })
                    logger.info(f"已上传SHAP数据到W&B: {data_name}")
                except Exception as e:
                    logger.error(f"上传SHAP数据失败 {data_path}: {e}")
                    
        
        # 创建SHAP结果汇总
        shap_summary = {
            "variant_id": variant_id,
            "shap_images_count": len(shap_images),
            "shap_data_files_count": len(shap_data_files),
            "has_shap_analysis": len(shap_images) > 0 or len(shap_data_files) > 0
        }
        
        # 上传汇总到W&B
        if wandb_run:
            wandb_run.summary.update({
                "shap_summary": shap_summary
            })
        
        logger.info(f"SHAP结果处理完成: {shap_summary}")


    def execute_experiment(self, 
                          experiment_name: str,
                          experiment_config: Dict, 
                          variant_id: str,
                          variant: Dict,
                          output_dir: str) -> Dict:
        """
        执行单个实验
        
        Args:
            experiment_name: 实验名称
            experiment_config: 实验配置
            variant_id: 变体ID
            variant: 变体配置
            output_dir: 输出目录
            
        Returns:
            实验执行结果
        """
        import logging
        logger = logging.getLogger(__name__)
        
        
        try:
            # 构建训练命令
            training_command = self._build_training_command(experiment_config, variant_id, variant, output_dir)
            # 创建实验上下文
            experiment_context = {
                'experiment_id': variant_id,
                'experiment_name': experiment_name,
                'variant_id': variant_id,
                'variant': variant,
                'config': experiment_config,
                'output_dir': output_dir
            }
            
            # 实验执行完成后，解析和处理结果
            execution_result = self._execute_training_simplified(training_command, experiment_context)
            
            # 添加实验信息到结果
            execution_result.update({
                'experiment_name': experiment_name,
                'variant_id': variant_id,
                'variant': variant,
                'experiment_config': experiment_config,
                'output_dir': output_dir,
                'status': 'completed'
            })

            logger.info(f"执行实验完成: {execution_result}")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"执行实验失败 {variant_id}: {e}")
            import traceback
            traceback.print_exc()
            


class BatchExperimentExecutor:
    """批量实验执行器"""
    
    def __init__(self, workspace_root: str, max_parallel_jobs: int = 1, exp_config: Optional[Dict] = None):
        self.workspace_root = workspace_root
        self.max_parallel_jobs = max_parallel_jobs
        self.exp_config = exp_config or {}  # 保存exp_config到实例变量
        self.executor = ExperimentExecutor(workspace_root, exp_config=exp_config)
        # 将并行度信息传递给 executor
        self.executor.max_parallel_jobs = max_parallel_jobs
        self.results = []

    def execute_ablation_suite(self, 
                              experiment_config: Dict,
                              base_output_dir: str) -> tuple[List, str]:
        """
        执行完整的消融实验套件
        
        Args:
            experiment_config: 实验配置
            base_output_dir: 基础输出目录
            
        Returns:
            元组：(所有实验的结果列表, 实验套件目录路径)
        """

        # 方案2：检查是否为标准实验
        ablation_strategy = experiment_config.get('experiment', {}).get('ablation_strategy', None)

        all_ablation_variants = experiment_config.get('ablation_variants', {})
        if not all_ablation_variants:
            logger.warning("未找到任何消融变体，结束实验")
            return [], base_output_dir


        if ablation_strategy is None :
            logger.info(f"未指定消融策略，将执行标准实验")
            ablation_strategy = 'standard'
        
        if ablation_strategy.lower() == 'standard' :
            # 标准实验：只运行基线变体，不运行消融变体
            baseline_variants = {}
            for variant_id, variant in all_ablation_variants.items():
                if variant.get('baseline', False) or variant_id == 'BASE':
                    baseline_variants[variant_id] = variant
                    logger.info(f"标准实验：只运行基线变体 {variant_id}")
                    break  # 只运行第一个基线变体
            
            if not baseline_variants:
                logger.error("未找到基线变体，结束实验")
                return [], base_output_dir

            ablation_variants = baseline_variants
            logger.info(f"标准实验模式：将运行 {len(ablation_variants)} 个基线实验")
        else:

            # 获取使能开关
            enable_config = experiment_config.get('experiment', {}).get('enable', {})
            
            # 过滤实验变体
            filtered_variants = {}
            
            for variant_id, variant in all_ablation_variants.items():
                variant_type = variant.get('type')
                
                # 检查该类型是否使能（默认使能，向后兼容）
                if enable_config.get(variant_type, True):
                    filtered_variants[variant_id] = variant
                    logger.info(f"✓ 使能实验: {variant_id} ({variant_type})")
                else:
                    logger.info(f"✗ 跳过实验: {variant_id} ({variant_type} - 未使能)")
        
            ablation_variants = filtered_variants
            logger.info(f"消融实验模式：将运行 {len(ablation_variants)} 个实验")

        
        if not ablation_variants:
            logger.warning("没有使能的实验变体！,结束实验")
            return [], ""
        
        logger.info(f"将执行 {len(ablation_variants)} 个使能的实验变体")
        
        experiment_name = f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建输出目录
        output_base = Path(base_output_dir) / experiment_name
        output_base.mkdir(parents=True, exist_ok=True)
        
        # 保存实验配置
        config_file = output_base / 'experiment_config.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(experiment_config, f, default_flow_style=False, allow_unicode=True)
        
        results = []
        
        if self.max_parallel_jobs == 1:
            # 串行执行
            for i, (variant_id, variant) in enumerate(ablation_variants.items()):
                
                experiment_output_dir = output_base / f"{variant_id}_{variant.get('name', '')}"
                
                logger.info(f"\n\n ============================ Executing experiment {i+1}/{len(ablation_variants)}: {variant_id} ==========================\n\n")

                result = self.executor.execute_experiment(
                    experiment_name, experiment_config, variant_id, variant, str(experiment_output_dir)
                )
                
                results.append(result)
                
                # # 保存中间结果
                # self._save_intermediate_results(results, output_base)
                
        else:
            # 并行执行
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
                futures = {}
                
                for i, (variant_id, variant) in enumerate(ablation_variants.items()):
                    experiment_output_dir = output_base / f"{variant_id}_{variant.get('name', '')}"
                    
                    logger.info(f"\n\n ============================ [Batch Mode]Executing experiment {i+1}/{len(ablation_variants)}: {variant_id}==========================\n\n")

                    future = executor.submit(
                        self.executor.execute_experiment,
                        experiment_name, experiment_config, variant_id, variant, str(experiment_output_dir)
                    )
                    futures[future] = (variant_id, variant, experiment_output_dir)
                
                for future in as_completed(futures):
                    variant_id, variant, experiment_output_dir = futures[future]
                    result = future.result()
                    results.append(result)
                    
                    logger.info(f"Completed experiment: {variant_id} - {result.get('status')}")


        # 执行完成后(无论并行还是串行)，统一执行 W&B 上传
        logger.info(f"开始统一上传 {len(results)} 个实验的 W&B 数据")

        # 初始化 WandB 运行
        wandb_run = None
        if self.executor.wandb_integration is not None:
            for result in results:
                if result.get('status') == 'completed' and 'parsed_metrics' in result:
                    variant_id = result.get('variant_id')
                    variant = result.get('variant')
                    experiment_name = result.get('experiment_name')

                    wandb_run = None
                    
                    try:
                        # 为每个成功完成的实验初始化 WandB 运行
                        wandb_run = self.executor.wandb_integration.init_experiment_run(
                            experiment_name=experiment_name,
                            experiment_config=result.get('experiment_config', {}),
                            variant_id=variant_id,
                            ablation_variant=variant
                        )
                        
                        # 上传数据
                        self.executor._upload_to_wandb(wandb_run, result['parsed_metrics'])
                        
                        logger.info(f"成功上传实验 {variant_id} 的 W&B 数据")

                        # 添加SHAP结果处理
                        self.executor._process_shap_results(result.get('output_dir'), variant_id, wandb_run)

                        logger.info(f"成功上传实验 {variant_id} 的 W&B shap 数据")


                        # 关闭运行
                        self.executor.wandb_integration.finish_run()
                        
                    except Exception as e:
                        logger.error(f"上传实验 {variant_id} 的 W&B 数据失败: {e}")
                        import traceback
                        traceback.print_exc()
            
            logger.info("所有实验的 W&B 数据上传完成")
        else:
            logger.info("WandB 集成未启用")
        
        # 保存最终结果
        self._save_final_results(results, output_base)
        
       
        # 生成实验报告
        self._generate_experiment_report(results, output_base)
        
        self.results = results
        return results, str(output_base)

    def _save_intermediate_results(self, results: List, output_dir: Path):
        """保存中间结果"""
        import json
        results_file = output_dir / 'intermediate_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def _save_final_results(self, results: List, output_dir: Path):
        """保存最终结果"""
        import json
        results_file = output_dir / 'final_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    
    def _generate_experiment_report(self, results: List, output_dir: Path):
        """生成实验报告"""
        # 统计信息
        total_experiments = len(results)
        completed_experiments = sum(1 for r in results if r.get('status') == 'completed')
        failed_experiments = total_experiments - completed_experiments
        
        report_lines = [
            "# 消融实验报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**实验总数**: {total_experiments}",
            f"**成功实验**: {completed_experiments}",
            f"**失败实验**: {failed_experiments}",
            "",
            "## 实验结果汇总",
            "",
            # 改进的表格格式，增加对齐
            "|   实验ID                 | 状态          | 最终准确率    | 最终F1分数    | 训练时间  |",
        ]
        
        for result in results:
            ablation_id = result.get('variant_id', 'N/A')
            # 获取消融变体的名称，如果没有则使用实验名称

            parsed_metrics = result.get('parsed_metrics', {})

            variant_name =  result.get('experiment_name') or 'N/A'
            status = result.get('status') or 'N/A'
            
            # 尝试从多个字段获取准确率和F1分数
            accuracy = None
            f1_score = None
            
            # 首先从 test_results 中获取 (来源于training.log中表格的输出)
            test_results = parsed_metrics.get('test_results', {})
            if test_results:
                accuracy = test_results.get('test_avg_precision') or test_results.get('test_accuracy')
                f1_score = test_results.get('test_f1')
            
            # 如果 test_results 中没有，尝试其他字段
            if accuracy is None:
               logger.warning(f"accuracy is None from test_results, try to get from other fields??")

            if f1_score is None:
                logger.warning(f"f1_score is None from test_results, try to get from other fields??")
            
            # 尝试获取训练时间
            # 首先从解析的指标中获取训练时长
            duration_str = "N/A"  # 默认值
            duration = result.get('duration')
            
            if duration is not None and isinstance(duration, (int, float)) and duration > 0:
                duration_str = f"{duration:.1f}s"
                logger.info(f"duration from parsed metrics: {duration_str}")
            else:
                duration_str = "N/A"
                logger.warning(f"duration not available, using N/A")
            
            # 格式化数值输出
            if accuracy is None:
                accuracy = "N/A"
            elif isinstance(accuracy, (int, float)):
                accuracy = f"{accuracy:.4f}"
            else:
                accuracy = str(accuracy)
                
            if f1_score is None:
                f1_score = "N/A"
            elif isinstance(f1_score, (int, float)):
                f1_score = f"{f1_score:.4f}"
            else:
                f1_score = str(f1_score)
            
            # 状态着色（使用Markdown）
            status_emoji = {
                'completed': '✅',
                'failed': '❌', 
                'running': '🔄',
                'skipped': '⏭️'
            }.get(status, '')
            status_display = f"{status_emoji} {status}" if status_emoji else (status or 'N/A')
            
            report_lines.append(
                f"|  {ablation_id:<24} |  {status_display:<12} | {accuracy:<12} | {f1_score:<12} | {duration_str:<9} |"
            )
        
        # 添加实验详情部分
        report_lines.extend([
            "",
            "## 实验详细信息",
            ""
        ])
        
        for result in results:
            if result.get('status') == 'failed':
                ablation_id = result.get('variant_id', 'N/A')
                error = result.get('error', '未知错误')
                report_lines.extend([
                    f"### {ablation_id} - 失败详情",
                    f"**错误信息**: {error}",
                    ""
                ])
        
        report_file = output_dir / 'experiment_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Experiment report generated: {report_file}")
        return report_lines