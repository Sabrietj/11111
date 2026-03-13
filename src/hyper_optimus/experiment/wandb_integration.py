"""
W&B集成模块 - 重构版本
负责消融实验的W&B数据上传和实时指标采集
"""

import wandb
import re
import yaml
import threading
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from datetime import datetime
import subprocess
import json
import fcntl
import select
import sys
import os

# 使用统一的日志配置
try:
    from utils.logging_config import setup_preset_logging
except ImportError:
    import logging
    def setup_preset_logging(log_level):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

import logging
logger = setup_preset_logging(log_level=logging.DEBUG)


class WandBIntegration:
    """W&B集成管理器"""
    
    def __init__(self, 
                 project_name: str = "multiview-ablation-studies",
                 entity: Optional[str] = None):
        self.project_name = project_name
        self.entity = entity
        self.current_run = None
        self.experiment_context = {}

    def init_experiment_run(self, 
                           experiment_name: str, 
                           experiment_config: Dict[str, Any],
                           variant_id: str,
                           ablation_variant: Dict[str, Any]) -> wandb.run:
        """
        初始化W&B实验运行
        
        Args:
            experiment_name: 实验名称
            experiment_config: 实验配置
            ablation_variant: 消融实验变体配置
            
        Returns:
            W&B运行对象
        """

        variant_name = ablation_variant.get('name', 'unknown')
        # 构建运行名称
        # experiment_name : 来自 experiment_executor.py:777 的 f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # variant_id : 消融变体ID（如 FT1, FU1, LS1）
        # variant_name : 消融变体名称（如 FT1_143025, FU1_143025, LS1_143025）
        # 最终运行名称：suite_143025_FT1_143025
        run_name = f"{experiment_name}_{variant_id}_{variant_name}"
        
        # 保存实验上下文
        self.experiment_context = {
            'experiment_name': experiment_name,
            'ablation_id': variant_id,
            'ablation_type': ablation_variant.get('type'),
            'model_type': experiment_config.get('experiment', {}).get('model_name', 'unknown'),
            'description': ablation_variant.get('description', ''),
            'baseline': ablation_variant.get('baseline', False)
        }
        
        # 配置标签
        tags = [
            ablation_variant.get('type', 'unknown'),
            self.experiment_context['model_type'],
            variant_id
        ]
        if ablation_variant.get('baseline', False):
            tags.append('baseline')
        
        # 设置环境变量控制W&B行为 - 更宽松的设置确保数据上传
        import os
        os.environ['WANDB_SILENT'] = 'false'  # 启用日志输出
        os.environ['WANDB_SAVE_CODE'] = 'false'
        os.environ['WANDB_SAVE_REQUIREMENTS'] = 'false'
        os.environ['WANDB_DISABLE_GIT'] = 'true'
        
        try:
            # 启动W&B运行
            self.current_run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                tags=tags,
                config={
                    'experiment_config': experiment_config,
                    'ablation_variant': ablation_variant,
                    'experiment_context': self.experiment_context
                },
                reinit=True,
                settings=wandb.Settings(
                    console='off',  # 禁用控制台输出
                    save_code=False,  # 不自动保存代码文件
                    git_root=None,  # 不收集git信息
                    job_name=None,  # 不设置job名称
                )
            )
            
            logger.info(f"Successfully initialized W&B run: {run_name}")
            logger.info(f"W&B run URL: {self.current_run.url if hasattr(self.current_run, 'url') else 'N/A'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {e}")
            # 创建一个模拟的run对象，避免程序崩溃
            def mock_finish(*args, **kwargs):
                return None
            
            def mock_log(*args, **kwargs):
                return None
            
            class MockConfig:
                def update(self, *args, **kwargs):
                    pass
                
            self.current_run = type('MockRun', (), {
                'log': mock_log,
                'finish': mock_finish,
                'summary': type('MockSummary', {'update': lambda self, *args, **kwargs: None})(),
                'config': MockConfig(),
                'url': None
            })()
            
        return self.current_run

    def log_experiment_config(self, config: Dict[str, Any]):
        """记录实验配置"""
        if self.current_run:
            self.current_run.config.update(config)

    def log_epoch_metrics(self, 
                         epoch: int, 
                         metrics: Dict[str, Any],
                         dataset_type: str = "train"):
        """
        记录每个epoch的指标
        
        Args:
            epoch: epoch编号
            metrics: 指标字典
            dataset_type: 数据集类型 (train/val/test)
        """
        if not self.current_run:
            logger.warning("No active W&B run to log metrics")
            return
        
        try:
            # 构建指标名称
            epoch_metrics = {
                'epoch': epoch,
                **self.experiment_context
            }
            
            for metric_name, value in metrics.items():
                prefixed_name = f"{dataset_type}/{metric_name}"
                epoch_metrics[prefixed_name] = value
            
            # 立即同步记录到W&B 
            self.current_run.log(epoch_metrics, commit=True)
            
        except Exception as e:
            logger.error(f"Failed to log metrics to W&B: {e}")
            # 尝试本地备份
            try:
                import json
                backup_data = {
                    'timestamp': datetime.now().isoformat(),
                    'epoch': epoch,
                    'dataset_type': dataset_type,
                    'metrics': metrics,
                    'error': str(e)
                }
                backup_file = f"wandb_metrics_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_file, 'w') as f:
                    json.dump(backup_data, f, indent=2)
                logger.info(f"Saved metrics backup to {backup_file}")
            except Exception as backup_e:
                logger.error(f"Failed to save backup metrics: {backup_e}")

    def log_final_results(self, results: Dict[str, Any]):
        """记录最终实验结果"""
        if not self.current_run:
            logger.warning("No active W&B run to log final results")
            return
        
        try:
            # 记录到summary
            summary_results = {
                **results,
                **self.experiment_context,
                'completed_at': datetime.now().isoformat()
            }
            
            # 确保关键数值指标被正确记录
            numeric_metrics = {}
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    numeric_metrics[f"final_{key}"] = value
                elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    try:
                        numeric_metrics[f"final_{key}"] = float(value)
                    except ValueError:
                        pass
            
            # 合并所有结果
            final_summary = {**summary_results, **numeric_metrics}
            
            self.current_run.summary.update(final_summary)
            logger.info(f"Successfully logged final results to W&B: {list(numeric_metrics.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to log final results to W&B: {e}")
            # 尝试本地备份
            try:
                import json
                backup_data = {
                    'timestamp': datetime.now().isoformat(),
                    'results': results,
                    'experiment_context': self.experiment_context,
                    'error': str(e)
                }
                backup_file = f"wandb_final_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_file, 'w') as f:
                    json.dump(backup_data, f, indent=2)
                logger.info(f"Saved final results backup to {backup_file}")
            except Exception as backup_e:
                logger.error(f"Failed to save final results backup: {backup_e}")

    def finish_run(self):
        """完成W&B运行"""
        if self.current_run:
            self.current_run.finish()
            self.current_run = None
            logger.info("Finished W&B run")


class SimplifiedMetricsCollector:
    """简化的指标采集器 - 专门负责从共享缓冲区解析指标"""
    
    def __init__(self, wandb_integration: WandBIntegration):
        self.wandb_integration = wandb_integration
        # 多种epoch匹配模式
        self.epoch_patterns = [
            re.compile(r'Epoch\s*(\d+):'),         # 实际格式: "Epoch 1:"
            re.compile(r'Epoch:\s*(\d+)/\d+.*'),  # 标准格式
            re.compile(r'epoch:\s*(\d+)'),         # 小写格式
            re.compile(r'\b(\d+)\b.*epoch'),       # 包含epoch的行
        ]
        # 指标匹配模式（精简版）
        self.metrics_patterns = [
            # 标准格式: train_loss: 0.1234
            re.compile(r'(train|val|test)_(loss|acc|accuracy|f1|precision|recall|lr|learning_rate|recon_loss|class_loss|total_loss):\s*([\d\.e-]+)'),
            # 等号格式: train_loss = 0.1234
            re.compile(r'(train|val|test)_(loss|acc|accuracy|f1|precision|recall|lr|learning_rate|recon_loss|class_loss|total_loss)\s*=\s*([\d\.e-]+)'),
            # PyTorch Lightning 格式
            re.compile(r'.*val_loss:\s*([\d\.e-]+).*'),
            re.compile(r'.*val_f1:\s*([\d\.e-]+).*'),
            re.compile(r'.*val_accuracy:\s*([\d\.e-]+).*'),
            re.compile(r'.*train_loss:\s*([\d\.e-]+).*'),
            re.compile(r'.*train_f1:\s*([\d\.e-]+).*'),
            re.compile(r'.*train_accuracy:\s*([\d\.e-]+).*'),
            # Test 指标
            re.compile(r'.*test_loss:\s*([\d\.e-]+).*'),
            re.compile(r'.*test_accuracy:\s*([\d\.e-]+).*'),
            re.compile(r'.*test_f1:\s*([\d\.e-]+).*'),
            re.compile(r'.*test_precision:\s*([\d\.e-]+).*'),
            re.compile(r'.*test_recall:\s*([\d\.e-]+).*'),
            re.compile(r'.*test_auc:\s*([\d\.e-]+).*'),
            # 表格格式
            re.compile(r'│\s*test_(\w+)\s*│\s*([\d\.e-]+)\s*│'),
            re.compile(r'test_(\w+).*\│\s*([\d\.e-]+)\s*\|'),
        ]
        
        self.current_epoch = 0
        self.pending_metrics = []
    
    def extract_epoch(self, line: str) -> Optional[int]:
        """从行中提取epoch信息"""
        for pattern in self.epoch_patterns:
            try:
                match = pattern.search(line)
                if match:
                    return int(match.group(1))
            except (ValueError, AttributeError):
                continue
        return None
    
    def extract_metrics(self, line: str) -> List[Tuple[str, str, float]]:
        """从行中提取指标，返回 (dataset, metric, value) 列表"""
        metrics = []
        line = line.strip()
        
        for pattern in self.metrics_patterns:
            try:
                matches = pattern.findall(line)
                for match in matches:
                    if not match or not all(match):
                        continue
                    
                    # 处理不同长度的匹配结果
                    if len(match) == 3:  # dataset, metric, value
                        dataset, metric, value = match
                    elif len(match) == 2:  # metric, value
                        metric, value = match
                        # 从行中推断dataset类型
                        if 'val_' in line or 'validation' in line.lower():
                            dataset = 'val'
                        elif 'train_' in line or 'training' in line.lower():
                            dataset = 'train'
                        else:
                            dataset = 'test'
                    else:
                        continue
                    
                    # 验证指标值
                    try:
                        metric_value = float(value)
                        # 验证指标值的合理性
                        loss_metrics = ['loss', 'recon_loss', 'class_loss', 'total_loss']
                        if not (0 <= metric_value <= 1000) and metric not in loss_metrics:
                            continue
                        if metric == 'learning_rate' and not (1e-8 <= metric_value <= 1.0):
                            continue
                        
                        metrics.append((dataset, metric, metric_value))
                    except (ValueError, TypeError):
                        continue
                        
            except Exception:
                continue
                
        return metrics
    
    def monitor_shared_buffer(self, shared_buffer, process: subprocess.Popen):
        """监控共享缓冲区，解析指标并上传到W&B"""
        logger.info("Starting  metrics monitoring")
        
        while True:
            # 获取未处理的行
            unprocessed = shared_buffer.get_unprocessed_lines()
            if not unprocessed:
                # 检查是否应该退出
                if shared_buffer.is_process_finished(process):
                    # 上传最后的指标
                    if self.pending_metrics:
                        self._upload_pending_metrics()
                    break
                time.sleep(0.1)  # 等待新数据
                continue
            
            processed_indices = []
            epoch_changed = False
            
            for index, line in unprocessed:
                # 检查epoch变化
                new_epoch = self.extract_epoch(line)
                if new_epoch and new_epoch != self.current_epoch:
                    # 上传上一个epoch的指标
                    if self.pending_metrics:
                        self._upload_pending_metrics()
                    self.current_epoch = new_epoch
                    epoch_changed = True
                
                # 提取指标
                metrics = self.extract_metrics(line)
                if metrics:
                    self.pending_metrics.extend(metrics)
                
                processed_indices.append(index)
            
            # 标记已处理
            shared_buffer.mark_processed(processed_indices)
            
            # 如果epoch完成（检测到新epoch开始且不是第一个epoch），上传指标
            if epoch_changed and self.current_epoch > 0:
                self._upload_pending_metrics()
        
        # 最终处理：查找测试指标
        self._process_final_test_metrics(shared_buffer)
        logger.info(" Metrics monitoring completed")
    
    def _upload_pending_metrics(self):
        """上传待处理的指标到W&B"""
        if not self.pending_metrics:
            return
        
        # 按数据集分组指标
        metrics_by_dataset = {}
        for dataset, metric, value in self.pending_metrics:
            if dataset not in metrics_by_dataset:
                metrics_by_dataset[dataset] = {}
            metrics_by_dataset[dataset][metric] = value
        
        # 上传每个数据集的指标
        for dataset, metrics in metrics_by_dataset.items():
            try:
                self.wandb_integration.log_epoch_metrics(
                    epoch=self.current_epoch,
                    metrics=metrics,
                    dataset_type=dataset
                )
                logger.info(f"Uploaded {len(metrics)} {dataset} metrics for epoch {self.current_epoch}")
            except Exception as e:
                logger.error(f"Failed to upload {dataset} metrics: {e}")
        
        self.pending_metrics.clear()
    
    def _process_final_test_metrics(self, shared_buffer):
        """处理最终测试指标"""
        all_lines = shared_buffer.get_all_lines()
        test_metrics = []
        
        for line in all_lines:
            # 专门查找测试指标
            if any(keyword in line.lower() for keyword in ['test_', 'test:', 'testing']):
                metrics = self.extract_metrics(line)
                test_metrics.extend(metrics)
        
        if test_metrics:
            # 使用epoch=-1表示这是最终测试结果
            self.current_epoch = -1
            self.pending_metrics = test_metrics
            self._upload_pending_metrics()
            logger.info(f"Processed {len(test_metrics)} final test metrics")


# 向后兼容性支持
class RealTimeMetricsCollector:
    """实时指标采集器 - 通过解析训练日志（保留向后兼容）"""
    
    def __init__(self, wandb_integration: WandBIntegration):
        logger.warning("RealTimeMetricsCollector is deprecated, using SimplifiedMetricsCollector instead")
        self.simplified = SimplifiedMetricsCollector(wandb_integration)
    
    def monitor_training_process(self, process: subprocess.Popen):
        """兼容性方法"""
        # 创建一个模拟的共享缓冲区
        from .experiment_executor import SharedOutputBuffer
        shared_buffer = SharedOutputBuffer()
        
        # 将进程输出添加到缓冲区
        for line in iter(process.stdout.readline, ''):
            if line:
                shared_buffer.add_line(line)
        
        # 使用简化的监控器
        self.simplified.monitor_shared_buffer(shared_buffer, process)


class CallbackBasedMetricsCollector:
    """基于回调的指标采集器 - 适用于PyTorch Lightning"""
    
    def __init__(self, wandb_integration: WandBIntegration):
        self.wandb_integration = wandb_integration
    
    def create_pytorch_lightning_callback(self):
        """创建PyTorch Lightning的W&B回调"""
        try:
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import Callback
            
            class WandBAblationCallback(Callback):
                def __init__(self, wandb_integration):
                    super().__init__()
                    self.wandb_integration = wandb_integration
                
                def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
                    """训练epoch结束时调用"""
                    if hasattr(pl_module, 'log_dict') and trainer.callback_metrics:
                        # 获取训练指标
                        train_metrics = {}
                        for name, value in trainer.callback_metrics.items():
                            if name.startswith('train_'):
                                train_metrics[name[6:]] = value.item() if hasattr(value, 'item') else value
                        
                        if train_metrics:
                            self.wandb_integration.log_epoch_metrics(
                                epoch=trainer.current_epoch,
                                metrics=train_metrics,
                                dataset_type="train"
                            )
                
                def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
                    """验证epoch结束时调用"""
                    if hasattr(pl_module, 'log_dict') and trainer.callback_metrics:
                        # 获取验证指标
                        val_metrics = {}
                        for name, value in trainer.callback_metrics.items():
                            if name.startswith('val_'):
                                val_metrics[name[4:]] = value.item() if hasattr(value, 'item') else value
                        
                        if val_metrics:
                            self.wandb_integration.log_epoch_metrics(
                                epoch=trainer.current_epoch,
                                metrics=val_metrics,
                                dataset_type="val"
                            )
                
                def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
                    """测试epoch结束时调用"""
                    if hasattr(pl_module, 'log_dict') and trainer.callback_metrics:
                        # 获取测试指标
                        test_metrics = {}
                        for name, value in trainer.callback_metrics.items():
                            if name.startswith('test_'):
                                test_metrics[name[5:]] = value.item() if hasattr(value, 'item') else value
                        
                        if test_metrics:
                            self.wandb_integration.log_epoch_metrics(
                                epoch=trainer.current_epoch,
                                metrics=test_metrics,
                                dataset_type="test"
                            )
            
            return WandBAblationCallback(self.wandb_integration)
            
        except ImportError:
            logger.warning("PyTorch Lightning not available, callback-based metrics collection disabled")
            return None


class BatchWandBUploader:
    """批量W&B上传器 - 用于延迟上传消融实验结果"""
    
    def __init__(self, project_name: str = "multiview-ablation-studies", entity: Optional[str] = None):
        self.project_name = project_name
        self.entity = entity
        self.uploaded_experiments = []
        
    def upload_experiment_results(self, 
                                 experiment_results: List[Dict[str, Any]], 
                                 experiment_suite_name: str):
        """批量上传实验结果到W&B
        
        Args:
            experiment_results: 消融实验结果列表
            experiment_suite_name: 实验套件名称
        """
        logger.info(f"开始批量上传 {len(experiment_results)} 个消融实验结果到W&B")
        
        for i, result in enumerate(experiment_results):
            try:
                # 提取实验信息
                ablation_id = result.get('ablation_id', f'experiment_{i}')
                ablation_config = result.get('ablation_config', {})
                variant_name = ablation_config.get('name', ablation_id)
                
                # 获取或创建默认的W&B配置
                wandb_config = self._get_default_wandb_config(result)
                
                # 初始化W&B运行
                run_name = f"{experiment_suite_name}_{ablation_id}_{variant_name}_upload"
                
                run = wandb.init(
                    project=self.project_name,
                    entity=self.entity,
                    name=run_name,
                    config=wandb_config,
                    tags=["ablation_upload", "batch_upload", ablation_config.get('type', 'unknown')],
                    reinit=True,
                    settings=wandb.Settings(
                        console='off',
                        save_code=False,
                        git_root=None,
                    )
                )
                
                # 上传训练指标
                self._upload_training_metrics(run, result)
                
                # 上传最终结果
                self._upload_final_results(run, result)
                
                # 上传系统信息
                self._upload_system_info(run, result)
                
                run.finish()
                
                self.uploaded_experiments.append({
                    'ablation_id': ablation_id,
                    'run_name': run_name,
                    'run_url': run.url if hasattr(run, 'url') else 'N/A',
                    'upload_time': datetime.now().isoformat()
                })
                
                logger.info(f"成功上传实验 {ablation_id} 到W&B: {run.url if hasattr(run, 'url') else 'N/A'}")
                
            except Exception as e:
                logger.error(f"上传实验 {ablation_id} 到W&B失败: {e}")
                continue
        
        # 保存上传记录
        self._save_upload_record(experiment_suite_name)
        logger.info(f"批量上传完成，成功上传 {len(self.uploaded_experiments)} 个实验")
    
    def _get_default_wandb_config(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """获取默认的W&B配置信息"""
        # 从项目中读取默认配置
        try:
            import sys
            import os
            sys.path.append('/data/qinyf/code-multiview-network-traffic-classification-model/src')
            from hyper_optimus.configs.config import get_project_config
            
            project_config = get_project_config()
            wandb_config = {
                'project': project_config.get('experiment', {}).get('wandb_project', self.project_name),
                'model_name': project_config.get('experiment', {}).get('model_name', 'unknown'),
                'workspace_root': '/data/qinyf/code-multiview-network-traffic-classification-model'
            }
        except Exception as e:
            logger.warning(f"无法获取项目配置，使用默认配置: {e}")
            wandb_config = {
                'project': self.project_name,
                'model_name': 'unknown',
                'workspace_root': '/data/qinyf/code-multiview-network-traffic-classification-model'
            }
        
        # 添加实验特定配置
        ablation_config = result.get('ablation_config', {})
        wandb_config.update({
            'ablation_id': result.get('ablation_id'),
            'ablation_type': ablation_config.get('type'),
            'ablation_name': ablation_config.get('name'),
            'ablation_description': ablation_config.get('description', ''),
            'baseline': ablation_config.get('baseline', False),
            'status': result.get('status', 'unknown')
        })
        
        return wandb_config
    
    def _upload_training_metrics(self, run, result: Dict[str, Any]):
        """上传训练指标"""
        parsed_metrics = result.get('parsed_metrics', {})
        epoch_metrics = parsed_metrics.get('epoch_metrics', [])
        
        for epoch_data in epoch_metrics:
            epoch = epoch_data.get('epoch', 0)
            
            # 分离训练和验证指标
            train_metrics = {}
            val_metrics = {}
            
            for key, value in epoch_data.items():
                if key.startswith('train_'):
                    train_metrics[key[6:]] = value
                elif key.startswith('val_'):
                    val_metrics[key[4:]] = value
                elif key == 'epoch':
                    continue
                else:
                    # 未分类的指标归类到训练指标
                    train_metrics[key] = value
            
            # 上传训练指标
            if train_metrics:
                run.log({f"train/{k}": v for k, v in train_metrics.items()}, step=epoch)
            
            # 上传验证指标
            if val_metrics:
                run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
    
    def _upload_final_results(self, run, result: Dict[str, Any]):
        """上传最终实验结果"""
        final_results = {}
        
        # 提取关键指标
        key_metrics = [
            'final_test_accuracy', 'final_val_accuracy', 'best_accuracy',
            'final_test_f1', 'final_val_f1', 'best_f1',
            'final_test_loss', 'final_val_loss', 'best_loss'
        ]
        
        for metric in key_metrics:
            if metric in result:
                final_results[metric] = result[metric]
        
        # 添加执行信息
        final_results.update({
            'return_code': result.get('return_code'),
            'duration': result.get('duration', 0),
            'status': result.get('status', 'unknown')
        })
        
        # 如果有最终的epoch指标，记录最佳性能
        parsed_metrics = result.get('parsed_metrics', {})
        epoch_metrics = parsed_metrics.get('epoch_metrics', [])
        if epoch_metrics:
            # 找到最佳验证准确率
            best_val_acc = 0
            best_epoch = 0
            for epoch_data in epoch_metrics:
                val_acc = epoch_data.get('val_accuracy', 0)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch_data.get('epoch', 0)
            
            final_results['best_val_accuracy_overall'] = best_val_acc
            final_results['best_epoch'] = best_epoch
        
        if final_results:
            run.log(final_results)
    
    def _upload_system_info(self, run, result: Dict[str, Any]):
        """上传系统信息"""
        system_info = {
            'upload_time': datetime.now().isoformat(),
            'experiment_type': 'ablation_study',
            'upload_mode': 'batch_upload'
        }
        
        # 添加执行环境信息
        if 'stdout' in result and result['stdout']:
            # 从stdout中提取系统信息
            stdout = result['stdout']
            if 'CUDA' in stdout or 'GPU' in stdout:
                system_info['gpu_available'] = True
            else:
                system_info['gpu_available'] = False
        
        run.log(system_info)
    
    def _save_upload_record(self, experiment_suite_name: str):
        """保存上传记录"""
        try:
            import json
            record_file = f"wandb_upload_record_{experiment_suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(record_file, 'w') as f:
                json.dump({
                    'experiment_suite': experiment_suite_name,
                    'upload_time': datetime.now().isoformat(),
                    'total_uploaded': len(self.uploaded_experiments),
                    'experiments': self.uploaded_experiments
                }, f, indent=2)
            
            logger.info(f"上传记录已保存到 {record_file}")
            
        except Exception as e:
            logger.error(f"保存上传记录失败: {e}")


def manual_upload_ablation_results(results_file: str, 
                                   project_name: Optional[str] = None,
                                   entity: Optional[str] = None) -> bool:
    """
    手动上传消融实验结果到W&B
    
    Args:
        results_file: 结果文件路径（JSON格式）
        project_name: W&B项目名称（可选，默认使用工程配置）
        entity: W&B实体名称（可选）
        
    Returns:
        上传是否成功
    """
    try:
        import json
        
        # 加载结果文件
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # 获取项目配置
        if project_name is None:
            try:
                import sys
                import os
                sys.path.append('/data/qinyf/code-multiview-network-traffic-classification-model/src')
                from hyper_optimus.configs.config import get_project_config
                project_config = get_project_config()
                project_name = project_config.get('experiment', {}).get('wandb_project', 'multiview-ablation-studies')
            except Exception:
                project_name = 'multiview-ablation-studies'
        
        # 从文件名推导实验套件名称
        import os
        suite_name = os.path.basename(os.path.dirname(results_file))
        if not suite_name:
            suite_name = f"manual_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"开始手动上传消融实验结果: {results_file}")
        logger.info(f"项目: {project_name}, 套件: {suite_name}")
        
        # 创建批量上传器
        uploader = BatchWandBUploader(project_name=project_name, entity=entity)
        
        # 执行批量上传
        uploader.upload_experiment_results(results, suite_name)
        
        logger.info("手动上传完成")
        return True
        
    except Exception as e:
        logger.error(f"手动上传失败: {e}")
        return False


def test_wandb_connection(project_name: str = "test", entity: Optional[str] = None) -> bool:
    """
    测试W&B连接是否正常
    
    Args:
        project_name: 测试项目名称
        entity: W&B实体名称
        
    Returns:
        连接是否成功
    """
    try:
        test_run = wandb.init(
            project=project_name,
            entity=entity,
            name="connection_test",
            tags=["test"],
            reinit=True,
            settings=wandb.Settings(
                console='off',
                save_code=False,
            )
        )
        
        # 测试日志记录
        test_run.log({
            "test_metric": 1.0,
            "test_timestamp": datetime.now().isoformat()
        })
        
        test_run.finish()
        logger.info("W&B connection test successful")
        return True
        
    except Exception as e:
        logger.error(f"W&B connection test failed: {e}")
        return False


class MetricsAggregator:
    """指标聚合器"""
    
    def __init__(self):
        self.epoch_metrics = {}
        self.final_results = {}
    
    def add_epoch_metrics(self, 
                         experiment_name: str,
                         epoch: int, 
                         metrics: Dict[str, Any],
                         dataset_type: str = "train"):
        """添加epoch指标"""
        key = f"{experiment_name}_{dataset_type}"
        if key not in self.epoch_metrics:
            self.epoch_metrics[key] = []
        
        metrics_with_epoch = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.epoch_metrics[key].append(metrics_with_epoch)
    
    def set_final_results(self, experiment_name: str, results: Dict[str, Any]):
        """设置最终结果"""
        self.final_results[experiment_name] = {
            'timestamp': datetime.now().isoformat(),
            **results
        }
    
    def export_metrics_to_file(self, output_path: str):
        """导出指标到文件"""
        export_data = {
            'epoch_metrics': self.epoch_metrics,
            'final_results': self.final_results,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported metrics to {output_path}")
    
    def get_best_metrics(self, experiment_name: str, metric_name: str = "f1_score") -> Dict[str, Any]:
        """获取最佳指标"""
        best_val = {'value': 0, 'epoch': 0}
        best_train = {'value': 0, 'epoch': 0}
        
        val_key = f"{experiment_name}_val"
        train_key = f"{experiment_name}_train"
        
        if val_key in self.epoch_metrics:
            for epoch_data in self.epoch_metrics[val_key]:
                if metric_name in epoch_data and epoch_data[metric_name] > best_val['value']:
                    best_val = {
                        'value': epoch_data[metric_name],
                        'epoch': epoch_data['epoch']
                    }
        
        if train_key in self.epoch_metrics:
            for epoch_data in self.epoch_metrics[train_key]:
                if metric_name in epoch_data and epoch_data[metric_name] > best_train['value']:
                    best_train = {
                        'value': epoch_data[metric_name],
                        'epoch': epoch_data['epoch']
                    }
        
        return {
            'best_val': best_val,
            'best_train': best_train
        }


def safe_wandb_log(wandb_run: wandb.run, 
                   metrics: Dict[str, Any], 
                   max_retries: int = 3) -> bool:
    """
    安全的W&B日志上传，支持重试
    
    Args:
        wandb_run: W&B运行对象
        metrics: 要上传的指标
        max_retries: 最大重试次数
        
    Returns:
        是否成功上传
    """
    for attempt in range(max_retries):
        try:
            wandb_run.log(metrics)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to log to W&B after {max_retries} attempts: {e}")
                # 保存到本地文件作为备份
                backup_path = f"wandb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                try:
                    with open(backup_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    logger.info(f"Saved metrics backup to {backup_path}")
                except:
                    pass
                return False
            
            # 指数退避
            wait_time = 2 ** attempt
            logger.warning(f"W&B logging failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
            time.sleep(wait_time)
    
    return False