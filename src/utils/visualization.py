# -*- coding: utf-8 -*-
"""
可视化工具模块（增强）
提供flow_autoencoder专用的可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

# 使用统一的日志配置
from .logging_config import setup_preset_logging
logger = setup_preset_logging(log_level=logging.DEBUG)

class FlowAutoencoderVisualizer:
    """flow_autoencoder专用可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("flow_autoencoder可视化器初始化完成")
    
    def plot_training_curves(self, training_history: Dict[str, List], 
                           save_path: Optional[str] = None) -> plt.Figure:
        """绘制训练曲线"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Flow Autoencoder 训练曲线', fontsize=16)
            
            # 损失曲线
            if 'train_loss' in training_history and 'val_loss' in training_history:
                axes[0, 0].plot(training_history['train_loss'], label='训练损失')
                axes[0, 0].plot(training_history['val_loss'], label='验证损失')
                axes[0, 0].set_title('损失曲线')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # 重构损失曲线
            if 'train_reconstruction_loss' in training_history and 'val_reconstruction_loss' in training_history:
                axes[0, 1].plot(training_history['train_reconstruction_loss'], label='训练重构损失')
                axes[0, 1].plot(training_history['val_reconstruction_loss'], label='验证重构损失')
                axes[0, 1].set_title('重构损失曲线')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Reconstruction Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # 学习率曲线（如果有）
            if 'learning_rate' in training_history:
                axes[1, 0].plot(training_history['learning_rate'])
                axes[1, 0].set_title('学习率变化')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].grid(True)
            
            # 其他指标
            other_metrics = [k for k in training_history.keys() 
                           if k not in ['train_loss', 'val_loss', 'train_reconstruction_loss', 
                                      'val_reconstruction_loss', 'learning_rate']]
            if other_metrics:
                for i, metric in enumerate(other_metrics[:4]):  # 最多显示4个其他指标
                    row, col = 1 + i//2, i%2
                    if row < 2 and col < 2:  # 确保在子图范围内
                        axes[row, col].plot(training_history[metric])
                        axes[row, col].set_title(f'{metric}')
                        axes[row, col].set_xlabel('Epoch')
                        axes[row, col].set_ylabel('Value')
                        axes[row, col].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"训练曲线已保存到: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"训练曲线绘制失败: {e}")
            raise
    
    def plot_reconstruction_comparison(self, original: np.ndarray, 
                                    reconstructed: np.ndarray,
                                    sample_indices: Optional[List[int]] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """绘制重构对比图"""
        try:
            if sample_indices is None:
                sample_indices = list(range(min(5, len(original))))
            
            n_samples = len(sample_indices)
            fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
            fig.suptitle('原始数据 vs 重构数据对比', fontsize=16)
            
            if n_samples == 1:
                axes = np.array([axes])
            
            for i, idx in enumerate(sample_indices):
                # 原始数据
                axes[i, 0].plot(original[idx], 'b-', alpha=0.7, label='原始')
                axes[i, 0].set_title(f'样本 {idx} - 原始数据')
                axes[i, 0].grid(True)
                
                # 重构数据
                axes[i, 1].plot(reconstructed[idx], 'r-', alpha=0.7, label='重构')
                axes[i, 1].set_title(f'样本 {idx} - 重构数据')
                axes[i, 1].grid(True)
                
                # 如果只有两个样本，添加图例
                if n_samples == 1:
                    axes[i, 0].legend()
                    axes[i, 1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"重构对比图已保存到: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"重构对比图绘制失败: {e}")
            raise
    
    def plot_latent_space(self, latent_vectors: np.ndarray, 
                         labels: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """绘制潜在空间"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('潜在空间可视化', fontsize=16)
            
            # PCA降维
            if latent_vectors.shape[1] > 2:
                pca = PCA(n_components=2)
                latent_pca = pca.fit_transform(latent_vectors)
            else:
                latent_pca = latent_vectors
            
            # t-SNE降维
            if latent_vectors.shape[1] > 2:
                tsne = TSNE(n_components=2, random_state=42)
                latent_tsne = tsne.fit_transform(latent_vectors)
            else:
                latent_tsne = latent_vectors
            
            # PCA图
            if labels is not None:
                scatter = axes[0].scatter(latent_pca[:, 0], latent_pca[:, 1], 
                                        c=labels, alpha=0.6, cmap='viridis')
                plt.colorbar(scatter, ax=axes[0])
            else:
                axes[0].scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.6)
            
            axes[0].set_title('PCA 降维')
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')
            axes[0].grid(True)
            
            # t-SNE图
            if labels is not None:
                scatter = axes[1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                                        c=labels, alpha=0.6, cmap='viridis')
                plt.colorbar(scatter, ax=axes[1])
            else:
                axes[1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.6)
            
            axes[1].set_title('t-SNE 降维')
            axes[1].set_xlabel('t-SNE1')
            axes[1].set_ylabel('t-SNE2')
            axes[1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"潜在空间图已保存到: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"潜在空间图绘制失败: {e}")
            raise
    
    def plot_ablation_results(self, ablation_results: Dict[str, Dict],
                            metric: str = 'val_loss',
                            save_path: Optional[str] = None) -> plt.Figure:
        """绘制消融实验结果"""
        try:
            # 准备数据
            variants = []
            metrics = []
            
            for variant_name, results in ablation_results.items():
                if 'performance_metrics' in results:
                    variant_metrics = results['performance_metrics']
                    if metric in variant_metrics:
                        variants.append(variant_name)
                        metrics.append(variant_metrics[metric])
            
            # 创建条形图
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bars = ax.bar(variants, metrics, color=sns.color_palette("husl", len(variants)))
            ax.set_title(f'消融实验 - {metric} 对比')
            ax.set_xlabel('变体')
            ax.set_ylabel(metric)
            ax.grid(True, axis='y', alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, metrics):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.4f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"消融结果图已保存到: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"消融结果图绘制失败: {e}")
            raise
    
    def plot_hyperparameter_importance(self, importance_scores: Dict[str, float],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """绘制超参数重要性"""
        try:
            if not importance_scores:
                logger.warning("超参数重要性数据为空")
                return None
            
            # 排序
            sorted_scores = dict(sorted(importance_scores.items(), 
                                      key=lambda x: x[1], reverse=True))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.barh(list(sorted_scores.keys()), list(sorted_scores.values()),
                          color=sns.color_palette("Blues_r", len(sorted_scores)))
            
            ax.set_title('超参数重要性排名')
            ax.set_xlabel('重要性分数')
            ax.grid(True, axis='x', alpha=0.3)
            
            # 添加数值标签
            for bar, (param, score) in zip(bars, sorted_scores.items()):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.4f}', ha='left', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"超参数重要性图已保存到: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"超参数重要性图绘制失败: {e}")
            raise
    
    def create_experiment_report(self, experiment_results: Dict[str, Any],
                               save_dir: Optional[str] = None) -> Dict[str, str]:
        """创建实验报告（包含多个图表）"""
        try:
            report_files = {}
            
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
            
            # 训练曲线
            if 'training_history' in experiment_results:
                training_fig = self.plot_training_curves(experiment_results['training_history'])
                if save_dir:
                    training_path = os.path.join(save_dir, 'training_curves.png')
                    training_fig.savefig(training_path, dpi=300, bbox_inches='tight')
                    report_files['training_curves'] = training_path
                    plt.close(training_fig)
            
            # 消融结果
            if 'ablation_results' in experiment_results:
                ablation_fig = self.plot_ablation_results(experiment_results['ablation_results'])
                if save_dir:
                    ablation_path = os.path.join(save_dir, 'ablation_results.png')
                    ablation_fig.savefig(ablation_path, dpi=300, bbox_inches='tight')
                    report_files['ablation_results'] = ablation_path
                    plt.close(ablation_fig)
            
            # 超参数重要性
            if 'hyperparameter_importance' in experiment_results:
                importance_fig = self.plot_hyperparameter_importance(
                    experiment_results['hyperparameter_importance'])
                if save_dir and importance_fig:
                    importance_path = os.path.join(save_dir, 'hyperparameter_importance.png')
                    importance_fig.savefig(importance_path, dpi=300, bbox_inches='tight')
                    report_files['hyperparameter_importance'] = importance_path
                    plt.close(importance_fig)
            
            logger.info(f"实验报告已生成，包含 {len(report_files)} 个图表")
            return report_files
            
        except Exception as e:
            logger.error(f"实验报告生成失败: {e}")
            return {}


class MetricsVisualizer:
    """通用指标可视化器"""
    
    def __init__(self, save_dir: str):
        """初始化可视化器"""
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"指标可视化器初始化完成，保存目录: {save_dir}")
    
    def create_comprehensive_report(self, metrics_history: Dict, final_metrics: Dict, 
                                  experiment_config: Dict, **kwargs):
        """创建综合报告"""
        try:
            import matplotlib.pyplot as plt
            
            # 绘制训练曲线
            if metrics_history:
                self.plot_training_curves(metrics_history, "comprehensive_training")
            
            # 绘制指标对比
            if final_metrics:
                self.plot_metrics_comparison(final_metrics, "comprehensive_metrics")
            
            # 生成实验报告
            self.generate_experiment_report(experiment_config, final_metrics, "comprehensive_report")
            
            logger.info("综合可视化报告生成完成")
        except Exception as e:
            logger.error(f"创建综合报告时出错: {e}")
    
    def plot_training_curves(self, metrics_history: Dict, filename_prefix: str):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
            
            # 提取关键指标
            train_loss = metrics_history.get('train_loss', [])
            val_loss = metrics_history.get('val_loss', [])
            train_acc = metrics_history.get('train_acc', metrics_history.get('train_accuracy', []))
            val_acc = metrics_history.get('val_acc', metrics_history.get('val_accuracy', []))
            
            if train_loss and val_loss:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # 损失曲线
                epochs = range(len(train_loss))
                ax1.plot(epochs, train_loss, label='训练损失')
                ax1.plot(epochs, val_loss, label='验证损失')
                ax1.set_title('训练和验证损失')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True)
                
                # 准确率曲线
                if train_acc and val_acc:
                    ax2.plot(epochs, train_acc, label='训练准确率')
                    ax2.plot(epochs, val_acc, label='验证准确率')
                    ax2.set_title('训练和验证准确率')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy')
                    ax2.legend()
                    ax2.grid(True)
                
                plt.tight_layout()
                
                # 保存图片
                import os
                file_path = os.path.join(self.save_dir, f"{filename_prefix}_curves.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"训练曲线已保存到: {file_path}")
        except Exception as e:
            logger.error(f"绘制训练曲线时出错: {e}")
    
    def plot_metrics_comparison(self, final_metrics: Dict, filename_prefix: str):
        """绘制指标对比"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not final_metrics:
                return
            
            # 筛选验证指标
            val_metrics = {k: v for k, v in final_metrics.items() if k.startswith('val_')}
            if not val_metrics:
                return
            
            # 创建条形图
            fig, ax = plt.subplots(figsize=(12, 6))
            
            metric_names = list(val_metrics.keys())
            metric_values = [val_metrics[name] for name in metric_names]
            
            bars = ax.bar(metric_names, metric_values, color='skyblue')
            ax.set_title('验证指标对比')
            ax.set_ylabel('Value')
            ax.grid(True, axis='y', alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, metric_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.4f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图片
            import os
            file_path = os.path.join(self.save_dir, f"{filename_prefix}_comparison.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"指标对比图已保存到: {file_path}")
        except Exception as e:
            logger.error(f"绘制指标对比时出错: {e}")
    
    def generate_experiment_report(self, experiment_config: Dict, final_metrics: Dict, filename_prefix: str):
        """生成实验报告"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建一个简单的报告图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 提取关键指标显示
            key_metrics = {}
            for metric_name in ['val_accuracy', 'val_f1_macro', 'val_loss']:
                if metric_name in final_metrics:
                    key_metrics[metric_name] = final_metrics[metric_name]
            
            if key_metrics:
                metric_names = list(key_metrics.keys())
                metric_values = [key_metrics[name] for name in metric_names]
                
                bars = ax.bar(metric_names, metric_values, color=['lightcoral', 'lightgreen', 'lightblue'])
                ax.set_title('实验关键指标')
                ax.set_ylabel('Value')
                ax.grid(True, axis='y', alpha=0.3)
                
                # 添加数值标签
                for bar, value in zip(bars, metric_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存图片
            import os
            file_path = os.path.join(self.save_dir, f"{filename_prefix}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"实验报告图表已保存到: {file_path}")
        except Exception as e:
            logger.error(f"生成实验报告时出错: {e}")