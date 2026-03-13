"""
通用SHAP分析器
统一的特征重要性分析入口
"""

import json
import logging
from typing import Dict, List, Any
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .feature_classifier import FeatureClassifier
from .analysis_strategies import AnalysisStrategyRegistry
from .scoring_normalizer import ScoringNormalizer
from .data_validator import DataValidator

logger = logging.getLogger(__name__)


class UniversalSHAPAnalyzer:
    """通用SHAP分析框架"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # 设置matplotlib非交互式后端，避免GUI依赖
        import matplotlib
        matplotlib.use('Agg')
        
        self.config = config or {}
        self.feature_classifier = FeatureClassifier()
        self.analysis_strategies = AnalysisStrategyRegistry()
        self.scoring_normalizer = ScoringNormalizer()
        self.data_validator = DataValidator()
        
        # 分析配置
        self.enabled_strategies = self.config.get('enabled_strategies', 
                                                 ['numeric', 'sequence', 'text', 'categorical'])
        
        # 输出目录配置 - 优先使用统一的实验输出路径
        explicit_output_dir = self.config.get('output_dir')
        if explicit_output_dir is not None:
            # 如果明确指定了output_dir，使用指定的路径
            self.output_dir = Path(explicit_output_dir)
        elif 'logging' in self.config and 'save_dir' in self.config['logging']:
            # 使用项目统一的日志输出目录
            base_output_dir = Path(self.config['logging']['save_dir'])
            log_name = self.config['logging'].get('name', 'experiment')
            log_version = self.config['logging'].get('version', 'default')
            self.output_dir = base_output_dir / log_name / log_version / 'shap_analysis'
        elif 'hydra' in self.config and 'runtime' in self.config.get('hydra', {}):
            # 如果使用Hydra，使用Hydra的输出目录
            hydra_dir = Path(self.config['hydra']['runtime']['cwd'])
            self.output_dir = hydra_dir / 'outputs' / 'shap_analysis'
        else:
            # 默认输出到当前工作目录下的shap_results
            self.output_dir = Path('./shap_results')
        
        self.save_plots = self.config.get('save_plots', True)
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, model: Any, batch: Dict[str, Any], model_type: str = None) -> Dict[str, Any]:
        """统一的分析入口"""
        try:
            logger.info(f"开始通用特征重要性分析，模型类型: {model_type}")
            
            # 0. 智能处理文本特征（combined_text分解）
            processed_batch = self._preprocess_batch_for_analysis(batch, model_type)
            
            # 1. 特征分类
            feature_types = self.feature_classifier.classify(processed_batch, model_type)
            logger.info(f"特征分类完成: {self.feature_classifier.get_feature_summary(feature_types)}")
            
            # 2. 数据质量验证（如果启用）
            enable_validation = self.config.get('enable_data_validation', True)
            if enable_validation:
                logger.info("开始数据质量验证...")
                validation_report = self.data_validator.validate_batch(processed_batch, feature_types)
                if not validation_report.get('overall_valid', True):
                    logger.warning("数据质量验证发现问题，但继续分析")
                logger.info(f"数据质量验证完成: {self.data_validator.get_validation_summary(validation_report)}")
            
            # 3. 执行分析
            results = {}
            analysis_summary = {
                'total_features_analyzed': 0,
                'successful_analyses': 0,
                'failed_analyses': 0,
                'feature_types_found': []
            }
            
            for feature_type, features in feature_types.items():
                if feature_type == 'excluded' or not features:
                    continue
                
                if feature_type not in self.enabled_strategies:
                    logger.info(f"跳过已禁用的分析策略: {feature_type}")
                    continue
                
                strategy = self.analysis_strategies.get_strategy(feature_type)
                if strategy:
                    try:
                        logger.info(f"使用 {feature_type} 策略分析 {len(features)} 个特征")
                        analysis_result = strategy.analyze(model, features, processed_batch)
                        results[feature_type] = analysis_result
                        
                        # 更新统计信息
                        success_count = sum(1 for feature_result in analysis_result.values() 
                                          if not feature_result.get('error', False))
                        fail_count = len(analysis_result) - success_count
                        
                        analysis_summary['successful_analyses'] += success_count
                        analysis_summary['failed_analyses'] += fail_count
                        analysis_summary['total_features_analyzed'] += len(features)
                        analysis_summary['feature_types_found'].append(feature_type)
                        
                        logger.info(f"{feature_type} 分析完成: 成功 {success_count} 个, 失败 {fail_count} 个")
                        
                    except Exception as e:
                        logger.error(f"{feature_type} 策略执行失败: {e}")
                        results[feature_type] = {'error': str(e)}
                else:
                    logger.warning(f"未找到 {feature_type} 类型的分析策略")
            
            # 4. 重新平衡评分机制（如果启用）
            enable_rebalancing = self.config.get('enable_score_rebalancing', True)
            if enable_rebalancing:
                logger.info("启用评分重新平衡...")
                results = self.scoring_normalizer.rebalance_composite_scores(results)
                logger.info("评分重新平衡完成")
            
            # 5. 聚合结果
            aggregated_results = self._aggregate_results(results, feature_types)
            
            # 6. 生成分析报告
            analysis_summary.update({
                'model_type': model_type,
                'analysis_timestamp': str(np.datetime64('now')),
                'batch_size': self._get_batch_size(batch),
                'analysis_config': self.config
            })
            
            final_results = {
                'detailed_analysis': results,
                'aggregated_importance': aggregated_results,
                'analysis_summary': analysis_summary,
                'feature_classification': feature_types
            }
            
            logger.info(f"通用特征重要性分析完成，共分析 {analysis_summary['total_features_analyzed']} 个特征")
            
            # 生成基于epoch的时间戳，避免重复文件
            import datetime
            import re
            
            # 尝试从模型或批次信息中获取epoch号
            epoch_num = 0
            if hasattr(self, '_current_epoch'):
                epoch_num = self._current_epoch
            elif hasattr(model, 'current_epoch'):
                epoch_num = getattr(model, 'current_epoch', 0)
            elif 'epoch_' in str(model_type):
                # 从模型类型中提取epoch号
                epoch_match = re.search(r'epoch_(\d+)', str(model_type))
                if epoch_match:
                    epoch_num = epoch_match.group(1)
            
            # 生成简化的时间戳（精确到分钟，避免秒级重复）
            timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M')
            
            # 保存分析结果
            filename_prefix = f'generic_shap_epoch_{epoch_num}'
            self._save_results(final_results, timestamp, filename_prefix)
            
            # 生成可视化图表（如果启用）
            if self.save_plots:
                self._generate_plots(final_results, timestamp, f'generic_shap_epoch_{model_type if model_type else "unknown"}')
            
            return final_results
            
        except Exception as e:
            logger.error(f"通用特征重要性分析失败: {e}")
            return {
                'error': str(e),
                'analysis_type': 'universal_shap_failed'
            }
    
    def _standardize_metric_values(self, metrics_dict):
        """标准化度量值为纯数值"""
        if not isinstance(metrics_dict, dict):
            return {}
        
        standardized = {}
        for key, value in metrics_dict.items():
            # 处理tensor字符串类型
            if isinstance(value, str) and value.startswith("tensor("):
                try:
                    import re
                    tensor_str = re.search(r'tensor\(([^)]+)\)', value)
                    if tensor_str:
                        numeric_value = float(tensor_str.group(1))
                        standardized[key] = numeric_value
                    else:
                        standardized[key] = 0.0
                except:
                    standardized[key] = 0.0
            elif isinstance(value, (int, float, np.number)):
                standardized[key] = float(value)
            elif isinstance(value, torch.Tensor):
                # 如果是tensor，转换为标量
                try:
                    standardized[key] = float(value.item())
                except:
                    standardized[key] = 0.0
            else:
                # 处理其他类型
                try:
                    standardized[key] = float(value)
                except:
                    standardized[key] = 0.0
        return standardized

    def _aggregate_results(self, results: Dict[str, Any], feature_types: Dict[str, List[str]]) -> Dict[str, Any]:
        """聚合不同类型的分析结果"""
        try:
            all_importance_scores = {}
            
            # 收集所有特征的评分
            for feature_type, feature_results in results.items():
                if 'error' in feature_results:
                    continue
                
                for feature_name, analysis in feature_results.items():
                    if 'error' not in analysis and 'metrics' in analysis:
                        # 标准化metrics中的所有值
                        standardized_metrics = self._standardize_metric_values(analysis['metrics'])
                        
                        # 优先使用标准化后的分数，否则使用原始分数
                        composite_score = standardized_metrics.get('standardized_score', 
                                         standardized_metrics.get('composite_score', 0.0))
                        
                        # 处理NaN值
                        if pd.isna(composite_score) or composite_score != composite_score:
                            composite_score = 0.0
                            logger.debug(f"Feature {feature_name} had NaN score, set to 0.0")
                        
                        # 使用完整的特征名包含类型信息
                        full_feature_name = f"{feature_type}:{feature_name}"
                        all_importance_scores[full_feature_name] = {
                            'score': composite_score,
                            'type': feature_type,
                            'metrics': standardized_metrics,
                            'analysis_type': analysis.get('analysis_type', 'unknown')
                        }
            
            # 按重要性排序
            sorted_features = sorted(all_importance_scores.items(), 
                                   key=lambda x: x[1]['score'], reverse=True)
            
            # 分类统计
            type_stats = {}
            for feature_type in feature_types.keys():
                if feature_type == 'excluded':
                    continue
                    
                type_features = [item for item in sorted_features 
                               if item[1]['type'] == feature_type]
                
                if type_features:
                    scores = [item[1]['score'] for item in type_features]
                    type_stats[feature_type] = {
                        'count': len(type_features),
                        'avg_importance': np.mean(scores),
                        'max_importance': np.max(scores),
                        'min_importance': np.min(scores),
                        'top_features': type_features[:3]  # 前3个重要特征
                    }
            
            return {
                'sorted_importance': dict(sorted_features),
                'type_statistics': type_stats,
                'total_analyzed_features': len(sorted_features),
                'most_important_feature': sorted_features[0] if sorted_features else None,
                'least_important_feature': sorted_features[-1] if sorted_features else None
            }
            
        except Exception as e:
            logger.error(f"聚合结果时出错: {e}")
            return {'error': str(e)}
    
    def _save_results(self, results: Dict[str, Any], timestamp: str, filename_prefix: str):
        """保存分析结果到文件"""
        try:
            # 检查是否已经存在相同epoch的结果文件，避免重复保存
            epoch_pattern = None
            if 'epoch_' in filename_prefix:
                # 提取epoch号
                import re
                epoch_match = re.search(r'epoch_(\d+)', filename_prefix)
                if epoch_match:
                    epoch_num = epoch_match.group(1)
                    epoch_pattern = f"*epoch_{epoch_num}_results_*.json"
                    
                    # 检查是否已存在该epoch的结果文件
                    existing_files = list(self.output_dir.glob(epoch_pattern))
                    if existing_files:
                        logger.info(f"Epoch {epoch_num} 的SHAP结果已存在，跳过重复保存")
                        return existing_files[0]  # 返回现有文件路径
            
            # 保存详细结果到JSON文件
            results_file = self.output_dir / f"{filename_prefix}_results_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"分析结果已保存到: {results_file}")
            
            # 生成摘要报告（仅在第一次保存时）
            summary_pattern = f"{filename_prefix}_summary_{timestamp}.txt"
            summary_file = self.output_dir / summary_pattern
            if not summary_file.exists():
                self._generate_summary_report(results, timestamp, filename_prefix)
            
        except Exception as e:
            logger.error(f"保存结果时出错: {e}")
    
    def _generate_summary_report(self, results: Dict[str, Any], timestamp: str, filename_prefix: str):
        """生成摘要报告"""
        try:
            summary_file = self.output_dir / f"{filename_prefix}_summary_{timestamp}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=== 通用特征重要性分析报告 ===\n\n")
                
                # 分析摘要
                if 'analysis_summary' in results:
                    summary = results['analysis_summary']
                    f.write(f"模型类型: {summary.get('model_type', 'unknown')}\n")
                    f.write(f"分析时间: {summary.get('analysis_timestamp', 'unknown')}\n")
                    f.write(f"批次大小: {summary.get('batch_size', 0)}\n")
                    f.write(f"总分析特征数: {summary.get('total_features_analyzed', 0)}\n")
                    f.write(f"成功分析: {summary.get('successful_analyses', 0)}\n")
                    f.write(f"失败分析: {summary.get('failed_analyses', 0)}\n")
                    f.write(f"发现的特征类型: {', '.join(summary.get('feature_types_found', []))}\n\n")
                
                # 聚合结果摘要
                if 'aggregated_importance' in results:
                    agg = results['aggregated_importance']
                    f.write("=== 特征重要性摘要 ===\n")
                    
                    if 'sorted_importance' in agg:
                        f.write(f"\n前10个最重要特征:\n")
                        for i, (feature_name, importance_info) in enumerate(list(agg['sorted_importance'].items())[:10]):
                            score = importance_info.get('score', 0.0)
                            feature_type = importance_info.get('type', 'unknown')
                            f.write(f"{i+1:2d}. {feature_name:<40} [分数: {score:.6f}, 类型: {feature_type}]\n")
                    
                    if 'type_statistics' in agg:
                        f.write(f"\n按特征类型统计:\n")
                        for feature_type, stats in agg['type_statistics'].items():
                            avg_importance = stats.get('avg_importance', 0.0)
                            count = stats.get('count', 0)
                            f.write(f"- {feature_type:<15}: {count:>3d} 个特征, 平均重要性: {avg_importance:.6f}\n")
                
                # 详细分析结果
                if 'detailed_analysis' in results:
                    f.write("\n=== 详细分析结果 ===\n")
                    for feature_type, feature_results in results['detailed_analysis'].items():
                        if 'error' in feature_results:
                            continue
                            
                        f.write(f"\n{feature_type.upper()} 类型特征分析:\n")
                        for feature_name, analysis in feature_results.items():
                            if 'error' in analysis:
                                f.write(f"  - {feature_name}: 分析失败 - {analysis.get('error', 'unknown error')}\n")
                            elif 'metrics' in analysis:
                                metrics = analysis['metrics']
                                composite_score = metrics.get('composite_score', 0.0)
                                f.write(f"  - {feature_name}: 综合评分 {composite_score:.6f}\n")
            
            logger.info(f"摘要报告已保存到: {summary_file}")
            
        except Exception as e:
            logger.error(f"生成摘要报告时出错: {e}")
    
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """获取批次大小"""
        try:
            # 尝试从常见的特征中获取批次大小
            for key in ['numeric_features', 'sequence_data', 'text_features']:
                if key in batch and hasattr(batch[key], 'shape'):
                    return batch[key].shape[0]
            
            # 如果没有找到，返回字典键的数量
            return len(batch)
            
        except:
            return 0
    
    def _generate_plots(self, results: Dict[str, Any], timestamp: str, filename_prefix: str):
        """生成可视化图表"""
        try:
            # 1. 特征重要性条形图
            if 'aggregated_importance' in results:
                agg = results['aggregated_importance']
                if 'sorted_importance' in agg:
                    self._plot_importance_bar_chart(agg['sorted_importance'], timestamp, filename_prefix)
                
                # 2. 特征类型分布饼图
                if 'type_statistics' in agg:
                    self._plot_type_distribution_pie(agg['type_statistics'], timestamp, filename_prefix)
            
            logger.info(f"可视化图表已生成，时间戳: {timestamp}")
            
        except Exception as e:
            logger.error(f"生成图表时出错: {e}")
    
    def _plot_importance_bar_chart(self, sorted_importance: Dict[str, Any], timestamp: str, filename_prefix: str):
        """生成特征重要性条形图"""
        try:
            # 临时设置matplotlib为静默模式，避免字体管理调试信息
            import matplotlib as mpl
            import matplotlib.font_manager as fm
            import logging
            
            # 保存原始的日志级别
            mpl_logger = logging.getLogger('matplotlib')
            font_logger = logging.getLogger('matplotlib.font_manager')
            
            original_level = mpl_logger.level
            original_font_level = font_logger.level
            
            mpl_logger.setLevel(logging.WARNING)
            font_logger.setLevel(logging.WARNING)
            
            # 设置Times New Roman字体和字体大小
            import warnings
            try:
                # 尝试设置Times New Roman字体
                plt.rcParams['font.family'] = 'Times New Roman'
                plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 备用字体
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.size'] = 12
                plt.rcParams['axes.titlesize'] = 16
                plt.rcParams['axes.labelsize'] = 14
                plt.rcParams['xtick.labelsize'] = 12
                plt.rcParams['ytick.labelsize'] = 12
                plt.rcParams['legend.fontsize'] = 12
                logger.info("字体已设置为Times New Roman")
            except Exception as e:
                logger.debug(f"Times New Roman字体设置失败，使用默认字体: {e}")
                # 备用字体设置
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            
            # 取前20个重要特征
            top_features = list(sorted_importance.items())[:20]
            
            if not top_features:
                # 恢复原始设置
                mpl_logger.setLevel(original_level)
                font_logger.setLevel(original_font_level)
                return
            
            # 处理特征名显示
            features = []
            scores = []
            types = []
            
            for item in top_features:
                feature_name = item[0]
                # 特征名截断和处理，确保能正常显示
                if len(feature_name) > 40:
                    # 如果特征名太长，进行智能截断
                    if ':' in feature_name:
                        type_part, name_part = feature_name.split(':', 1)
                        if len(name_part) > 30:
                            name_part = name_part[:27] + '...'
                        features.append(f"{type_part}:{name_part}")
                    else:
                        features.append(feature_name[:37] + '...')
                else:
                    features.append(feature_name)
                
                scores.append(item[1]['score'])
                types.append(item[1]['type'])
            
            # 创建更大的图形尺寸，优化布局
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # 创建颜色映射和纹理映射 - 使用灰度色彩+纹理区分，适合黑白印刷
            color_map = {'numeric': '#404040', 'sequence': '#606060', 'text': '#808080', 'categorical': '#A0A0A0'}
            hatch_map = {
                'numeric': '///',      # 斜线密集
                'sequence': '\\\\\\\\',   # 反斜线密集
                'text': '|||',        # 竖线密集
                'categorical': '---'   # 横线密集
            }
            colors = [color_map.get(t, '#505050') for t in types]
            hatches = [hatch_map.get(t, '...') for t in types]
            
            # 绘制水平条形图，添加纹理模式（增强纹理可见性）
            bars = ax.barh(range(len(features)), scores, color=colors, alpha=0.9, 
                          edgecolor='black', linewidth=1.0, hatch=hatches)
            
            # 设置y轴标签
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=10)
            
            # 设置x轴标签和单位
            ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
            
            # 设置标题，确保完整显示
            ax.set_title('Top 20 Feature Importance Ranking', fontsize=16, fontweight='bold', pad=20)
            
            # 反转y轴，使最重要的特征在顶部
            ax.invert_yaxis()
            
            # 在条形图上添加数值标签 - 优化显示位置和格式
            max_score = max(scores) if scores else 1.0
            for i, (bar, score) in enumerate(zip(bars, scores)):
                # 处理NaN值
                if pd.isna(score) or score != score:
                    label_text = '0.0'
                    score = 0.0
                else:
                    # 格式化显示数字
                    if score >= 1.0:
                        label_text = f'{score:.1f}'
                    else:
                        label_text = f'{score:.3f}'
                
                # 计算标签位置，确保不超出图形边界
                label_x = bar.get_width() + max_score * 0.02
                label_y = bar.get_y() + bar.get_height()/2
                
                ax.text(label_x, label_y, label_text, ha='left', va='center', 
                       fontsize=10, fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 添加图例，包含纹理模式 - 增强纹理可见性
            legend_elements = []
            for t in set(types):
                if t in color_map:
                    # 创建带纹理的图例元素
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((0,0),1,1, facecolor=color_map[t], hatch=hatch_map.get(t, '...'), 
                                   edgecolor='black', linewidth=1.0, alpha=0.9, label=f'{t} feature')
                    legend_elements.append(rect)
            ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.9, 
                     title='Feature Types', title_fontsize=14)
            
            # 添加网格线提升可读性
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # 调整布局，确保所有元素都能完整显示
            plt.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.08, hspace=0.3)
            
            # 保存图表，使用更高的DPI和质量
            plot_file = self.output_dir / f"{filename_prefix}_importance_chart_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"特征重要性条形图已保存: {plot_file}")
            
            # 恢复原始日志设置
            mpl_logger.setLevel(original_level)
            font_logger.setLevel(original_font_level)
            
        except Exception as e:
            logger.error(f"生成重要性条形图时出错: {e}")
            # 确保恢复设置
            try:
                mpl_logger.setLevel(original_level)
                font_logger.setLevel(original_font_level)
                plt.close()
            except:
                pass
    
    def _plot_type_distribution_pie(self, type_statistics: Dict[str, Any], timestamp: str, filename_prefix: str):
        """生成特征类型分布饼图"""
        try:
            # 设置matplotlib静默模式
            import matplotlib as mpl
            import matplotlib.font_manager as fm
            import logging
            
            # 保存原始的日志级别
            mpl_logger = logging.getLogger('matplotlib')
            font_logger = logging.getLogger('matplotlib.font_manager')
            
            original_level = mpl_logger.level
            original_font_level = font_logger.level
            
            mpl_logger.setLevel(logging.WARNING)
            font_logger.setLevel(logging.WARNING)
            
            # 设置Times New Roman字体和字体大小
            try:
                plt.rcParams['font.family'] = 'Times New Roman'
                plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 备用字体
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.size'] = 14
                plt.rcParams['axes.titlesize'] = 18
                plt.rcParams['axes.labelsize'] = 14
                plt.rcParams['legend.fontsize'] = 12
                logger.info("饼图字体已设置为Times New Roman")
            except Exception as e:
                logger.debug(f"饼图Times New Roman字体设置失败，使用默认字体: {e}")
                # 备用字体设置
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            
            if not type_statistics:
                mpl_logger.setLevel(original_level)
                font_logger.setLevel(original_font_level)
                return
            
            types = list(type_statistics.keys())
            counts = [stats['count'] for stats in type_statistics.values()]
            
            # 创建更大的图形，优化布局 - 增加尺寸确保标题显示
            fig, ax = plt.subplots(figsize=(12, 9))
            
            # 设置灰度配色方案和密集纹理模式（适合黑白印刷）
            colors = ['#404040', '#606060', '#808080', '#A0A0A0', '#C0C0C0', '#303030', '#505050', '#707070']
            hatches = ['///', '\\\\\\\\', '|||', '---', '+++', '...', 'xxx', 'OOO']  # 密集纹理模式
            colors = colors[:len(types)]
            hatches = hatches[:len(types)]
            
            # 生成饼图，科学论文风格（无阴影、无爆炸效果）
            wedges, texts, autotexts = ax.pie(counts, labels=types, colors=colors, autopct='%1.1f%%', 
                                              startangle=90, shadow=False,
                                              textprops={'fontsize': 14, 'fontweight': 'bold'})
            
            # 为每个饼图块添加密集纹理模式，增强可见性
            for i, wedge in enumerate(wedges):
                wedge.set_hatch(hatches[i])
                wedge.set_edgecolor('black')
                wedge.set_linewidth(1.0)  # 增强边框可见性
            
            # 设置标题，确保完整显示
            ax.set_title('Feature Type Distribution', fontsize=18, fontweight='bold', pad=30)
            
            # 确保饼图是圆形
            ax.axis('equal')
            
            # 调整布局，无图例时使用更大空间
            plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
            
            # 保存图表，使用更高的DPI和质量
            plot_file = self.output_dir / f"{filename_prefix}_type_distribution_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"特征类型分布饼图已保存: {plot_file}")
            
            # 恢复日志设置
            mpl_logger.setLevel(original_level)
            font_logger.setLevel(original_font_level)
            
        except Exception as e:
            logger.error(f"生成类型分布饼图时出错: {e}")
            try:
                mpl_logger.setLevel(original_level)
                font_logger.setLevel(original_font_level)
                plt.close()
            except:
                pass
    
    def get_supported_feature_types(self) -> List[str]:
        """获取支持的特征类型"""
        return self.analysis_strategies.list_available_strategies()
    
    def enable_strategy(self, feature_type: str):
        """启用分析策略"""
        if feature_type not in self.enabled_strategies:
            self.enabled_strategies.append(feature_type)
            logger.info(f"已启用分析策略: {feature_type}")
    
    def disable_strategy(self, feature_type: str):
        """禁用分析策略"""
        if feature_type in self.enabled_strategies:
            self.enabled_strategies.remove(feature_type)
            logger.info(f"已禁用分析策略: {feature_type}")

    def _preprocess_batch_for_analysis(self, batch: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """为分析预处理批次数据，包括combined_text分解和特征过滤"""
        processed_batch = batch.copy()
        
        # 获取模型映射配置
        model_mapping = self.feature_classifier.model_specific_mappings.get(model_type, {})
        
        # 处理combined_text分解
        if 'combined_text' in processed_batch and model_mapping.get('decompose_combined_text', False):
            combined_text = processed_batch['combined_text']
            decomposed_features = self.feature_classifier._decompose_combined_text(combined_text)
            
            # 将分解的特征添加到批次中
            for feature_name, feature_value in decomposed_features.items():
                processed_batch[feature_name] = feature_value
            
            # 标记combined_text将被排除
            logger.info(f"combined_text已分解为{len(decomposed_features)}个独立特征: {list(decomposed_features.keys())}")
        
        # 移除需要排除的特征
        exclude_patterns = model_mapping.get('exclude_patterns', [])
        exclude_keywords = ['mask', 'labels', 'target', 'split', 'uid', 'index', 'enabled', 'idx']
        features_to_remove = []
        
        for key in processed_batch.keys():
            # 检查是否匹配排除模式
            should_exclude = False
            
            # 模式匹配
            for pattern in exclude_patterns:
                import re
                if re.search(pattern, key):
                    should_exclude = True
                    break
            
            # 关键词匹配
            if not should_exclude:
                for keyword in exclude_keywords:
                    if keyword in key:
                        should_exclude = True
                        break
            
            # 检查是否为combined_text（需要分解）
            if key == 'combined_text' and model_mapping.get('decompose_combined_text', False):
                should_exclude = True
            
            if should_exclude:
                features_to_remove.append(key)
        
        # 移除排除的特征
        for feature in features_to_remove:
            del processed_batch[feature]
            logger.debug(f"已排除特征: {feature}")
        
        if features_to_remove:
            logger.info(f"已排除{len(features_to_remove)}个特征: {features_to_remove[:5]}...")  # 只显示前5个
        
        return processed_batch

    def save_results(self, results: Dict[str, Any], filename_prefix: str):
        """保存分析结果的公共接口
        
        Args:
            results: 分析结果字典
            filename_prefix: 文件名前缀
            
        Returns:
            str: 保存的文件路径
        """
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            self._save_results(results, timestamp, filename_prefix)
            
            result_file = self.output_dir / f"{filename_prefix}_results_{timestamp}.json"
            return str(result_file)
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            return None