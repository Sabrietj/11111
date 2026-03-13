"""
五大特征类别分析器 - 完整的五大特征类别SHAP分析实现
"""

import torch
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from .dimension_extractor import ConfigDimensionExtractor
from .five_tier_classifier import FiveTierFeatureClassifier
from .multi_model_adapter import MultiModelSHAPAdapter
from .hierarchical_calculator import HierarchicalSHAPCalculator

logger = logging.getLogger(__name__)

class FiveTierSHAPAnalyzer:
    """五大特征类别SHAP分析器 - 完整实现五大特征类别分析架构"""
    
    def __init__(self, model, model_type: Optional[str] = None, config: Optional[Dict] = None):
        """
        初始化五大特征类别分析器
        
        Args:
            model: 要分析的模型
            model_type: 模型类型，如果为None则自动检测
            config: 分析配置
        """
        self.model = model
        self.cfg = model.cfg
        self.config = config or self._get_default_config()
        
        # 自动检测模型类型
        if model_type is None:
            adapter = MultiModelSHAPAdapter()
            self.model_type = adapter.auto_detect_model_type(model)
        else:
            self.model_type = model_type
        
        logger.info(f"五大特征类别分析器初始化，模型类型: {self.model_type}")
        
        # 初始化核心组件
        self.dimension_extractor = ConfigDimensionExtractor(self.cfg)
        self.feature_classifier = FiveTierFeatureClassifier()
        self.hierarchical_calculator = None  # 延迟初始化
        
        # 分析结果存储
        self.analysis_results = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'shap': {
                'enable_level1_analysis': True,   # 启用大类别分析（饼图）
                'enable_level2_analysis': True,   # 启用具体特征分析（柱状图）
                'focus_numeric_features': True,   # 重点分析数值特征
                'dynamic_dimension_calculation': True,  # 动态计算特征维度
                'enable_data_validation': True,
                'save_plots': True,
                'background_samples': 50,
                'eval_samples': 20
            },
            'logging': {
                'save_dir': './outputs',
                'name': 'five_tier_analysis',
                'version': 'default'
            }
        }
    
    def analyze(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行完整的五大特征类别分析
        
        Args:
            batch_data: 批次数据
            
        Returns:
            分析结果字典
        """
        logger.info("开始执行五大特征类别SHAP分析...")
        
        try:
            # 1. 特征分类
            feature_classification = self.feature_classifier.classify_from_config(self.cfg, batch_data)
            
            # 2. 维度验证
            if self.config['shap']['enable_data_validation']:
                validation_result = self.dimension_extractor.validate_dimensions(batch_data)
                if not validation_result['is_compatible']:
                    logger.warning(f"维度验证发现问题: {validation_result['mismatches']}")
                    # 继续执行，但记录警告
            
            # 3. 初始化分层计算器
            self.hierarchical_calculator = HierarchicalSHAPCalculator(self.model, feature_classification)
            
            # 4. 准备SHAP分析数据
            background_inputs, eval_inputs = self._prepare_shap_inputs(batch_data)
            
            # 5. 验证计算准备情况
            readiness_check = self.hierarchical_calculator.validate_computation_readiness(
                background_inputs, eval_inputs
            )
            if not readiness_check['is_ready']:
                raise RuntimeError(f"SHAP计算准备失败: {readiness_check['issues']}")
            
            # 6. 执行分层分析
            comprehensive_results = self.hierarchical_calculator.calculate_comprehensive_analysis(
                background_inputs, eval_inputs
            )
            
            # 7. 构建最终结果
            final_results = self._build_final_results(
                feature_classification, comprehensive_results, batch_data
            )
            
            self.analysis_results = final_results
            logger.info("五大特征类别SHAP分析完成")
            
            return final_results
            
        except Exception as e:
            logger.error(f"五大特征类别分析失败: {e}", exc_info=True)
            raise
    
    def _prepare_shap_inputs(self, batch_data: Dict[str, Any]) -> tuple:
        """准备SHAP分析所需的输入数据"""
        feature_dims = self.dimension_extractor.calculate_all_dimensions()
        
        # 1. 划分背景集和评估集
        total_samples = batch_data['numeric_features'].shape[0]
        bg_size = min(
            self.config['shap']['background_samples'], 
            int(total_samples * 0.7)
        )
        eval_size = min(
            self.config['shap']['eval_samples'], 
            total_samples - bg_size
        )
        
        device = self.model.device
        
        # 2. 构建背景输入
        bg_inputs = self._build_feature_inputs(
            batch_data, 0, bg_size, device, feature_dims
        )
        
        # 3. 构建评估输入
        eval_inputs = self._build_feature_inputs(
            batch_data, bg_size, bg_size + eval_size, device, feature_dims
        )
        
        return bg_inputs, eval_inputs
    
    def _build_feature_inputs(self, batch_data: Dict[str, Any], start: int, end: int, 
                           device: torch.device, feature_dims: Dict[str, int]) -> List[torch.Tensor]:
        """构建SHAP分析的特征输入"""
        # 构建切片
        slice_batch = {}
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                slice_batch[k] = v[start:end].to(device)
            elif isinstance(v, list):
                slice_batch[k] = v[start:end]
        
        # 预计算嵌入 - 移除no_grad以支持SHAP梯度计算
        # 数值特征
        numeric_feats = slice_batch['numeric_features']
        
        # 域名嵌入特征
        if self.model.domain_embedding_enabled and 'domain_embedding_features' in slice_batch:
            domain_feats = slice_batch['domain_embedding_features']
        else:
            domain_feats = torch.zeros(
                numeric_feats.shape[0], 
                feature_dims['domain_dims'], 
                device=device
            )
        
        # 类别特征嵌入
        categorical_columns_effective = getattr(self.model, 'categorical_columns_effective', [])
        if len(categorical_columns_effective) > 0:
            # 这里需要实际计算类别特征嵌入
            # 简化处理，使用零张量
            cat_feats = torch.zeros(
                numeric_feats.shape[0], 
                feature_dims['categorical_dims'], 
                device=device
            )
        else:
            cat_feats = torch.zeros(
                numeric_feats.shape[0], 
                feature_dims['categorical_dims'] if feature_dims['categorical_dims'] > 0 else 1, 
                device=device
            )
        
        # 序列特征
        if self.model.sequence_features_enabled:
            seq_data = {
                'iat_times': slice_batch['iat_times'],
                'payload_sizes': slice_batch['payload_sizes'],
                'sequence_mask': slice_batch['sequence_mask']
            }
            seq_emb = self.model.sequence_encoder(seq_data)["sequence_embedding"]
        else:
            seq_emb = torch.zeros(
                numeric_feats.shape[0], 
                feature_dims['sequence_dims'], 
                device=device
            )
        
        # 文本特征
        if self.model.text_features_enabled:
            text_emb = self.model._process_text_features(slice_batch)
        else:
            text_emb = torch.zeros(
                numeric_feats.shape[0], 
                feature_dims['text_dims'], 
                device=device
            )
        
        # 强制启用梯度 - 传递5个输入给EnhancedShapFusionWrapper
        inputs = [numeric_feats, domain_feats, cat_feats, seq_emb, text_emb]
        final_inputs = []
        
        for t in inputs:
            t = t.detach().clone()
            if not t.is_floating_point():
                t = t.float()
            t.requires_grad_(True)
            final_inputs.append(t)
        
        return final_inputs
    
    def _build_final_results(self, feature_classification: Dict[str, List[str]], 
                          comprehensive_results: Dict[str, Any], 
                          batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """构建最终分析结果"""
        
        # 验证维度信息
        dimension_validation = self.dimension_extractor.validate_dimensions(batch_data)
        
        final_results = {
            'five_tier_analysis': {
                'level1_category_importance': comprehensive_results['level1_category_importance'],
                'level2_numeric_importance': comprehensive_results['level2_numeric_importance'],
                'dimension_validation': {
                    'expected_dimensions': dimension_validation['expected'],
                    'actual_dimensions': dimension_validation['actual'],
                    'validation_status': 'PASSED' if dimension_validation['is_compatible'] else 'FAILED'
                },
                'feature_classification': feature_classification
            },
            'analysis_summary': {
                'total_categories_analyzed': 5,
                'total_numeric_features': len(feature_classification.get('numeric_features', [])),
                'model_type': self.model_type,
                'dimension_compatibility': 'FULLY_COMPATIBLE' if dimension_validation['is_compatible'] else 'DIMENSION_MISMATCH',
                'analysis_timestamp': datetime.now().isoformat(),
                'shap_method': 'DeepLIFT (check_additivity=False)',
                'visualization_generated': {}
            },
            'metadata': comprehensive_results['analysis_metadata']
        }
        
        return final_results
    
    def save_results(self, results: Dict[str, Any], experiment_name: str = 'five_tier_analysis') -> str:
        """保存分析结果和可视化图表"""
        save_dir = os.path.join(
            self.config['logging']['save_dir'],
            self.config['logging']['name'],
            experiment_name
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 保存JSON结果
        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M')
        json_path = os.path.join(save_dir, f'five_tier_shap_epoch_0_results_{timestamp}.json')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 2. 生成可视化图表
        if self.config['shap']['save_plots']:
            pie_path, bar_path = self._generate_visualizations(results, save_dir, timestamp)
            results['analysis_summary']['visualization_generated'] = {
                'pie_chart': pie_path,
                'bar_chart': bar_path
            }
        
        # 3. 生成摘要报告
        summary_path = os.path.join(save_dir, f'five_tier_shap_epoch_0_summary_{timestamp}.txt')
        self._generate_summary_report(results, summary_path)
        
        logger.info(f"五大特征类别分析结果已保存到: {save_dir}")
        return save_dir
    
    def _generate_visualizations(self, results: Dict[str, Any], save_dir: str, timestamp: str) -> tuple:
        """生成可视化图表"""
        
        # 设置matplotlib
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        # 设置日志级别
        mpl_logger = logging.getLogger('matplotlib')
        font_logger = logging.getLogger('matplotlib.font_manager')
        original_mpl_level = mpl_logger.level
        original_font_level = font_logger.level
        mpl_logger.setLevel(logging.WARNING)
        font_logger.setLevel(logging.WARNING)
        
        pie_path = None
        bar_path = None
        
        try:
            # 1. 生成饼图 (Level 1 大类别分析)
            if self.config['shap']['enable_level1_analysis']:
                pie_path = self._generate_pie_chart(results, save_dir, timestamp)
            
            # 2. 生成柱状图 (Level 2 具体特征分析)
            if self.config['shap']['enable_level2_analysis']:
                bar_path = self._generate_bar_chart(results, save_dir, timestamp)
                
        finally:
            # 恢复日志级别
            mpl_logger.setLevel(original_mpl_level)
            font_logger.setLevel(original_font_level)
        
        return pie_path, bar_path
    
    def _generate_pie_chart(self, results: Dict[str, Any], save_dir: str, timestamp: str) -> str:
        """生成五大特征类别饼图"""
        category_importance = results['five_tier_analysis']['level1_category_importance']
        
        # 过滤掉重要性极小的类别（降低阈值）
        filtered_importance = {
            k: v for k, v in category_importance.items() 
            if v > 0.01  # 阈值：0.01%
        }
        
        if not filtered_importance:
            logger.warning("所有特征类别重要性都低于阈值，跳过饼图生成")
            logger.warning(f"原始重要性数据: {category_importance}")
            logger.warning(f"过滤后重要性数据: {filtered_importance}")
            return None
        
        plt.figure(figsize=(12, 10))
        
        # 灰度色彩方案
        colors = ['#404040', '#606060', '#808080', '#A0A0A0', '#C0C0C0']
        colors = colors[:len(filtered_importance)]
        
        # 密集纹理模式
        hatches = ['///', '\\\\\\\\', '|||', '---', '+++']
        hatches = hatches[:len(filtered_importance)]
        
        # 生成饼图
        wedges, texts, autotexts = plt.pie(
            filtered_importance.values(), 
            labels=[k.replace('_', ' ').title() for k in filtered_importance.keys()],
            colors=colors, 
            autopct='%1.1f%%', 
            startangle=90, 
            shadow=False,
            textprops={'fontsize': 14, 'fontweight': 'bold'}
        )
        
        # 为每个饼图块添加纹理
        for i, wedge in enumerate(wedges):
            wedge.set_hatch(hatches[i])
            wedge.set_edgecolor('black')
            wedge.set_linewidth(1.0)
        
        plt.tight_layout()
        
        pie_path = os.path.join(save_dir, f'shap_five_category_pie_{timestamp}.png')
        plt.savefig(pie_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"五大特征类别饼图已保存: {pie_path}")
        return pie_path
    
    def _generate_bar_chart(self, results: Dict[str, Any], save_dir: str, timestamp: str) -> str:
        """生成数值特征重要性柱状图"""
        numeric_importance = results['five_tier_analysis']['level2_numeric_importance']
        
        if not numeric_importance:
            logger.warning("没有数值特征重要性数据，跳过柱状图生成")
            return None
        
        # 转换为DataFrame并排序
        df = pd.DataFrame(list(numeric_importance.items()), columns=['Feature', 'Importance'])
        df = df.sort_values(by='Importance', ascending=False).head(20)
        
        plt.figure(figsize=(14, 10))
        
        # 灰度色彩
        colors = ['#404040'] * len(df)
        
        # 创建纹理模式
        hatches = ['///', '\\\\\\\\', '|||', '---', '+++', '...', 'xxx', 'ooo', 
                  '///', '\\\\\\\\', '|||', '---', '+++', '...', 'xxx', 'ooo', 
                  '///', '\\\\\\\\', '|||', '---']
        hatches = hatches[:len(df)]
        
        # 绘制条形图
        ax = sns.barplot(x='Importance', y='Feature', data=df, palette=colors, 
                       edgecolor='black', linewidth=1.0)
        
        # 添加纹理
        for i, bar in enumerate(ax.patches):
            bar.set_hatch(hatches[i])
            bar.set_alpha(0.9)
        
        # 添加数值标签
        for i, (imp, _) in enumerate(zip(df['Importance'], df['Feature'])):
            ax.text(imp + max(df['Importance']) * 0.01, i, f'{imp:.3f}', 
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.xlabel("mean(|SHAP value|)", fontsize=14, fontweight='bold')
        plt.ylabel("Feature Name", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        bar_path = os.path.join(save_dir, f'shap_numeric_top20_bar_{timestamp}.png')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"数值特征重要性柱状图已保存: {bar_path}")
        return bar_path
    
    def _generate_summary_report(self, results: Dict[str, Any], summary_path: str):
        """生成摘要报告"""
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== 五大特征类别SHAP分析报告 ===\n\n")
            
            summary = results['analysis_summary']
            f.write(f"模型类型: {summary['model_type']}\n")
            f.write(f"分析时间: {summary['analysis_timestamp']}\n")
            f.write(f"维度兼容性: {summary['dimension_compatibility']}\n")
            f.write(f"SHAP方法: {summary['shap_method']}\n\n")
            
            f.write("=== Level 1: 大类别重要性 ===\n")
            category_importance = results['five_tier_analysis']['level1_category_importance']
            for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{category}: {importance:.2f}%\n")
            
            f.write("\n=== Level 2: Top 10 数值特征 ===\n")
            numeric_importance = results['five_tier_analysis']['level2_numeric_importance']
            for i, (feature, importance) in enumerate(list(numeric_importance.items())[:10], 1):
                f.write(f"{i:2d}. {feature}: {importance:.2f}\n")
            
            f.write(f"\n分析完成，结果文件已保存。\n")
        
        logger.info(f"摘要报告已保存: {summary_path}")