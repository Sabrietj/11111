# models/shap_component.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 代理模型 Wrapper (用于 DeepExplainer 求导)
# ==============================================================================
class ShapFusionWrapper(nn.Module):
    """
    SHAP 专用包装器。
    输入：Embedding 向量 (Numeric, Domain, Seq, Text)
    输出：Logits
    功能：跳过 Tokenizer 和 Encoders，直接计算融合层和分类器的梯度。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, numeric_feats, domain_feats, seq_emb, text_emb):
        # 1. 表格特征路径
        if self.model.domain_embedding_enabled:
            tabular_input = torch.cat([numeric_feats, domain_feats], dim=1)
        else:
            tabular_input = numeric_feats
        
        tabular_out = self.model.tabular_projection(tabular_input)

        # 2. 序列特征路径 (seq_emb 已经是 SequenceEncoder 输出)
        if self.model.sequence_features_enabled:
            seq_out = self.model.sequence_projection(seq_emb)
        else:
            seq_out = torch.zeros_like(tabular_out)

        # 3. 文本特征路径 (text_emb 已经是 BERT CLS 输出)
        if self.model.text_features_enabled:
            text_out = text_emb
        else:
            text_out = torch.zeros_like(tabular_out)

        # 4. 多视图融合
        multiview_out = self.model._fuse_multi_views(seq_out, text_out, tabular_out)

        # 5. 分类器
        logits = self.model.classifier(multiview_out)
        return logits

# ==============================================================================
# 2. SHAP 分析器组件 (主逻辑类) - 增强版支持新架构
# ==============================================================================
class ShapAnalyzer:
    def __init__(self, model, use_enhanced_mode: bool = True):
        """
        初始化 SHAP 分析器
        :param model: FlowBertMultiview 模型实例 (需要访问其 config 和 encoders)
        :param use_enhanced_mode: 是否使用增强版五大特征类别分析架构
        """
        self.model = model
        self.cfg = model.cfg
        self.buffer = []
        self.sample_limit = 100  # 限制分析样本数，防止 OOM
        self.collected_count = 0
        self.enabled = True # 可以通过配置关闭
        
        # 增强模式配置
        self.use_enhanced_mode = use_enhanced_mode
        self.enhanced_analyzer = None
        
        if use_enhanced_mode:
            try:
                from .enhanced_shap_component import EnhancedShapAnalyzer
                self.enhanced_analyzer = EnhancedShapAnalyzer(
                    model=model, 
                    enable_five_tier_analysis=True
                )
                logger.info("ShapAnalyzer已启用增强版五大特征类别分析模式")
            except ImportError as e:
                logger.warning(f"无法导入增强版分析器，使用传统模式: {e}")
                self.use_enhanced_mode = False
        else:
            logger.info("ShapAnalyzer使用传统分析模式")

    def reset(self):
        """每个 epoch 开始时重置缓冲区"""
        self.buffer = []
        self.collected_count = 0
        
        # 同步重置增强版分析器
        if self.enhanced_analyzer is not None:
            self.enhanced_analyzer.reset()

    def collect_batch(self, batch):
        """收集测试阶段的 Batch 数据 (CPU 缓存)"""
        if not self.enabled or self.collected_count >= self.sample_limit:
            return

        # 移动到 CPU 并分离计算图，只保留数据
        batch_cpu = {}
        batch_size = 0
        
        with torch.no_grad():
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_cpu[k] = v.detach().cpu()
                    if batch_size == 0: batch_size = v.shape[0]
                elif isinstance(v, list):
                    batch_cpu[k] = v
                else:
                    batch_cpu[k] = v # 元数据等
        
        self.buffer.append(batch_cpu)
        self.collected_count += batch_size

    def finalize(self):
        """测试结束时执行 SHAP 分析"""
        # 只在主进程执行
        if hasattr(self.model, 'trainer') and not self.model.trainer.is_global_zero:
            return

        if not self.buffer:
            logger.warning("[ShapComponent] 未收集到样本，跳过分析")
            return

        logger.info("=" * 60)
        logger.info(f"🔍 [ShapComponent] 开始执行特征归因分析 (样本数: {self.collected_count})...")
        
        try:
            if self.use_enhanced_mode and self.enhanced_analyzer is not None:
                # 使用增强版五大特征类别分析
                logger.info("🚀 使用增强版五大特征类别分析架构")
                self.enhanced_analyzer.buffer = self.buffer
                self.enhanced_analyzer.collected_count = self.collected_count
                self.enhanced_analyzer.finalize()
            else:
                # 使用传统分析
                self._run_analysis()
        except Exception as e:
            logger.error(f"❌ [ShapComponent] 分析失败: {e}", exc_info=True)
        
        logger.info("=" * 60)

    def _run_analysis(self):
        import shap
        import matplotlib.pyplot as plt
        import seaborn as sns

        # ======================================================================
        # 🔴 终极修复: 退出 Inference Mode + 开启 Grad
        # PyTorch Lightning 的 test 阶段默认处于 inference_mode (比 no_grad 更强)
        # 必须显式退出 inference_mode 才能构建计算图
        # ======================================================================
        with torch.inference_mode(False):  # 1. 退出推理模式
            with torch.enable_grad():      # 2. 开启梯度计算
                
                try:
                    # 3. 再次确保模型参数允许求导 (双重保险)
                    for param in self.model.parameters():
                        param.requires_grad = True
                        
                    # 1. 合并 Buffer 数据
                    combined_batch = self._merge_buffer()
                    
                    # 2. 划分背景集 (Background) 和 解释集 (Eval)
                    total_samples = combined_batch['numeric_features'].shape[0]
                    bg_size = min(50, int(total_samples * 0.7))
                    eval_size = min(20, total_samples - bg_size)
                    
                    # 3. 预计算 Embeddings
                    device = self.model.device
                    self.model.eval() # 保持 eval 模式 (Dropout 不随机)

                    bg_inputs = self._precompute_embeddings(combined_batch, 0, bg_size, device)
                    eval_inputs = self._precompute_embeddings(combined_batch, bg_size, bg_size + eval_size, device)

                    # 初始化 Wrapper
                    wrapper = ShapFusionWrapper(self.model)

                    # ==========================================================
                    # 4. 计算图连通性自检 (Sanity Check)
                    # ==========================================================
                    logger.info("[ShapComponent] 执行计算图连通性自检...")
                    
                    # 手动前向传播
                    test_out = wrapper(*eval_inputs)
                    
                    # 检查输出是否依赖于输入
                    if test_out.grad_fn is None:
                        logger.error("❌ [Fatal] Wrapper 输出没有 grad_fn！计算图依然断裂。")
                        logger.error(f"当前梯度状态: is_grad_enabled={torch.is_grad_enabled()}, is_inference_mode={torch.is_inference_mode_enabled()}")
                        raise RuntimeError("无法构建计算图，请检查 PyTorch 版本或 Lightning 配置。")
                    
                    logger.info(f"✅ 计算图检查通过! Output grad_fn: {test_out.grad_fn}")

                    # 5. 初始化 DeepExplainer
                    explainer = shap.DeepExplainer(wrapper, bg_inputs)

                    logger.info("   正在计算 SHAP 值 (DeepLIFT)...")
                    # ==================================================================
                    # 🔴 关键修复: 关闭加性检查 (check_additivity=False)
                    # 原因: 模型包含 LayerNorm/GELU 等非线性层，DeepLIFT 无法完美分解，
                    # 导致贡献值之和与模型输出存在偏差。这是 Transformer 模型的常见现象。
                    # ==================================================================
                    shap_values = explainer.shap_values(eval_inputs, check_additivity=False)
                    
                    # 6. 处理输出格式
                    target_shap = shap_values
                    if isinstance(shap_values, list) and not isinstance(shap_values[0], np.ndarray) and not torch.is_tensor(shap_values[0]):
                        if len(shap_values) > 1:
                            target_shap = shap_values[1]

                    # 7. 聚合与绘图
                    feature_importance = self._aggregate_importance(target_shap)
                    self._plot_results(feature_importance)

                except Exception as e:
                    logger.error(f"SHAP 分析过程中发生错误: {e}", exc_info=True)

    def _merge_buffer(self):
        combined = {}
        keys = self.buffer[0].keys()
        for k in keys:
            val = self.buffer[0][k]
            if isinstance(val, torch.Tensor):
                combined[k] = torch.cat([b[k] for b in self.buffer], dim=0)
            elif isinstance(val, list):
                combined[k] = [item for b in self.buffer for item in b[k]]
        return combined

    def _precompute_embeddings(self, batch, start, end, device):
        """预计算 Embedding，返回 [Num, Dom, Seq, Text] 的 Tensor 列表"""
        # 构建切片并移至 GPU
        slice_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                slice_batch[k] = v[start:end].to(device)
            elif isinstance(v, list):
                slice_batch[k] = v[start:end]
        
        # 1. 提取 Embedding (在 no_grad 下进行，避免计算图过深)
        with torch.no_grad():
            # A. Numeric
            num_feats = slice_batch['numeric_features']
            
            # B. Domain
            if self.model.domain_embedding_enabled:
                dom_feats = slice_batch['domain_embedding_features']
            else:
                dom_feats = torch.zeros(num_feats.shape[0], 0, device=device)
            
            # C. Sequence
            if self.model.sequence_features_enabled:
                seq_data = {
                    'iat_times': slice_batch['iat_times'],
                    'payload_sizes': slice_batch['payload_sizes'],
                    'sequence_mask': slice_batch['sequence_mask']
                }
                # 注意：这里调用的是 encoder，不是 projection
                seq_emb = self.model.sequence_encoder(seq_data)["sequence_embedding"]
            else:
                seq_emb = torch.zeros(num_feats.shape[0], self.cfg.model.sequence.embedding_dim, device=device)
            
            # D. Text
            if self.model.text_features_enabled:
                text_emb = self.model._process_text_features(slice_batch)
            else:
                text_emb = torch.zeros(num_feats.shape[0], self.model.bert_config.hidden_size, device=device)
        
        # ======================================================================
        # 🔴 终极修复：强制类型转换 + 显式开启梯度
        # ======================================================================
        
        # ======================================================================
        # 🔴 终极修复：强制类型转换 + 显式开启梯度
        # ======================================================================
        raw_inputs = [num_feats, dom_feats, seq_emb, text_emb]
        final_inputs = []
        
        for t in raw_inputs:
            t = t.detach().clone()
            if not t.is_floating_point(): # 确保是浮点数
                t = t.float()
            t.requires_grad_(True) # 标记需要梯度
            final_inputs.append(t)

        # 验证所有输入张量都启用了梯度
        for i, t in enumerate(final_inputs):
            logger.debug(f"输入 {i} (shape: {t.shape}) requires_grad: {t.requires_grad}")

        return final_inputs

    def _aggregate_importance(self, target_shap):
        """聚合 SHAP 值到特征名和视图"""
        importance = {}
        
        # 数值特征 (1对1)
        shap_num = target_shap[0]
        if torch.is_tensor(shap_num): shap_num = shap_num.cpu().numpy()
        mean_shap_num = np.abs(shap_num).mean(axis=0)
        
        feat_names = self.cfg.data.numeric_features.flow_features
        for i, val in enumerate(mean_shap_num):
            if i < len(feat_names):
                importance[feat_names[i]] = float(val)

        # 视图聚合函数
        def agg_view(tensor, name):
            if torch.is_tensor(tensor): tensor = tensor.cpu().numpy()
            score = np.abs(tensor).sum(axis=1).mean() # Sum dim -> Mean batch
            importance[name] = float(score)

        if self.model.domain_embedding_enabled:
            agg_view(target_shap[1], 'View: Domain')
        if self.model.sequence_features_enabled:
            agg_view(target_shap[2], 'View: Packet Seq')
        if self.model.text_features_enabled:
            agg_view(target_shap[3], 'View: Text Metadata')
            
        return importance

    def _plot_results(self, importance):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 设置matplotlib为静默模式
        import logging
        mpl_logger = logging.getLogger('matplotlib')
        font_logger = logging.getLogger('matplotlib.font_manager')
        original_level = mpl_logger.level
        original_font_level = font_logger.level
        mpl_logger.setLevel(logging.WARNING)
        font_logger.setLevel(logging.WARNING)
        
        try:
            # 设置Times New Roman字体
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['xtick.labelsize'] = 12
            plt.rcParams['ytick.labelsize'] = 12
            plt.rcParams['legend.fontsize'] = 12
            logger.info("ShapComponent字体已设置为Times New Roman")
        except Exception as e:
            logger.debug(f"ShapComponent字体设置失败，使用默认字体: {e}")
            plt.rcParams['font.family'] = 'serif'
        
        # 路径设置
        if hasattr(self.model.logger, 'log_dir'):
            save_dir = os.path.join(self.model.logger.log_dir, "shap_analysis")
        else:
            save_dir = "shap_analysis"
        os.makedirs(save_dir, exist_ok=True)

        # 转换为 DataFrame
        df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
        
        # ======================================================================
        # 1. Top 20 棒图 (Bar Chart) - 灰度+纹理版本
        # 目标: 只展示具体的特征，排除宏观的 "View: ..." 聚合特征
        # ======================================================================
        
        # 过滤掉所有以 "View:" 开头的特征名
        df_bar = df[~df['Feature'].str.startswith('View:')]
        
        # 排序并取前 20
        df_bar = df_bar.sort_values(by='Importance', ascending=False).head(20)

        plt.figure(figsize=(12, 10))
        if len(df_bar) > 0:
            # 创建灰度色彩
            colors = ['#404040'] * len(df_bar)  # 深灰色
            
            # 创建不同的纹理模式
            hatches = ['///', '\\\\\\\\', '|||', '---', '+++', '...', 'xxx', 'ooo', '///', '\\\\\\\\', 
                      '|||', '---', '+++', '...', 'xxx', 'ooo', '///', '\\\\\\\\', '|||', '---']
            hatches = hatches[:len(df_bar)]
            
            # 绘制条形图
            ax = sns.barplot(x='Importance', y='Feature', data=df_bar, palette=colors, 
                           edgecolor='black', linewidth=1.0)
            
            # 添加纹理到每个条形
            for i, bar in enumerate(ax.patches):
                bar.set_hatch(hatches[i])
                bar.set_alpha(0.9)
            
            # 添加数字标签
            for i, (imp, _) in enumerate(zip(df_bar['Importance'], df_bar['Feature'])):
                ax.text(imp + max(df_bar['Importance']) * 0.01, i, f'{imp:.2f}', 
                       ha='left', va='center', fontsize=10, fontweight='bold')
            
            # plt.title("SHAP Feature Importance (Top 20 Features)", fontsize=16, fontweight='bold', pad=20)
            plt.xlabel("mean(|SHAP value|)", fontsize=14, fontweight='bold')
            plt.ylabel("Feature Name", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "shap_top20.png"), dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        else:
            logger.warning("[ShapComponent] 没有具体的数值特征可供绘制 Top 20 棒图")
        plt.close()

        # ======================================================================
        # 2. 视图饼图 (Pie Chart) - 灰度+密集纹理版本
        # 目标: 展示宏观视图的贡献 (Numeric vs Sequence vs Text vs Domain)
        # ======================================================================
        
        view_scores = {'Numeric': 0.0, 'Sequence': 0.0, 'Text': 0.0, 'Domain': 0.0}
        
        # 使用原始的 df (包含 View 特征) 进行统计
        for idx, row in df.iterrows():
            name = row['Feature']
            val = row['Importance']
            
            # 如果是聚合特征，直接加到对应的视图分数中
            if 'View: Packet Seq' in name: 
                view_scores['Sequence'] += val
            elif 'View: Text Metadata' in name: 
                view_scores['Text'] += val
            elif 'View: Domain' in name: 
                view_scores['Domain'] += val
            # 如果不是聚合特征（即具体的数值特征），加到 Numeric 分数中
            elif not name.startswith('View:'): 
                view_scores['Numeric'] += val
        
        # 过滤掉贡献极小的视图
        view_scores = {k: v for k, v in view_scores.items() if v > 1e-6}
        
        if view_scores:
            plt.figure(figsize=(10, 8))
            
            # 灰度色彩方案
            colors = ['#404040', '#606060', '#808080', '#A0A0A0']
            colors = colors[:len(view_scores)]
            
            # 密集纹理模式
            hatches = ['///', '\\\\\\\\', '|||', '---']
            hatches = hatches[:len(view_scores)]
            
            # 生成饼图
            wedges, texts, autotexts = plt.pie(view_scores.values(), labels=view_scores.keys(), 
                                             colors=colors, autopct='%1.1f%%', 
                                             startangle=90, shadow=False,
                                             textprops={'fontsize': 14, 'fontweight': 'bold'})
            
            # 为每个饼图块添加密集纹理
            for i, wedge in enumerate(wedges):
                wedge.set_hatch(hatches[i])
                wedge.set_edgecolor('black')
                wedge.set_linewidth(1.0)
            
            # plt.title("Global View Contribution", fontsize=18, fontweight='bold', pad=30)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "shap_view_pie.png"), dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
        
        logger.info(f"✅ [ShapComponent] 图表已保存至: {save_dir}")
        
        # 恢复原始日志级别
        mpl_logger.setLevel(original_level)
        font_logger.setLevel(original_font_level)