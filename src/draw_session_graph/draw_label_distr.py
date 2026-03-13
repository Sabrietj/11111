# -*- coding: utf-8 -*-
import datetime
import sys
import numpy as np
import argparse
import json
from dgl.data.utils import load_graphs, load_info

import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

import matplotlib.pyplot as plt


# 添加 ../utils 到路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
import config_manager

# =========================
# 数据加载与标签分布统计
# =========================
class LabelDistributionAnalyzer:
    def __init__(self, bin_path=None):
        self.bin_path = bin_path
        self.graphs = []
        self.label_id_list = []
        self.label_name_set = []
        self.train_index = []
        self.test_index = []
        self.validate_index = []
        
        if bin_path and os.path.exists(bin_path):
            self._load_data(bin_path)

    def _load_data(self, dumpFilename):
        """加载数据并处理标签信息 - 保持原始ID"""
        self.graphs, _ = load_graphs(dumpFilename)

        info_path = dumpFilename.replace(".bin", "_info.pkl")
        if not os.path.exists(info_path):
            info_path = dumpFilename + "__info.pkl"
        data = load_info(info_path)

        self.label_name_list = data.get('label_name', [])
        self.label_id_list = data.get('label_id', [])

        # 保持原始标签ID，不进行重映射
        def _to_int(v):
            try:
                return int(v)
            except Exception:
                raise ValueError(f"label_id 包含无法转换为整数的值: {v!r}")
        
        self.label_id_list = list(map(_to_int, self.label_id_list))
        
        # 获取所有存在的标签ID和对应的名称
        unique_ids = sorted(set(self.label_id_list))
        id_to_name = {}
        
        # 构建ID到名称的映射
        for label_id, label_name in zip(self.label_id_list, self.label_name_list):
            if label_id not in id_to_name:
                id_to_name[label_id] = str(label_name)
        
        # 按ID顺序创建标签名称集合
        self.label_name_set = [id_to_name.get(label_id, f"Unknown_{label_id}") for label_id in unique_ids]
        
        print(f"原始标签ID: {unique_ids}")
        print(f"标签名称: {self.label_name_set}")
        print(f"标签对应关系: {dict(zip(unique_ids, self.label_name_set))}")

        self.train_index = data.get('train_index', [])
        self.test_index = data.get('test_index', [])
        self.validate_index = data.get('validate_index', [])

        print(f"Loaded {len(self.graphs)} graphs. Train: {len(self.train_index)}, "
            f"Valid: {len(self.validate_index)}, Test: {len(self.test_index)}")
        
    def get_label_distribution(self):
        """获取标签分布统计 - 保持原始ID"""
        labels = np.array(self.label_id_list)
        
        # 获取所有存在的标签ID
        unique_ids = sorted(set(self.label_id_list))
        
        distributions = {}
        
        splits = {
            'train': self.train_index,
            'valid': self.validate_index, 
            'test': self.test_index
        }

        for split_name, indices in splits.items():
            if not indices:
                distributions[split_name] = {'counts': [], 'percentages': [], 'total': 0}
                continue

            split_labels = labels[indices]
            unique, counts = np.unique(split_labels, return_counts=True)
            class_count_dict = dict(zip(unique, counts))
            
            # 按原始ID顺序创建计数数组
            counts_array = []
            percentages = []
            
            for label_id in unique_ids:  # 按原始ID顺序：2,3,4,5
                count = class_count_dict.get(label_id, 0)
                counts_array.append(int(count))
                percentages.append(float(count) / len(indices) * 100)
            
            distributions[split_name] = {
                'counts': counts_array,
                'percentages': percentages, 
                'total': int(len(indices))
            }
        
        return distributions, unique_ids, self.label_name_set

    def get_detailed_statistics(self):
        """获取详细的统计信息"""
        distributions, label_ids, label_names = self.get_label_distribution()
        
        # 修复：num_classes应该是标签数量
        num_classes = len(label_names)
        
        stats = {
            'total_graphs': int(len(self.graphs)),
            'num_classes': int(num_classes),  # 现在是整数
            'class_names': label_names,
            'label_ids': label_ids,  # 添加标签ID列表
            'split_sizes': {
                'train': int(len(self.train_index)),
                'valid': int(len(self.validate_index)),
                'test': int(len(self.test_index))
            }
        }
        
        for split_name in ['train', 'valid', 'test']:
            if split_name in distributions:
                stats[f'{split_name}_distribution'] = distributions[split_name]
        
        return stats

    @staticmethod
    def to_distribution_string(distributions, label_names, data_type="label"):
        """格式化输出标签分布"""
        out = [f"Label Distribution Analysis", "="*50]
        
        for split_name, dist_data in distributions.items():
            out.append(f"\n{split_name.upper()} Set ({dist_data['total']} samples):")
            out.append(f"{'Class ID':<10} {'Class Name':<20} {'Count':<8} {'Percent':<10}")
            out.append("-" * 50)
            
            # 修复：按标签名称索引而不是按ID索引
            for cls_id in range(len(label_names)):
                count = dist_data['counts'][cls_id]
                percent = dist_data['percentages'][cls_id]
                class_name = label_names[cls_id] if cls_id < len(label_names) else f"Unknown_{cls_id}"
                out.append(f"{cls_id:<10} {class_name:<20} {count:<8} {percent:6.2f}%")
        
        return "\n".join(out)

    def numpy_to_python(self, obj):
        """将NumPy数据类型转换为Python原生数据类型"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.numpy_to_python(item) for item in obj]
        else:
            return obj

    def load_or_compute_distribution(self, json_file, dataset_name, use_cache=True):
        """从JSON缓存读取或从bin统计"""
        if use_cache and os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                key = f"{dataset_name}_label_distribution"
                if key in data:
                    print(f"Loaded cached label distribution from {json_file}")
                    # 这里可以添加解析缓存数据的逻辑
                    # 暂时直接重新计算以确保数据正确
                    pass
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading cache file {json_file}: {e}. Recomputing distribution...")

        # 从二进制文件统计
        print(f"Computing label distribution from .bin file...")
        distributions, label_ids, label_names = self.get_label_distribution()
        
        dist_str = self.to_distribution_string(distributions, label_names)

        # 保存缓存 - 确保所有数据都是JSON可序列化的
        os.makedirs(os.path.dirname(json_file) or ".", exist_ok=True)
        
        # 读取现有数据或创建新字典
        if os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                data = {}
        else:
            data = {}
            
        key = f"{dataset_name}_label_distribution"
        
        # 修复：num_classes 应该是标签数量，不是标签ID列表
        num_classes = len(label_names)  # 这才是正确的类别数量
        
        # 确保数据可JSON序列化
        cache_data = {
            'distributions': self.numpy_to_python(distributions),
            'num_classes': int(num_classes),  # 现在num_classes是整数
            'label_ids': self.numpy_to_python(label_ids),  # 添加label_ids字段
            'label_names': label_names,
            'distribution_string': dist_str
        }
        
        data[key] = cache_data
        
        try:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Label distribution saved to {json_file}")
        except TypeError as e:
            print(f"Error saving cache: {e}")
            # 如果仍然有序列化问题，尝试更彻底的转换
            data_serializable = json.loads(json.dumps(data, default=str))
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data_serializable, f, ensure_ascii=False, indent=2)
            print(f"Label distribution saved to {json_file} (with fallback serialization)")
        
        return distributions, label_ids, label_names


# =========================
# 绘图函数
# =========================
def plot_label_distribution_bar(dataset_name, distributions, label_names, split_name, figsize=(7.0, 3.5)):
    """绘制标签分布的柱状图"""
    plt.figure(figsize=figsize)
    
    dist_data = distributions[split_name]
    counts = dist_data['counts']
    total_samples = dist_data['total']
    
    # 创建类别标签
    x_labels = [f"{name}\n(ID:{i})" for i, name in enumerate(label_names)]
    x_pos = np.arange(len(x_labels))
    
    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(x_labels)))
    
    plot_counts = [c if c > 0 else 0.1 for c in counts]
    # 绘制柱状图
    bars = plt.bar(x_pos, plot_counts, color=colors, edgecolor='black', alpha=1.0)
    
    plt.xlabel('Class Labels')
    plt.yscale('log', nonpositive='clip', base=10)
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='auto'))
    plt.ylabel('Number of Samples - Log Scale')
    plt.title(f'Label Distribution - {split_name.upper()} Set\n{dataset_name} (Total: {total_samples} samples)')
    plt.xticks(x_pos, x_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return plt

def plot_combined_label_distribution(dataset_name, distributions, label_names, figsize=(7.0, 3.5)):
    """绘制三个数据集的组合标签分布图"""
    plt.figure(figsize=figsize)
    
    x_pos = np.arange(len(label_names))
    width = 0.25
    
    # [关键修改1] 获取百分比数据而不是绝对数量
    train_data = distributions['train']['percentages'] if 'train' in distributions else [0] * len(label_names)
    train_data = [p if p > 0 else 1e-4 for p in train_data]

    valid_data = distributions['valid']['percentages'] if 'valid' in distributions else [0] * len(label_names)
    valid_data = [p if p > 0 else 1e-4 for p in valid_data]

    test_data = distributions['test']['percentages'] if 'test' in distributions else [0] * len(label_names)
    test_data = [p if p > 0 else 1e-4 for p in test_data]

    # 绘制柱状图
    bars1 = plt.bar(x_pos - width, train_data, width, label='Train', 
                   color='steelblue', alpha=1.0, edgecolor='black')
    bars2 = plt.bar(x_pos, valid_data, width, label='Valid', 
                   color='orange', alpha=1.0, edgecolor='black')
    bars3 = plt.bar(x_pos + width, test_data, width, label='Test', 
                   color='green', alpha=1.0, edgecolor='black')
    
    # 为每个数据集准备数据
    # train_counts = distributions['train']['counts'] if 'train' in distributions else [0] * len(label_names)
    # validate_counts = distributions['valid']['counts'] if 'valid' in distributions else [0] * len(label_names)
    # test_counts = distributions['test']['counts'] if 'test' in distributions else [0] * len(label_names)
    
    # 绘制三个数据集的柱状图
    #bars1 = plt.bar(x_pos - width, train_counts, width, label='Train', 
    #               color='steelblue', alpha=1.0, edgecolor='black')
    #bars2 = plt.bar(x_pos, validate_counts, width, label='Valid', 
    #               color='orange', alpha=1.0, edgecolor='black')
    #bars3 = plt.bar(x_pos + width, test_counts, width, label='Test', 
    #               color='green', alpha=1.0, edgecolor='black')
    
    plt.xlabel('Class Labels')
    # plt.ylabel('Number of Samples')
    # [关键修改2] 设置Y轴为对数坐标，这样才能看清 Heartbleed 等稀有类别
    plt.yscale('log', nonpositive='clip', base=10)
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='auto'))
    plt.ylabel('Percentage (%) - Log Scale')

    plt.title(f'Label Distribution Comparison - {dataset_name}')
    plt.xticks(x_pos, [f"{name}\n(ID:{i})" for i, name in enumerate(label_names)], 
               rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # [关键修改3] 添加数值标签（仅显示大于0的值）
    # 由于是对数轴，标签位置稍微上移一点
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                # 对于极小值（如 <0.1%），显示更精确的小数位
                label_text = f'{height:.2f}%' if height < 1 else f'{height:.1f}%'
                plt.text(bar.get_x() + bar.get_width()/2., height * 1.1, # 对数轴上乘法偏移比加法偏移更自然
                        label_text, ha='center', va='bottom', fontsize=7, rotation=90)
                
    # 添加数值标签（只显示非零值）
    #for bars in [bars1, bars2, bars3]:
    #    for bar in bars:
    #        height = bar.get_height()
    #        if height > 0:
    #            plt.text(bar.get_x() + bar.get_width()/2., height + max(max(train_counts), max(validate_counts), max(test_counts))*0.01,
    #                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return plt


def plot_label_distribution_percentage(dataset_name, distributions, label_ids, label_names, split_name, figsize=(7.0, 3.5)):
    """绘制标签分布的百分比柱状图 - 使用原始ID"""
    
    dist_data = distributions[split_name]
    percentages = dist_data['percentages']
    total_samples = dist_data['total']
    
    # 使用原始标签ID和对应的名称
    x_labels = [f"{name}\n(ID:{label_id})" for label_id, name in zip(label_ids, label_names)]
    x_pos = np.arange(len(x_labels))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(x_labels)))
    
    percentages = [p if p > 0 else 1e-4 for p in percentages]

    bars = plt.bar(x_pos, percentages, color=colors, edgecolor='black', alpha=1.0)
    
    plt.xlabel('Class Labels')
    plt.yscale('log', nonpositive='clip', base=10)
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='auto'))
    plt.ylabel('Percentage (%) - Log Scale')
    plt.title(f'Label Distribution (Percentage) - {split_name.upper()} Set\n{dataset_name} (Total: {total_samples} samples)')
    plt.xticks(x_pos, x_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加百分比标签
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        if percentage > 0:  # 只显示非零值
            plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return plt


def print_detailed_report(analyzer, dataset_name):
    """打印详细的标签分布报告"""
    stats = analyzer.get_detailed_statistics()
    distributions, label_ids, label_names = analyzer.get_label_distribution()
    
    print("\n" + "="*60)
    print(f"LABEL DISTRIBUTION REPORT - {dataset_name}")
    print("="*60)
    print(f"Total graphs: {stats['total_graphs']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Label IDs: {label_ids}")  # 显示实际的标签ID
    print(f"Train set size: {stats['split_sizes']['train']}")
    print(f"Valid set size: {stats['split_sizes']['valid']}")
    print(f"Test set size: {stats['split_sizes']['test']}")
    
    for split_name in ['train', 'valid', 'test']:
        if split_name in distributions:
            dist_data = distributions[split_name]
            print(f"\n{split_name.upper()} Set Distribution:")
            for cls_idx, label_id in enumerate(label_ids):
                count = dist_data['counts'][cls_idx]
                percent = dist_data['percentages'][cls_idx]
                if count > 0:  # 只显示有样本的类别
                    print(f"  Class {label_id} ({label_names[cls_idx]}): {count} samples ({percent:.2f}%)")
    
    print("="*60 + "\n")


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser(description='Draw label distribution for dataset.')
    parser.add_argument('--type', '-t', choices=['separate', 'combined', 'percentage', 'all'], 
                       default='separate', help='Plot type: separate, combined, percentage, or all')
    parser.add_argument('--split', '-s', choices=['train', 'valid', 'test', 'all'], 
                       default='all', help='Which split to plot: train, valid, test, or all')
    parser.add_argument('--no-cache', action='store_true', help='Recalculate even if JSON cache exists')
    parser.add_argument('--report', action='store_true', help='Print detailed report of label distribution')
    parser.add_argument("--graph_file_name", type=str, 
                        default="all_session_graph", help="Graph file prefix without suffix, e.g. all_session_graph__xxx")

    args = parser.parse_args()

    # 获取配置
    plot_data_path = config_manager.read_plot_data_path_config()
    concurrent_flow_iat_threshold = config_manager.read_concurrent_flow_iat_threshold()
    sequential_flow_iat_threshold = config_manager.read_sequential_flow_iat_threshold()    

    dataset_name = os.path.basename(plot_data_path.rstrip("/\\"))
    # graph_file_path = os.path.join(dataset_dir, "all_session_graph.bin")
    graph_prefix = args.graph_file_name
    graph_file_path = os.path.join(plot_data_path, f"{graph_prefix}.bin")    
    
    if not os.path.exists(graph_file_path):
        print(f"Error: input file not found: {graph_file_path}")
        return
    
    # 初始化分析器
    analyzer = LabelDistributionAnalyzer(graph_file_path)
    
    # 打印详细的报告
    if args.report:
        print_detailed_report(analyzer, dataset_name)
    
    # JSON文件路径
    json_file = os.path.join(plot_data_path, 
                            f'label_distribution_{concurrent_flow_iat_threshold}_{sequential_flow_iat_threshold}.json')
    
    # 创建输出目录
    figs_dir = os.path.join('.', 'figs', dataset_name)
    os.makedirs(figs_dir, exist_ok=True)
    
    # 加载或计算分布数据
    distributions, label_ids, label_names = analyzer.load_or_compute_distribution(
        json_file, dataset_name, not args.no_cache
    )
    
    # 确定要绘制的数据集分割
    splits_to_plot = []
    if args.split == 'all':
        splits_to_plot = ['train', 'valid', 'test']
    else:
        splits_to_plot = [args.split]
    
    # 根据类型生成相应的图
    plot_types = []
    if args.type == 'all':
        plot_types = ['separate', 'combined', 'percentage']
    else:
        plot_types = [args.type]
    
    # 生成图表
    for plot_type in plot_types:
        if plot_type == 'separate':
            # 分别绘制每个数据集的分布图
            for split_name in splits_to_plot:
                if split_name in distributions and distributions[split_name]['total'] > 0:
                    plt_separate = plot_label_distribution_bar(dataset_name, distributions, label_names, split_name)
                    
                    fig_path = os.path.join(figs_dir, 
                                          f'label_distribution_{split_name}_{concurrent_flow_iat_threshold}_{sequential_flow_iat_threshold}.pdf')
                    plt_separate.savefig(fig_path, format='pdf', bbox_inches='tight')
                    print(f"Label distribution plot for {split_name} saved to: {fig_path}")
                    plt_separate.show()
                else:
                    print(f"Warning: No data available for {split_name} split")
        
        elif plot_type == 'combined':
            # 绘制组合分布图
            if any(split_name in distributions and distributions[split_name]['total'] > 0 for split_name in splits_to_plot):
                plt_combined = plot_combined_label_distribution(dataset_name, distributions, label_names)
                
                fig_path = os.path.join(figs_dir, 
                                      f'label_distribution_combined_{concurrent_flow_iat_threshold}_{sequential_flow_iat_threshold}.pdf')
                plt_combined.savefig(fig_path, format='pdf', bbox_inches='tight')
                print(f"Combined label distribution plot saved to: {fig_path}")
                plt_combined.show()
            else:
                print("Warning: No data available for combined plot")
        
        elif plot_type == 'percentage':
            # 绘制百分比分布图
            for split_name in splits_to_plot:
                if split_name in distributions and distributions[split_name]['total'] > 0:
                    # 绘图时传递正确的参数
                    plt_percentage = plot_label_distribution_percentage(
                        dataset_name, distributions, label_ids, label_names, split_name
                    )
                    
                    fig_path = os.path.join(figs_dir, 
                                          f'label_distribution_percentage_{split_name}_{concurrent_flow_iat_threshold}_{sequential_flow_iat_threshold}.pdf')
                    plt_percentage.savefig(fig_path, format='pdf', bbox_inches='tight')
                    print(f"Percentage label distribution plot for {split_name} saved to: {fig_path}")
                    plt_percentage.show()
                else:
                    print(f"Warning: No data available for {split_name} split")
    
    print("All label distribution plots generated successfully!")


if __name__ == "__main__":
    main()