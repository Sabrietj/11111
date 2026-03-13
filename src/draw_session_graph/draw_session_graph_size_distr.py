# -*- coding: utf-8 -*-
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs, load_info

# 添加 ../utils 到路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)    
import config_manager as ConfigManager

import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# =========================
# 数据加载与分布统计
# =========================
class DatasetAnalyzer:
    def __init__(self, bin_path=None):
        self.bin_path = bin_path
        self.graphs = []
        if bin_path and os.path.exists(bin_path):
            self.graphs, _ = load_graphs(bin_path)
            logger.info(f"Loaded {len(self.graphs)} graphs from {bin_path}")

    def get_max_graph_size(self):
        """返回最大节点数和最大边数"""
        if not self.graphs:
            return 0, 0

        max_nodes = max(g.num_nodes() for g in self.graphs)
        max_edges = max(g.num_edges() for g in self.graphs)

        return max_nodes, max_edges

    def get_percentile_graph_size(self, percentile=99):
        if not self.graphs:
            return (0, 0)

        node_counts = np.array([g.num_nodes() for g in self.graphs])
        edge_counts = np.array([g.num_edges() for g in self.graphs])

        return (
            int(np.percentile(node_counts, percentile)),
            int(np.percentile(edge_counts, percentile))
        )

    def get_node_count_distribution(self):
        """统计节点数量分布"""
        node_counts = [g.num_nodes() for g in self.graphs]
        unique, counts = np.unique(node_counts, return_counts=True)
        total_graphs = len(self.graphs)
        graph_count_dict = dict(zip(unique, counts))
        return graph_count_dict, total_graphs

    def get_edge_count_distribution(self):
        """统计边数量分布"""
        edge_counts = [g.num_edges() for g in self.graphs]
        unique, counts = np.unique(edge_counts, return_counts=True)
        total_graphs = len(self.graphs)
        edge_count_dict = dict(zip(unique, counts))
        return edge_count_dict, total_graphs

    def get_detailed_statistics(self):
        """获取详细的统计信息"""
        node_counts = [g.num_nodes() for g in self.graphs]
        edge_counts = [g.num_edges() for g in self.graphs]
        
        # 统计特殊情况
        single_node_graphs = sum(1 for n in node_counts if n == 1)
        zero_edge_graphs = sum(1 for e in edge_counts if e == 0)
        single_node_zero_edge = sum(1 for i in range(len(node_counts)) 
                                 if node_counts[i] == 1 and edge_counts[i] == 0)
        
        return {
            'total_graphs': len(self.graphs),
            'single_node_graphs': single_node_graphs,
            'zero_edge_graphs': zero_edge_graphs,
            'single_node_zero_edge': single_node_zero_edge,
            'single_node_percent': single_node_graphs / len(self.graphs) * 100,
            'zero_edge_percent': zero_edge_graphs / len(self.graphs) * 100
        }

    @staticmethod
    def parse_distribution_string(distribution_str):
        """从字符串解析分布数据"""
        lines = distribution_str.strip().split('\n')
        counts_dict = {}
        total_graphs = 0
        for line in lines:
            if line.startswith("Total Graphs:"):
                total_graphs = int(line.split(":")[1])
            elif line and line.split()[0].isdigit():
                parts = line.split()
                counts_dict[int(parts[0])] = int(parts[1])
        return counts_dict, total_graphs

    @staticmethod
    def to_distribution_string(count_dict, total_graphs, data_type="nodes"):
        """格式化输出"""
        if data_type == "nodes":
            title = "Nodes"
        else:
            title = "Edges"
            
        out = [f"Total Graphs: {total_graphs}", f"{title}      Count    Percent   ", "-----------------------------------"]
        for n in sorted(count_dict.keys()):
            c = count_dict[n]
            p = c / total_graphs * 100
            out.append(f"{n:<10} {c:<8} {p:6.2f}%")
        return "\n".join(out)

    def load_or_compute_distribution(self, json_file, dataset_name, use_cache=True, data_type="nodes"):
        """从JSON缓存读取或从bin统计"""
        if use_cache and os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = f"{dataset_name}_{data_type}"
            if key in data:
                logger.info(f"Loaded cached {data_type} distribution from {json_file}")
                return self.parse_distribution_string(data[key])

        # 从二进制文件统计
        logger.info(f"Computing {data_type} distribution from .bin file...")
        if data_type == "nodes":
            count_dict, total_graphs = self.get_node_count_distribution()
        else:
            count_dict, total_graphs = self.get_edge_count_distribution()
            
        dist_str = self.to_distribution_string(count_dict, total_graphs, data_type)

        # 保存缓存
        os.makedirs(os.path.dirname(json_file) or ".", exist_ok=True)
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
            
        key = f"{dataset_name}_{data_type}"
        data[key] = dist_str
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"{data_type.capitalize()} distribution saved to {json_file}")
        return count_dict, total_graphs


# =========================
# 绘图函数 - 处理零值的特殊版本
# =========================
def plot_loglog_dot_handled_zeros(dataset_name, counts_dict, total_graphs, data_type="nodes"):
    """绘制对数散点图，专门处理零值情况"""
    plt.figure(figsize=(10, 7))
    
    # 分离零值和非零值
    zero_count = counts_dict.get(0, 0)
    non_zero_items = [n for n in counts_dict.keys() if n > 0]
    non_zero_counts = [counts_dict[n] for n in non_zero_items if counts_dict[n] > 0]
    
    if data_type == "nodes":
        xlabel = 'Number of Nodes'
        title = 'Node'
        color = 'steelblue'
        zero_label = '1 node'  # 对于节点，0通常表示1个节点
    else:
        xlabel = 'Number of Edges'
        title = 'Edge'
        color = 'coral'
        zero_label = '0 edges'
    
    # 绘制非零值的散点图（对数坐标）
    if non_zero_items:
        plt.scatter(non_zero_items, non_zero_counts, color=color, alpha=0.7, 
                   edgecolors='black', s=30, label=f'Non-zero {title.lower()} counts')
    
    # 特殊处理零值 - 在x=0.5的位置显示（避免log(0)的问题）
    if zero_count > 0:
        # 在x轴左侧添加一个特殊标记
        plt.scatter([0.1], [zero_count], color='red', alpha=0.8, marker='X', 
                   s=100, label=f'{zero_label} graphs: {zero_count}')
        
        # 添加注释说明
        plt.annotate(f'{zero_count} graphs with {zero_label}', 
                    xy=(0.1, zero_count), xytext=(10, 20),
                    textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 设置对数坐标轴
    plt.xscale('log')
    plt.yscale('log')
    
    # 调整x轴范围，为特殊标记留出空间
    if zero_count > 0:
        x_min = 0.05  # 为特殊标记留出空间
    else:
        x_min = min(non_zero_items) * 0.8 if non_zero_items else 1
    
    x_max = max(non_zero_items) * 1.2 if non_zero_items else 100
    plt.xlim(x_min, x_max)
    
    plt.xlabel(f'{xlabel} (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title(f'{title} Size Distribution - {dataset_name}')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)
    plt.tight_layout()
    return plt


def plot_normal_bar_with_zeros(dataset_name, counts_dict, total_graphs, maxshow=100, data_type="nodes"):
    """绘制普通柱状图，包含零值"""
    # 创建分组
    bin_labels = ['0']
    bin_sums = [counts_dict.get(0, 0)]
    
    # 添加其他分组
    bin_edges = list(range(1, maxshow + 1, 5))
    for start in bin_edges:
        end = start + 4
        s = sum(counts_dict.get(n, 0) for n in range(start, end + 1))
        bin_labels.append(f"{start}–{end}")
        bin_sums.append(s)
    
    over = sum(c for n, c in counts_dict.items() if n > maxshow)
    if over > 0:
        bin_labels.append(f">{maxshow}")
        bin_sums.append(over)
    
    # 转换为千分比
    bin_permille = [s / total_graphs * 1000 for s in bin_sums]

    plt.figure(figsize=(10, 6))
    
    if data_type == "nodes":
        color = 'steelblue'
        title = 'Node'
        zero_label = '1 node'
    else:
        color = 'coral'
        title = 'Edge'
        zero_label = '0 edges'
    
    # 为0值使用特殊颜色
    colors = ['red' if label == '0' else color for label in bin_labels]
    
    bars = plt.bar(bin_labels, bin_permille, color=colors, edgecolor='black', alpha=0.8)
    plt.xlabel(f'Number of {title}s')
    plt.ylabel('Frequency (‰)')
    plt.title(f'{title} Size Distribution - {dataset_name}\n(Red bar indicates {zero_label} graphs)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar, val, label in zip(bars, bin_permille, bin_labels):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3, 
                f"{val:.1f}‰", ha='center', va='bottom', fontsize=8,
                fontweight='bold' if label == '0' else 'normal')
    
    plt.tight_layout()
    return plt


def log_special_cases_report(analyzer, dataset_name):
    """打印特殊情况的报告"""
    stats = analyzer.get_detailed_statistics()
    
    logger.info("="*60)
    logger.info(f"SPECIAL CASES REPORT - {dataset_name}")
    logger.info("="*60)
    logger.info(f"Total graphs: {stats['total_graphs']}")
    logger.info(f"Graphs with 1 node: {stats['single_node_graphs']} ({stats['single_node_percent']:.2f}%)")
    logger.info(f"Graphs with 0 edges: {stats['zero_edge_graphs']} ({stats['zero_edge_percent']:.2f}%)")
    logger.info(f"Graphs with 1 node and 0 edges: {stats['single_node_zero_edge']}")
    logger.info("="*60)


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser(description='Draw session graph node and edge number distribution.')
    parser.add_argument('--type', '-t', choices=['nodes', 'edges', 'both', 'comparison'], 
                       default='both', help='Plot type: nodes, edges, both, or comparison')
    parser.add_argument('--plot-style', '-p', choices=['loglog_dot', 'normal_bar', 'both'], 
                       default='loglog_dot', help='Plot style: loglog_dot, normal_bar, or both')
    parser.add_argument('--no-cache', action='store_true', help='Recalculate even if JSON cache exists')
    parser.add_argument('--maxshow', type=int, default=100, help='Max count for normal_bar plot')
    parser.add_argument('--report', action='store_true', help='logger.info detailed report of special cases')
    parser.add_argument("--graph_file_name", type=str, 
                        default="all_session_graph", help="Graph file prefix without suffix, e.g. all_session_graph__xxx")

    args = parser.parse_args()

    # 获取配置
    dataset_dir = ConfigManager.read_plot_data_path_config()
    concurrent_flow_iat_threshold = ConfigManager.read_concurrent_flow_iat_threshold()
    sequential_flow_iat_threshold = ConfigManager.read_sequential_flow_iat_threshold()    

    dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
    # graph_file_path = os.path.join(dataset_dir, "all_session_graph.bin")
    graph_prefix = args.graph_file_name
    graph_file_path = os.path.join(dataset_dir, f"{graph_prefix}.bin")
    
    if not os.path.exists(graph_file_path):
        logger.info(f"Error: input file not found: {graph_file_path}")
        return
    
    # 初始化分析器
    analyzer = DatasetAnalyzer(graph_file_path)

    max_nodes, max_edges = analyzer.get_max_graph_size()

    logger.info("========== GRAPH SIZE UPPER BOUND ==========")
    logger.info(f"Max nodes in a single graph : {max_nodes}")
    logger.info(f"Max edges in a single graph : {max_edges}")
    logger.info("============================================")    

    if max_nodes > 5000:
        logger.warning("Graph size exceeds safe threshold!")

    p99_nodes, p99_edges = analyzer.get_percentile_graph_size(99)

    logger.info(f"P99 nodes : {p99_nodes}")
    logger.info(f"P99 edges : {p99_edges}")

    # 打印特殊情况的报告
    if args.report:
        log_special_cases_report(analyzer, dataset_name)
    

    # JSON文件路径
    plot_data_path = ConfigManager.read_plot_data_path_config()
    json_file = os.path.join(plot_data_path, 
                            f'session_graph_size_distr_{concurrent_flow_iat_threshold}_{sequential_flow_iat_threshold}.json')
    
    # 创建输出目录
    figs_dir = os.path.join('.', 'figs', dataset_name)
    os.makedirs(figs_dir, exist_ok=True)
    
    # 根据类型生成相应的图
    plot_styles = []
    if args.plot_style == 'both':
        plot_styles = ['loglog_dot', 'normal_bar']
    else:
        plot_styles = [args.plot_style]
    
    # 处理节点分布
    if args.type in ['nodes', 'both', 'comparison']:
        node_counts, total_graphs = analyzer.load_or_compute_distribution(
            json_file, dataset_name, not args.no_cache, "nodes"
        )
        
        for plot_style in plot_styles:
            if args.type == 'nodes' or args.type == 'both':
                if plot_style == 'loglog_dot':
                    plt_node = plot_loglog_dot_handled_zeros(dataset_name, node_counts, total_graphs, "nodes")
                else:
                    plt_node = plot_normal_bar_with_zeros(dataset_name, node_counts, total_graphs, args.maxshow, "nodes")
                
                fig_path = os.path.join(figs_dir, 
                                      f'session_graph_node_distr_{plot_style}_{concurrent_flow_iat_threshold}_{sequential_flow_iat_threshold}.pdf')
                plt_node.savefig(fig_path, dpi=300, bbox_inches='tight')
                logger.info(f"Node plot ({plot_style}) saved to: {fig_path}")
                #plt_node.show()
    
    # 处理边分布
    if args.type in ['edges', 'both', 'comparison']:
        edge_counts, total_graphs = analyzer.load_or_compute_distribution(
            json_file, dataset_name, not args.no_cache, "edges"
        )
        
        for plot_style in plot_styles:
            if args.type == 'edges' or args.type == 'both':
                if plot_style == 'loglog_dot':
                    plt_edge = plot_loglog_dot_handled_zeros(dataset_name, edge_counts, total_graphs, "edges")
                else:
                    plt_edge = plot_normal_bar_with_zeros(dataset_name, edge_counts, total_graphs, args.maxshow, "edges")
                
                fig_path = os.path.join(figs_dir, 
                                      f'session_graph_edge_distr_{plot_style}_{concurrent_flow_iat_threshold}_{sequential_flow_iat_threshold}.pdf')
                plt_edge.savefig(fig_path, dpi=300, bbox_inches='tight')
                logger.info(f"Edge plot ({plot_style}) saved to: {fig_path}")
                #plt_edge.show()
    
    logger.info("All plots generated successfully!")


if __name__ == "__main__":
    main()