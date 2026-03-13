import pickle
import dgl
from pathlib import Path
import numpy as np
import os, sys
import argparse
# 添加../utils目录到Python搜索路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
import config_manager as ConfigManager
import logging
from logging_config import setup_preset_logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.DEBUG)    

import networkx as nx
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免多线程问题
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from matplotlib import font_manager
import platform
from tqdm import tqdm

# ====== DGL 适配：把 DGLGraph 转成本模块所需的 nodes/edges ======
from typing import Iterable

use_burst_id_flag = True

# 设置Matplotlib日志级别为WARNING，忽略DEBUG信息
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

def format_floats(values, precision=6):
    """格式化浮点数列表"""
    return [f"{v:.{precision}f}" if isinstance(v, float) else str(v) for v in values]


def identify_bursts(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], delta: float = 1.0) -> List[List[Dict[str, Any]]]:
    """
    识别burst聚类 - 添加详细调试信息
    """
    if not nodes:
        return []
    
    # 检查节点是否包含有效的burst_id字段
    has_valid_burst_id = any('burst_id' in node and node['burst_id'] >= 0 for node in nodes)
    
    # 记录基本信息
    logger.debug(f"identify_bursts: 节点数={len(nodes)}, 有效burst_id={has_valid_burst_id}, delta={delta}")

    bursts_by_id = _identify_bursts_by_burst_id(nodes)
    bursts_by_time = _identify_bursts_by_time(nodes, delta)
    
    # 详细比较两种方法的结果
    if len(bursts_by_id) != len(bursts_by_time):
        logger.info(f"[WARN] 分组结果不一致: burst_id={len(bursts_by_id)} vs 时间聚类={len(bursts_by_time)}")
        
        # 详细分析差异
        _compare_burst_grouping(nodes, bursts_by_id, bursts_by_time, delta)

    if use_burst_id_flag and has_valid_burst_id:        
        return bursts_by_id
    else:
        return bursts_by_time

def _compare_burst_grouping(nodes, bursts_by_id, bursts_by_time, delta):
    """详细比较两种分组方法的差异"""
    # 按时间戳排序所有节点
    sorted_nodes = sorted(nodes, key=lambda x: x.get('ts', 0))
    
    # 构建burst_id到节点列表的映射
    id_burst_map = {}
    for burst in bursts_by_id:
        for node in burst:
            burst_id = node.get('burst_id', -1)
            if burst_id not in id_burst_map:
                id_burst_map[burst_id] = []
            id_burst_map[burst_id].append(node)
    
    # 构建时间聚类到节点列表的映射（按时间顺序）
    time_burst_map = {}
    for i, burst in enumerate(bursts_by_time):
        time_burst_map[i] = burst
    
    # 分析每个burst_id分组
    logger.debug("=== burst_id分组详情 ===")
    for burst_id in sorted(id_burst_map.keys()):
        burst_nodes = id_burst_map[burst_id]
        burst_nodes_sorted = sorted(burst_nodes, key=lambda x: x.get('ts', 0))
        if burst_nodes_sorted:
            logger.debug(f"burst_id={burst_id}: {len(burst_nodes)}个节点, "
                     f"时间范围: {burst_nodes_sorted[0].get('ts', 0):.6f} - {burst_nodes_sorted[-1].get('ts', 0):.6f}")
    
    # 分析时间聚类分组
    logger.debug("=== 时间聚类分组详情 ===")
    for i, burst in enumerate(bursts_by_time):
        burst_sorted = sorted(burst, key=lambda x: x.get('ts', 0))
        if burst_sorted:
            logger.debug(f"时间聚类burst_{i}: {len(burst)}个节点, "
                     f"时间范围: {burst_sorted[0].get('ts', 0):.6f} - {burst_sorted[-1].get('ts', 0):.6f}")
    
    # 找出具体的不一致点
    logger.debug("=== 不一致分析 ===")
    
    # 检查每个burst_id分组是否被正确拆分
    for burst_id in sorted(id_burst_map.keys()):
        burst_nodes = id_burst_map[burst_id]
        if len(burst_nodes) > 1:  # 只检查有多个节点的burst
            # 检查这个burst在时间聚类中是否被拆分
            burst_sorted = sorted(burst_nodes, key=lambda x: x.get('ts', 0))
            
            # 计算这个burst内相邻节点的时间差
            time_diffs = []
            for j in range(1, len(burst_sorted)):
                time_diff = abs(burst_sorted[j].get('ts', 0) - burst_sorted[j-1].get('ts', 0))
                time_diffs.append(time_diff)
            
            # 找出超过阈值的时间差
            large_gaps = [(j, diff) for j, diff in enumerate(time_diffs) if diff > delta]
            
            if large_gaps:
                logger.error(f"burst_id={burst_id} 内发现超过阈值的时间间隔:")
                for gap_idx, gap_diff in large_gaps:
                    logger.debug(f"  位置{gap_idx}: 时间差={gap_diff:.6f}s > delta={delta}s")
                    logger.debug(f"  节点{burst_sorted[gap_idx].get('node_id')} 和 节点{burst_sorted[gap_idx+1].get('node_id')}")

def _identify_bursts_by_burst_id(nodes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    使用burst_id字段进行burst分组
    """
    # 按burst_id分组
    bursts_dict = {}
    
    for node in nodes:
        burst_id = node.get('burst_id', -1)
        if burst_id not in bursts_dict:
            bursts_dict[burst_id] = []
        bursts_dict[burst_id].append(node)
    
    # 按burst_id排序
    bursts = []
    for burst_id in sorted(bursts_dict.keys()):
        bursts.append(bursts_dict[burst_id])
    
    # logger.info(f"使用burst_id分组，识别到 {len(bursts)} 个burst")    
    # 输出每个burst的详细信息
    for i, burst in enumerate(bursts):
        burst_id = burst[0].get('burst_id', 'unknown') if burst else 'unknown'
        logger.info(f"  Burst {i}: {len(burst)} 个节点, burst_id={burst_id}")
    
    return bursts


def _identify_bursts_by_time(nodes: List[Dict[str, Any]], delta: float = 1.0) -> List[List[Dict[str, Any]]]:
    """
    使用时间聚类进行burst分组。与 _perform_burst_clustering 保持一致的逻辑。
    """
    if not nodes:
        return []
    
    # 按时间戳排序节点
    sorted_nodes = sorted(nodes, key=lambda x: x.get('ts', 0))   

    # 添加调试信息
    # if len(sorted_nodes) > 0:
    #     logger.debug(f"时间聚类: 时间戳范围 {sorted_nodes[0].get('ts', 0):.6f} - {sorted_nodes[-1].get('ts', 0):.6f}")

    # 创建突发聚类
    bursts = []
    current_burst = [sorted_nodes[0]]  # 直接处理第一个节点
    
    for idx in range(1, len(sorted_nodes)):  # 从第二个节点开始
        node = sorted_nodes[idx]
        time_diff = abs(node.get('ts', 0) - current_burst[-1].get('ts', 0))
        
        if time_diff <= delta:
            current_burst.append(node)
        else:
            bursts.append(current_burst)
            current_burst = [node]
    
    # 处理最后一个burst
    if current_burst:
        bursts.append(current_burst)
    
    return bursts


def create_burst_layout(bursts: List[List[Dict[str, Any]]], 
                       canvas_width: float = 10.0, 
                       canvas_height: float = 12.0) -> Dict[str, Tuple[float, float]]:
    """
    为burst创建优化的布局
    - 每个burst在一个纵列中
    - burst之间横向排列
    
    Args:
        bursts: burst聚类列表
        canvas_width: 画布宽度
        canvas_height: 画布高度
        
    Returns:
        节点位置字典
    """
    pos = {}
    
    if not bursts:
        return pos
    
    # 计算burst间的横向间距
    num_bursts = len(bursts)
    if num_bursts == 1:
        x_positions = [canvas_width / 2]
    else:
        x_spacing = canvas_width / (num_bursts + 1)
        x_positions = [x_spacing * (i + 1) for i in range(num_bursts)]
    
    # 为每个burst分配位置
    for burst_idx, burst in enumerate(bursts):
        x_pos = x_positions[burst_idx]
        
        # 计算该burst内节点的纵向间距
        num_nodes = len(burst)
        if num_nodes == 1:
            y_positions = [canvas_height / 2]
        else:
            y_spacing = canvas_height / (num_nodes + 1)
            y_positions = [canvas_height - y_spacing * (i + 1) for i in range(num_nodes)]
        
        # 按时间顺序排列节点（从上到下）
        sorted_burst = sorted(burst, key=lambda x: x.get('ts', 0))
        
        for node_idx, node in enumerate(sorted_burst):
            node_id = node['node_id']
            y_pos = y_positions[node_idx]
            pos[node_id] = (x_pos, y_pos)
    
    return pos


def get_burst_colors(bursts: List[List[Dict[str, Any]]]) -> Dict[str, str]:
    """
    为不同burst分配不同颜色
    
    Args:
        bursts: burst聚类列表
        
    Returns:
        节点ID到颜色的映射
    """
    # 定义颜色调色板
    colors = [
        '#FF6B6B',  # 红色
        '#4ECDC4',  # 青色
        '#45B7D1',  # 蓝色
        '#96CEB4',  # 绿色
        '#FFEAA7',  # 黄色
        '#DDA0DD',  # 紫色
        '#98D8C8',  # 薄荷绿
        '#F7DC6F',  # 金黄色
        '#BB8FCE',  # 淡紫色
        '#85C1E9',  # 天蓝色
    ]
    
    node_colors = {}
    
    for burst_idx, burst in enumerate(bursts):
        color = colors[burst_idx % len(colors)]
        for node in burst:
            node_colors[node['node_id']] = color
    
    return node_colors


def visualize_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], 
                   output_path: Optional[str] = None, family_label: Optional[str] = None, 
                   burst_num_see: int = 50, delta: float = 1.0):
    """
    优化的burst图可视化
    1. burst节点统一在一个纵列，burst之间按照横向排列
    2. 每个burst节点不同颜色标注
    3. 节点序号展示在节点上
    4. 边上增加边起始节点和终节点的ts差
    
    Args:
        nodes: 节点列表
        edges: 边列表
        output_path: 输出路径
        family_label: 家族标签
        burst_num_see: 显示的burst节点数量限制
        delta: burst时间窗口阈值
    """
    if not nodes:
        logger.warning("No nodes to visualize")
        return
    
    logger.info(f"burst_num_see = {burst_num_see}")
    try:
        # 按时间戳排序节点
        nodes = sorted(nodes, key=lambda x: x.get('ts', 0))
        logger.info(f"nodes时间戳: {format_floats([node.get('ts', 0) for node in nodes[:10]])}")
        # 识别全部bursts
        all_bursts = identify_bursts(nodes, edges, delta)
        # 限制可视化的burst数量
        vis_bursts = all_bursts[:burst_num_see] if len(all_bursts) > burst_num_see else all_bursts
        # 展开为可视节点
        vis_nodes = [node for burst in vis_bursts for node in burst]
        vis_node_ids = {node['node_id'] for node in vis_nodes}
        
        # 过滤相关的边
        vis_edges = [edge for edge in edges 
                    if edge.get('source') in vis_node_ids and edge.get('target') in vis_node_ids]
        
        # 识别burst聚类
        bursts = identify_bursts(vis_nodes, vis_edges, delta)
        # logger.info(f"number of bursts = {len(bursts)}!!")        
        # 创建NetworkX图
        G = nx.DiGraph()
        
        # 添加节点
        for node in vis_nodes:
            node_id = node['node_id']
            G.add_node(node_id, **node)
        
        # 添加边
        for edge in vis_edges:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                G.add_edge(source, target,** edge)
        
        # 创建图形
        # fig, ax = plt.subplots(figsize=(max(12, len(bursts) * 3), 16))
        fig, ax = plt.subplots(figsize=(max(16, len(bursts) * 4), 20))  # 进一步增加尺寸
        plt.clf()
        
        # 创建burst布局
        pos = create_burst_layout(bursts, canvas_width=max(10, len(bursts) * 2), canvas_height=12)
        
        # 获取burst颜色
        node_colors_map = get_burst_colors(bursts)
        
        # 绘制节点
        if len(list(G.nodes())) > 0:
            node_colors = []
            node_sizes = []
            
            for node_id in G.nodes():
                # 使用burst颜色
                color = node_colors_map.get(node_id, '#9E9E9E')
                node_colors.append(color)
                node_sizes.append(1200)  # 较大的节点以便显示编号
            
            nx.draw_networkx_nodes(G, pos, 
                                 node_color=node_colors, 
                                 node_size=node_sizes, 
                                 alpha=0.8,
                                 edgecolors='black',
                                 linewidths=2)
        
        # 绘制边
        if len(list(G.edges())) > 0:
            # 区分不同类型的边
            internal_edges = []
            external_start_edges = []
            external_end_edges = []
            external_single_edges = []
            external_old_edges = []
            
            for edge in vis_edges:
                source = edge.get('source')
                target = edge.get('target')
                edge_type = edge.get('edge_type', 'unknown')
                
                if edge_type == 'burst_internal':
                    internal_edges.append((source, target))
                elif edge_type == 'burst_external_start':
                    external_start_edges.append((source, target))
                elif edge_type == 'burst_external_end':
                    external_end_edges.append((source, target))
                elif edge_type == 'burst_external_single':
                    external_single_edges.append((source, target))
                elif edge_type == 'burst_external':
                    external_old_edges.append((source, target))
                else:
                    external_old_edges.append((source, target))
            
            # 统一burst内部边颜色为蓝色
            if internal_edges:
                nx.draw_networkx_edges(G, pos, 
                                     edgelist=internal_edges,
                                     edge_color='blue',
                                     style='solid',
                                     width=2.5,
                                     alpha=0.8, 
                                     arrows=True,
                                     arrowsize=25,
                                     arrowstyle='->')
            
            # 统一burst之间的边颜色为红色
            all_external_edges = external_start_edges + external_end_edges + external_single_edges + external_old_edges
            if all_external_edges:
                nx.draw_networkx_edges(G, pos, 
                                     edgelist=all_external_edges,
                                     edge_color='red',
                                     style='solid',
                                     width=3.0,
                                     alpha=0.9, 
                                     arrows=True,
                                     arrowsize=30,
                                     arrowstyle='->')
        
        # 添加节点标签（节点序号）
        node_labels = {}
        for node_id in G.nodes():
            # 提取节点编号（如 node_0 -> 0）
            if node_id.startswith('node_'):
                node_num = node_id.split('_')[1]
                node_labels[node_id] = node_num
            else:
                node_labels[node_id] = node_id
        
        nx.draw_networkx_labels(G, pos, node_labels, 
                              font_size=14, font_weight='bold', 
                              font_color='black')
        
        # 添加边标签（时间差）
        # 根据节点数量调整边标签显示策略
        max_edges_for_labels = min(100, len(vis_nodes) * 3)  # 动态调整边标签显示限制
        
        if len(vis_edges) <= max_edges_for_labels:
            edge_labels = {}
            for edge in vis_edges:
                source = edge.get('source')
                target = edge.get('target')
                time_diff = edge.get('time_diff', 0)
                
                if source and target and time_diff is not None:
                    # 格式化时间差
                    if time_diff < 0.001:
                        time_str = f"{time_diff*1000:.2f}ms"
                    elif time_diff < 1.0:
                        time_str = f"{time_diff:.3f}s"
                    else:
                        time_str = f"{time_diff:.2f}s"
                    edge_labels[(source, target)] = time_str
            
            if edge_labels:
                # 根据边数量调整字体大小
                font_size = max(8, min(12, 300 // len(edge_labels)))
                nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                           font_size=font_size, font_color='black',
                                           bbox=dict(boxstyle='round,pad=0.2', 
                                                   facecolor='white', alpha=0.8))
        else:
            # 如果边太多，只显示burst间的边标签
            edge_labels = {}
            for edge in vis_edges:
                edge_type = edge.get('edge_type', '')
                if edge_type in ['burst_external_start', 'burst_external_end', 'burst_external_single', 'burst_external']:
                    source = edge.get('source')
                    target = edge.get('target')
                    time_diff = edge.get('time_diff', 0)
                    
                    if source and target and time_diff is not None:
                        # 格式化时间差
                        if time_diff < 0.001:
                            time_str = f"{time_diff*1000:.2f}ms"
                        elif time_diff < 1.0:
                            time_str = f"{time_diff:.3f}s"
                        else:
                            time_str = f"{time_diff:.2f}s"
                        edge_labels[(source, target)] = time_str
            
            if edge_labels:
                nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                           font_size=10, font_color='black',
                                           bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='yellow', alpha=0.9))
        
        # 添加burst标签
        for burst_idx, burst in enumerate(bursts):
            if burst:
                # 计算burst的中心位置
                burst_node_ids = [node['node_id'] for node in burst]
                burst_positions = [pos[node_id] for node_id in burst_node_ids if node_id in pos]
                
                if burst_positions:
                    center_x = np.mean([p[0] for p in burst_positions])
                    max_y = max([p[1] for p in burst_positions])
                    
                    # 在burst上方添加标签
                    plt.text(center_x, max_y + 0.8, f'Burst {burst_idx + 1}', 
                           ha='center', va='bottom', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
        
        # 设置标题
        title = f"Optimized Burst Flow Graph"
        if family_label:
            title += f" - {family_label}"
        title += f"\n({len(vis_nodes)} nodes, {len(vis_edges)} edges, {len(bursts)} bursts)"
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 设置坐标轴
        plt.axis('off')
        
        # 调整布局
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        
        # 保存图形
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                        
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in graph visualization: {str(e)}")
        plt.close()


def save_graph_data(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], 
                   output_dir: str, family_label: str):
    """
    保存图数据到JSON文件
    
    Args:
        nodes: 节点列表
        edges: 边列表
        output_dir: 输出目录
        family_label: 家族标签
    """
    import json
    
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存节点数据
        nodes_file = os.path.join(output_dir, 'nodes.json')
        with open(nodes_file, 'w', encoding='utf-8') as f:
            json.dump(nodes, f, indent=2, ensure_ascii=False)
        
        # 保存边数据
        edges_file = os.path.join(output_dir, 'edges.json')
        with open(edges_file, 'w', encoding='utf-8') as f:
            json.dump(edges, f, indent=2, ensure_ascii=False)
        
        # 保存图的可视化
        vis_file = os.path.join(output_dir, f'{family_label}.pdf')
        visualize_graph(nodes, edges, vis_file, family_label, delta = ConfigManager.read_concurrent_flow_iat_threshold())
        

        
    except Exception as e:
        logger.error(f"Error saving graph data: {str(e)}")

def visualize_dgl_graph(
    g,
    output_path: Optional[str] = None,
    family_label: Optional[str] = None,
    burst_num_see: int = 50,
    delta: float = 1.0
):
    """
    直接可视化一张 DGLGraph：自动做转换 + 调用 visualize_graph
    """
    nodes, edges = dgl_graph_to_nodes_edges(g)

    # 如果节点里已经有 burst_id，则改用“按 burst_id 分组 + ts 竖排”的策略：
    # 你的 visualize_graph 默认用 identify_bursts(ts,delta) 来分组，
    # 这里不强改逻辑，只给 nodes/edges，保留兼容性。
    # 若你希望“优先按 burst_id 分组”，可把 identify_bursts 替换成按 burst_id 的分组函数（见下方注释）。

    visualize_graph(
        nodes=nodes,
        edges=edges,
        output_path=output_path,
        family_label=family_label,
        burst_num_see=burst_num_see,
        delta=delta
    )

def is_malicious_label_string(label_str):
    label_norm = str(label_str).strip().lower()
    if label_norm == "benign" or label_norm.startswith("benign"):
        return False
    else:
        return True

def visualize_dgl_bin(
    bin_file_path: str,
    out_dir: str,
    min_graph_index: int = 0,  # 新增：最小图ID
    max_graph_index: int = 100,  # 修改：最大图ID（原来是max_graphs）
    burst_num_see: int = 50,
    delta: float = 1.0,
    family_prefix: str = "graph",
    only_draw_non_benign: bool = False,   # ⭐ 新增
    skip_labels = None,
):
    """
    从 DGL 的 .bin 文件批量读取并可视化指定范围内的图，同时打印每个图的label_name、label_id和split信息
    
    Args:
        bin_file_path: .bin文件路径
        out_dir: 输出目录
        min_graph_index: 最小图ID（包含）
        max_graph_index: 最大图ID（包含）
        burst_num_see: 显示burst数量限制
        delta: burst时间窗口阈值
        family_prefix: 图名前缀
    """

    # -- 生成并读取配套的label_info.pkl文件 --
    # 从.bin路径生成xxx_info.pkl路径（例如：data.bin → data_info.pkl）
    base, _ = os.path.splitext(bin_file_path)
    info_file_path = base + "_info.pkl"  # 与save_results中保存的info文件路径完全对应
    
    # 检查info文件是否存在
    if not os.path.exists(info_file_path):
        raise FileNotFoundError(f"配套的标签文件不存在：{info_file_path}\n请确保与.bin文件放在同一目录，且前缀一致")
    
    # 读取label_name、label_id和split信息（使用pickle加载，与save_results中的save_info对应）
    try:
        with open(info_file_path, "rb") as f:
            graph_infos = pickle.load(f)  # 加载info字典
        # 提取标签列表（确保键名与save_results中一致）
        label_names = graph_infos.get("label_name", [])
        label_ids = graph_infos.get("label_id", [])
        splits = graph_infos.get("split", [])  # 新增：读取split信息
    except Exception as e:
        raise RuntimeError(f"读取标签文件失败：{str(e)}") from e

    # -- 校验标签数量与图数量一致 --
    # 先加载图，获取图总数
    file_size = os.path.getsize(bin_file_path)
    logger.info(f"Loading DGL graphs from {bin_file_path} ({file_size/1024/1024:.2f} MB) ...")

    # 按字节流读取文件，显示真实进度
    with open(bin_file_path, "rb") as f, tqdm(
        total=file_size, unit='B', unit_scale=True, desc="Reading file"
    ) as pbar:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            pbar.update(len(chunk))
    
    logger.info(f"Caling dgl.load_graphs for {bin_file_path} ...")
    graphs, _ = dgl.load_graphs(bin_file_path)
    total_graphs = len(graphs)
    logger.info(f"Loaded {total_graphs} graphs!")
    
    # 检查标签数量是否匹配图数量
    if len(label_names) != total_graphs or len(label_ids) != total_graphs or len(splits) != total_graphs:
        raise ValueError(
            f"标签数量与图数量不匹配！\n"
            f"图总数：{total_graphs}, label_name数量：{len(label_names)}, label_id数量：{len(label_ids)}, split数量：{len(splits)}\n"
            "请确认xxx_info.pkl与.bin文件是同一批数据生成的"
        )

    # 获取 bin_file_path 里面 `processed_data`目录的下级目录名
    path_parts = Path(bin_file_path).parts
    try:
        # 找到 processed_data 的索引位置
        processed_index = path_parts.index("processed_data")
        # 获取 processed_data 的下一个目录名
        subdirectory = path_parts[processed_index + 1]
        
        # 拼接路径
        out_dir_path = Path(out_dir) / subdirectory
        logger.info(f"结果路径: {out_dir_path}")
        
    except (ValueError, IndexError):
        # 如果找不到 processed_data 或没有下级目录
        out_dir_path = Path(out_dir)
        logger.info(f"无法找到 processed_data 的下级目录，使用: {out_dir_path}")

    # 创建目录（如果不存在）
    out_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"目录已创建或已存在: {out_dir_path}")

    # 删除该目录下的所有 .pdf 文件（可选，根据需求决定是否保留）
    pdf_files = list(out_dir_path.glob("*.pdf"))
    if pdf_files:
        logger.info(f"找到 {len(pdf_files)} 个 PDF 文件，正在删除...")
        for pdf_file in pdf_files:
            try:
                pdf_file.unlink()  # 删除文件
                logger.info(f"已删除: {pdf_file}")
            except Exception as e:
                logger.info(f"删除文件 {pdf_file} 时出错: {e}")
    else:
        logger.info("没有找到 PDF 文件")
            
    graphs, _ = dgl.load_graphs(bin_file_path)
    total = len(graphs)
    logger.info(f"Loaded {total} graphs from {bin_file_path}")

    # 验证索引范围
    if min_graph_index < 0:
        min_graph_index = 0
        logger.info(f"警告: min_graph_index 小于0，已调整为0")
    
    if max_graph_index >= total:
        max_graph_index = total - 1
        logger.info(f"警告: max_graph_index 超出范围，已调整为 {total - 1}")
    
    if min_graph_index > max_graph_index:
        min_graph_index, max_graph_index = max_graph_index, min_graph_index
        logger.info(f"警告: min_graph_index > max_graph_index，已交换两者值")

    logger.info(f"可视化图ID范围: [{min_graph_index}, {max_graph_index}]")
    logger.info(f"总共可视图数: {max_graph_index - min_graph_index + 1}")

    # 统计split分布
    split_counts = {'train': 0, 'valid': 0, 'test': 0, 'other': 0}
    
    # 统计信息
    stats = []
    
    for i in range(min_graph_index, max_graph_index + 1):
        if i >= len(graphs):
            break
        
        logger.info(f"processing the graph with index {i} ...")
        g = graphs[i].cpu()

        # 获取当前图的label_name、label_id和split信息
        current_label_name = label_names[i]  # 按图索引匹配标签
        current_label_id = label_ids[i]
        current_split = splits[i] if i < len(splits) else 'train'  # 新增：获取split信息

        if not is_malicious_label_string(current_label_name):
            current_label_str = "benign"
        else:
            current_label_str = current_label_name.strip()
            
        # ====== 是否只绘制非 benign 图 ======
        if only_draw_non_benign:
            if current_label_str == "benign":
                logger.debug(f"Skip benign graph {i}: label_name={current_label_name}")
                continue

        if skip_labels is not None:
            if current_label_str.lower() in skip_labels:
                logger.debug(f"Skip graph {i}: label_name={current_label_name} (matched skip_labels)")
                continue

        # 统计split分布
        split_str = str(current_split).lower().strip()
        if split_str == 'train':
            split_counts['train'] += 1
        elif split_str in ['validate', 'valid', 'validation', 'val', 'dev']:
            split_counts['valid'] += 1
        elif split_str == 'test':
            split_counts['test'] += 1
        else:
            split_counts['other'] += 1

        # 获取图的统计信息
        num_nodes = g.num_nodes()
        
        # 计算burst数量
        if 'burst_id' in g.ndata:
            burst_ids = g.ndata['burst_id']
            if hasattr(burst_ids, 'unique'):
                unique_bursts_by_burst_id = burst_ids.unique().numel()
            else:
                unique_bursts_by_burst_id = len(set(burst_ids.tolist()))

        nodes, edges = dgl_graph_to_nodes_edges(g)
        bursts_by_delta = identify_bursts(nodes, edges, delta)
        unique_bursts_by_delta = len(bursts_by_delta)
        
        # 记录统计信息
        if use_burst_id_flag:
            stats.append({
                'graph_id': i,
                'num_nodes': num_nodes,
                'num_bursts': unique_bursts_by_burst_id,
                'split': current_split  # 新增：记录split信息
            })
        else:
            stats.append({
                'graph_id': i,
                'num_nodes': num_nodes,
                'num_bursts': unique_bursts_by_delta,
                'split': current_split  # 新增：记录split信息
            })
        
        # 输出当前图的信息（新增split打印）
        logger.info(
            f"图 {i:4d}: "
            f"label_name={current_label_name:10s}, label_id={current_label_id:3d}, split={current_split:8s}, "  # 新增split打印
            f"节点数={num_nodes:3d}, "
            f"burst数目_by_burst_id={unique_bursts_by_burst_id:2d}, "
            f"burst数目_by_delta={unique_bursts_by_delta:2d}"
        )
        
        out_path = os.path.join(
            out_dir_path,
            f"{family_prefix}_{i:04d}_{current_label_str}"
            f"_v{num_nodes}_b{unique_bursts_by_delta}.pdf"
        )
        
        # 正常处理所有图
        visualize_dgl_graph(
            g,
            output_path=out_path,
            family_label=current_label_str,
            burst_num_see=burst_num_see,
            delta=delta
        )
        logger.info(f"Saved {out_path}")

    # 输出split分布统计
    logger.info(f"\n=== Split分布统计 (图ID范围: [{min_graph_index}, {max_graph_index}]) ===")
    total_processed = len(stats)
    if total_processed > 0:
        logger.info(f"训练集: {split_counts['train']} 图 ({split_counts['train']/total_processed:.1%})")
        logger.info(f"验证集: {split_counts['valid']} 图 ({split_counts['valid']/total_processed:.1%})")
        logger.info(f"测试集: {split_counts['test']} 图 ({split_counts['test']/total_processed:.1%})")
        if split_counts['other'] > 0:
            logger.info(f"其他split: {split_counts['other']} 图 ({split_counts['other']/total_processed:.1%})")

    # 输出总体统计信息
    logger.info(f"\n=== 总体统计 (图ID范围: [{min_graph_index}, {max_graph_index}]) ===")
    logger.info(f"总共处理了 {len(stats)} 个图")
    
    if stats:
        total_nodes = sum(s['num_nodes'] for s in stats)
        total_bursts = sum(s['num_bursts'] for s in stats)
        avg_nodes = total_nodes / len(stats)
        avg_bursts = total_bursts / len(stats)
        
        logger.info(f"总节点数: {total_nodes}")
        logger.info(f"总burst数: {total_bursts}")
        logger.info(f"平均每个图节点数: {avg_nodes:.2f}")
        logger.info(f"平均每个图burst数: {avg_bursts:.2f}")
        
        # 按split分组统计
        logger.info(f"\n=== 按Split分组统计 (图ID范围: [{min_graph_index}, {max_graph_index}]) ===")
        split_groups = {}
        for stat in stats:
            split = str(stat['split']).lower().strip()
            if split not in split_groups:
                split_groups[split] = []
            split_groups[split].append(stat)
        
        for split, group_stats in split_groups.items():
            group_nodes = sum(s['num_nodes'] for s in group_stats)
            group_bursts = sum(s['num_bursts'] for s in group_stats)
            avg_group_nodes = group_nodes / len(group_stats)
            avg_group_bursts = group_bursts / len(group_stats)
            logger.info(f"Split '{split}': {len(group_stats)} 图, 平均节点数={avg_group_nodes:.2f}, 平均burst数={avg_group_bursts:.2f}")
        
        # 输出统计摘要
        logger.info(f"\n=== 统计摘要 (图ID范围: [{min_graph_index}, {max_graph_index}]) ===")
        node_counts = [s['num_nodes'] for s in stats]
        burst_counts = [s['num_bursts'] for s in stats]
        
        logger.info(f"节点数范围: {min(node_counts)} - {max(node_counts)}")
        logger.info(f"burst数范围: {min(burst_counts)} - {max(burst_counts)}")
        logger.info(f"节点数中位数: {np.median(node_counts):.1f}")
        logger.info(f"burst数中位数: {np.median(burst_counts):.1f}")
        
        # 绘制统计分布图（可选）
        try:
            _plot_statistics(stats, out_dir_path)
        except Exception as e:
            logger.info(f"绘制统计图时出错: {e}")


def _plot_statistics(stats, output_dir):
    """绘制统计分布图 - 包含split信息"""
    import matplotlib.pyplot as plt
    
    # 节点数量分布
    node_counts = [s['num_nodes'] for s in stats]
    burst_counts = [s['num_bursts'] for s in stats]
    splits = [str(s['split']).lower().strip() for s in stats]
    
    # 为不同split分配颜色
    split_colors = {
        'train': 'blue',
        'valid': 'green', 
        'validation': 'green',
        'val': 'green',
        'dev': 'green',
        'test': 'red'
    }
    
    # 默认颜色
    default_color = 'gray'
    
    # 节点数量分布（按split着色）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 节点数量直方图 - 按split着色
    for split in set(splits):
        split_node_counts = [node_counts[i] for i in range(len(node_counts)) if splits[i] == split]
        if split_node_counts:
            color = split_colors.get(split, default_color)
            ax1.hist(split_node_counts, bins=20, alpha=0.7, color=color, edgecolor='black', label=split)
    
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Node Count Distribution by Split')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # burst数量直方图 - 按split着色
    for split in set(splits):
        split_burst_counts = [burst_counts[i] for i in range(len(burst_counts)) if splits[i] == split]
        if split_burst_counts:
            color = split_colors.get(split, default_color)
            ax2.hist(split_burst_counts, bins=20, alpha=0.7, color=color, edgecolor='black', label=split)
    
    ax2.set_xlabel('Number of Bursts')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Burst Count Distribution by Split')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistics_distribution_by_split.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 节点与burst的关系散点图 - 按split着色
    plt.figure(figsize=(10, 8))
    
    for split in set(splits):
        split_nodes = [node_counts[i] for i in range(len(node_counts)) if splits[i] == split]
        split_bursts = [burst_counts[i] for i in range(len(burst_counts)) if splits[i] == split]
        
        if split_nodes and split_bursts:
            color = split_colors.get(split, default_color)
            plt.scatter(split_nodes, split_bursts, alpha=0.6, color=color, label=split, s=60)
    
    plt.xlabel('Number of Nodes')
    plt.ylabel('Number of Bursts')
    plt.title('Nodes vs Bursts Relationship by Split')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加趋势线
    if len(node_counts) > 1:
        z = np.polyfit(node_counts, burst_counts, 1)
        p = np.poly1d(z)
        plt.plot(node_counts, p(node_counts), "k--", alpha=0.8, label=f'Overall Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nodes_vs_bursts_by_split.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Statistical distribution charts with split information saved")
    
# # 修改 dgl_graph_to_nodes_edges 函数以处理 burst_id
# def dgl_graph_to_nodes_edges(g) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
#     """
#     将 DGLGraph 转换为本模块 visualize_graph 可用的 nodes / edges 列表。
#     """
#     import torch
#     nodes: List[Dict[str, Any]] = []
#     edges: List[Dict[str, Any]] = []

#     num_nodes = g.num_nodes()
#     num_edges = g.num_edges()

#     # ---- 取节点属性 ----
#     if 'ts' in g.ndata:
#         ts_tensor = g.ndata['ts']
#     else:
#         ts_tensor = None

#     if 'burst_id' in g.ndata:
#         bid_tensor = g.ndata['burst_id']
#     else:
#         bid_tensor = None

#     # 装配节点
#     for n in range(num_nodes):
#         nd: Dict[str, Any] = {}
#         nd['node_id'] = f'node_{n}'

#         if ts_tensor is not None:
#             v = ts_tensor[n]
#             if isinstance(v, torch.Tensor):
#                 v = v.item()
#             nd['ts'] = float(v)

#         if bid_tensor is not None:
#             v = bid_tensor[n]
#             if isinstance(v, torch.Tensor):
#                 v = v.item()
#             try:
#                 nd['burst_id'] = int(v)
#             except Exception:
#                 nd['burst_id'] = -1

#         nodes.append(nd)

#     # ---- 取边属性 ----
#     src, dst = g.edges()
#     etype_tensor = g.edata.get('etype', None)

#     for i in range(num_edges):
#         u = int(src[i])
#         v = int(dst[i])
#         e: Dict[str, Any] = {'source': f'node_{u}', 'target': f'node_{v}'}

#         if etype_tensor is not None:
#             et = etype_tensor[i]
#             if hasattr(et, 'item'):
#                 et = et.item()
#             e['edge_type'] = str(et)

#         # 计算时间差
#         if ts_tensor is not None:
#             ts_u = ts_tensor[u].item() if hasattr(ts_tensor[u], 'item') else ts_tensor[u]
#             ts_v = ts_tensor[v].item() if hasattr(ts_tensor[v], 'item') else ts_tensor[v]
#             try:
#                 e['time_diff'] = float(ts_v) - float(ts_u)
#             except Exception:
#                 pass

#         edges.append(e)

#     return nodes, edges


def dgl_graph_to_nodes_edges(g) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将 DGLGraph 转换为本模块 visualize_graph 可用的 nodes / edges 列表。
    期望从 g.ndata / g.edata 中读取字段:
      - 节点: 'ts' (float), 'burst_id' (int) （可选）
      - 边:   'etype' (int 或 str) （可选，用于区分内部/外部边）
    """
    import torch
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    # ---- 取节点属性（若不存在则给默认值）----
    # ts
    if 'ts' in g.ndata:
        ts_tensor = g.ndata['ts']
    else:
        ts_tensor = None

    # burst_id
    if 'burst_id' in g.ndata:
        bid_tensor = g.ndata['burst_id']
    else:
        bid_tensor = None

    # 装配节点
    for n in range(num_nodes):
        nd: Dict[str, Any] = {}
        nd['node_id'] = f'node_{n}'

        if ts_tensor is not None:
            v = ts_tensor[n]
            if isinstance(v, torch.Tensor):
                v = v.item()
            nd['ts'] = float(v)
        else:
            # 没有 ts 就不给，identify_bursts 会用默认 0
            pass

        if bid_tensor is not None:
            v = bid_tensor[n]
            if isinstance(v, torch.Tensor):
                v = v.item()
            try:
                nd['burst_id'] = int(v)
            except Exception:
                nd['burst_id'] = -1

        nodes.append(nd)

    # ---- 取边属性 ----
    src, dst = g.edges()
    # etype
    etype_tensor = g.edata.get('etype', None)

    # 枚举边，决定 edge_type（给颜色/样式用）
    # 这里的规则：
    # - 同 burst_id 内: 'burst_internal'
    # - 非同 burst_id : 'burst_external'
    # - 如果没有 burst_id，则统一 'unknown'
    def edge_type_of(u: int, v: int) -> str:
        if bid_tensor is None:
            return 'unknown'
        bu = bid_tensor[u].item() if isinstance(bid_tensor[u], torch.Tensor) else bid_tensor[u]
        bv = bid_tensor[v].item() if isinstance(bid_tensor[v], torch.Tensor) else bid_tensor[v]
        return 'burst_internal' if bu == bv else 'burst_external'

    for i in range(num_edges):
        u = int(src[i])
        v = int(dst[i])
        e: Dict[str, Any] = {'source': f'node_{u}', 'target': f'node_{v}'}

        # 赋 edge_type（优先用 edata['etype']，否则用 burst 内/外判定）
        if etype_tensor is not None:
            et = etype_tensor[i]
            # int/str 都支持
            if hasattr(et, 'item'):
                et = et.item()
            e['edge_type'] = str(et)
        else:
            e['edge_type'] = edge_type_of(u, v)

        # 计算 time_diff（如果 ts 可用）
        if ts_tensor is not None:
            ts_u = ts_tensor[u].item() if hasattr(ts_tensor[u], 'item') else ts_tensor[u]
            ts_v = ts_tensor[v].item() if hasattr(ts_tensor[v], 'item') else ts_tensor[v]
            try:
                e['time_diff'] = float(ts_v) - float(ts_u)
            except Exception:
                pass

        edges.append(e)

    return nodes, edges


# 设置中文字体
def setup_matplotlib_font():
    """配置Matplotlib字体支持"""
    system = platform.system()
    
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    elif system == "Darwin":  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'DejaVu Sans']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False


# ===========================
# 主入口
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 DGL .bin 文件中的会话图")
    parser.add_argument("--out_dir", default="figs", help="输出目录")
    parser.add_argument("--min_idx", type=int, default=0, help="起始图索引")
    parser.add_argument("--max_idx", type=int, default=1000000, help="结束图索引")
    parser.add_argument("--burst_num_see", type=int, default=100, help="可视burst数上限")
    parser.add_argument("--only_draw_non_benign", action="store_true", help="只绘制非 benign 图（默认关闭）")
    parser.add_argument("--skip_labels", type=str, default="", 
                        help="Comma-separated labels to skip when drawing (visualization only), e.g. portscan,benign")
    parser.add_argument("--graph_file_name", type=str, default="all_session_graph", 
                        help="Graph file prefix without suffix, e.g. all_session_graph__xxx")

    args = parser.parse_args()

    plot_data_dir = ConfigManager.read_plot_data_path_config()
    # bin_path = os.path.join(plot_data_dir, "all_session_graph.bin")
    graph_prefix = args.graph_file_name

    bin_path = os.path.join(plot_data_dir, f"{graph_prefix}.bin")    
    logger.info(f"The path to the session graph binary file is {bin_path}!!")

    skip_labels = [x.strip().lower() for x in args.skip_labels.split(",") if x.strip()]

    
    # 在文件开头调用
    setup_matplotlib_font()

    # 运行可视化
    visualize_dgl_bin(
        bin_file_path=bin_path,
        out_dir=args.out_dir,
        min_graph_index=args.min_idx,
        max_graph_index=args.max_idx,
        burst_num_see=args.burst_num_see,
        delta=ConfigManager.read_concurrent_flow_iat_threshold(),
        family_prefix="session_graph",
        only_draw_non_benign=args.only_draw_non_benign,   # ⭐ 只画恶意
        skip_labels=skip_labels
    )