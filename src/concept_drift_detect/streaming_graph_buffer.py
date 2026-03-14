import torch
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)


class StreamingSessionGraphBuffer:
    """
    流式会话图缓存器：
    在线接收单条流的特征，根据时间戳和 Session ID 进行缓冲。
    当检测到某个 Session 已经结束（超时未活动），则将其构建为 PyG Data 图对象并吐出，供 GraphMAE 预测。
    """

    def __init__(self, concurrent_threshold=0.1, sequential_threshold=1.0):
        self.concurrent_threshold = concurrent_threshold
        self.sequential_threshold = sequential_threshold
        self.active_sessions = {}

    def add_flow_and_check_completion(self, session_id, timestamp, x_i, y_bin, y_multi):
        """
        向缓存池添加一条流，并检查是否有过期的会话可以生成图
        """
        completed_graph = None

        # 1. 如果是新的 Session
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                'features': [x_i.detach().cpu()],
                'timestamps': [timestamp],
                'y_bin': y_bin.detach().cpu() if isinstance(y_bin, torch.Tensor) else torch.tensor([y_bin]),
                'y_multi': y_multi.detach().cpu() if isinstance(y_multi, torch.Tensor) else torch.tensor([y_multi])
            }
        else:
            # 2. 如果是已存在的 Session
            sess = self.active_sessions[session_id]
            last_time = sess['timestamps'][-1]

            # 检查是否超时（Sequential 间隔过大，认为上一个 Session 已经结束）
            if timestamp - last_time > self.sequential_threshold:
                # 吐出旧的图
                completed_graph = self._build_graph(sess)
                # 重新初始化该 Session
                self.active_sessions[session_id] = {
                    'features': [x_i.detach().cpu()],
                    'timestamps': [timestamp],
                    'y_bin': y_bin.detach().cpu() if isinstance(y_bin, torch.Tensor) else torch.tensor([y_bin]),
                    'y_multi': y_multi.detach().cpu() if isinstance(y_multi, torch.Tensor) else torch.tensor([y_multi])
                }
            else:
                # 未超时，继续追加特征
                sess['features'].append(x_i.detach().cpu())
                sess['timestamps'].append(timestamp)

        return completed_graph

    def _build_graph(self, sess_data):
        """将收集到的流特征构建为全连接或时序图"""
        features = torch.cat(sess_data['features'], dim=0)  # [num_nodes, 768]
        num_nodes = features.size(0)

        # 构建边：这里以最简单的全连接图或顺序时序图为例
        # 如果是时序图，只连接相邻的流
        if num_nodes == 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            # 构建一条线性的时序边 (0->1, 1->2 ...)
            src = torch.arange(0, num_nodes - 1, dtype=torch.long)
            dst = torch.arange(1, num_nodes, dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)

            # 为了 GIN 网络聚合，加上自环 (Self-loops)
            self_loops = torch.arange(0, num_nodes, dtype=torch.long)
            self_loops = torch.stack([self_loops, self_loops], dim=0)
            edge_index = torch.cat([edge_index, self_loops], dim=1)

        y_bin = sess_data['y_bin']
        y_multi = sess_data['y_multi']

        # 返回 PyG 的 Data 对象，完美适配 GraphMAE 的输入
        return Data(x=features, edge_index=edge_index, y_bin=y_bin, y_multi=y_multi)

    def force_flush_all(self):
        """在测试集遍历结束时，强制清空池子并吐出所有剩余的图"""
        graphs = []
        for sess_id, sess_data in self.active_sessions.items():
            graphs.append(self._build_graph(sess_data))
        self.active_sessions.clear()
        return graphs