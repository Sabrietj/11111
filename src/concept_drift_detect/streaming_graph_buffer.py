import torch
from torch_geometric.data import Data
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class StreamingSessionGraphBuffer:
    """
    流式会话图缓存器：
    在线接收单条流的特征，并100%严格对齐离线离线的标签聚合(Dominant)逻辑。
    """

    def __init__(self, concurrent_threshold=0.1, sequential_threshold=1.0, dominant_ratio_threshold=0.8):
        self.concurrent_threshold = concurrent_threshold
        self.sequential_threshold = sequential_threshold
        self.dominant_ratio_threshold = dominant_ratio_threshold
        self.active_sessions = {}

    def add_flow_and_check_completion(self, session_id, timestamp, x_i, y_bin, y_multi):
        completed_graph = None

        # 统一转为 python 标量以便于存储和统计
        cur_bin = y_bin.item() if isinstance(y_bin, torch.Tensor) else y_bin
        cur_multi = y_multi.item() if isinstance(y_multi, torch.Tensor) else y_multi

        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                'features': [x_i.detach().cpu()],
                'timestamps': [timestamp],
                'y_bin_list': [cur_bin],
                'y_multi_list': [cur_multi]
            }
        else:
            sess = self.active_sessions[session_id]
            last_time = sess['timestamps'][-1]

            if timestamp - last_time > self.sequential_threshold:
                # 吐出旧的图
                completed_graph = self._build_graph(sess)
                # 重新初始化该 Session
                self.active_sessions[session_id] = {
                    'features': [x_i.detach().cpu()],
                    'timestamps': [timestamp],
                    'y_bin_list': [cur_bin],
                    'y_multi_list': [cur_multi]
                }
            else:
                # 未超时，继续追加特征和标签历史
                sess['features'].append(x_i.detach().cpu())
                sess['timestamps'].append(timestamp)
                sess['y_bin_list'].append(cur_bin)
                sess['y_multi_list'].append(cur_multi)

        return completed_graph

    def _build_graph(self, sess_data):
        features = torch.cat(sess_data['features'], dim=0)  # [num_nodes, 768/128]
        num_nodes = features.size(0)

        # 构建边
        if num_nodes == 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            src = torch.arange(0, num_nodes - 1, dtype=torch.long)
            dst = torch.arange(1, num_nodes, dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)

            self_loops = torch.arange(0, num_nodes, dtype=torch.long)
            self_loops = torch.stack([self_loops, self_loops], dim=0)
            edge_index = torch.cat([edge_index, self_loops], dim=1)

        # ---------------------------------------------------------
        # 🎯 核心修复：100% 对齐 SessionParser 的聚合逻辑
        # ---------------------------------------------------------
        # 1. 二分类：一票否决
        final_y_bin = 1.0 if any(b == 1.0 for b in sess_data['y_bin_list']) else 0.0

        # 2. 多分类：Dominant Ratio 逻辑
        malicious_labels = [m for m in sess_data['y_multi_list'] if m > 0]

        if len(malicious_labels) == 0:
            final_y_multi = 0  # 纯净 Benign
        elif len(set(malicious_labels)) == 1:
            final_y_multi = malicious_labels[0]  # 只有一种攻击
        else:
            # 多种攻击混合，计算 dominant
            counter = Counter(malicious_labels)
            total_attack = len(malicious_labels)
            dominant_label, dominant_count = counter.most_common(1)[0]
            dominant_ratio = dominant_count / total_attack

            if dominant_ratio >= self.dominant_ratio_threshold:
                final_y_multi = dominant_label
            else:
                final_y_multi = -1  # Mixed，测试阶段计算指标时会自动忽略它

        y_bin_tensor = torch.tensor([final_y_bin], dtype=torch.float32)
        y_multi_tensor = torch.tensor([final_y_multi], dtype=torch.long)

        return Data(x=features, edge_index=edge_index, y_bin=y_bin_tensor, y_multi=y_multi_tensor)

    def force_flush_all(self):
        graphs = []
        for sess_id, sess_data in self.active_sessions.items():
            graphs.append(self._build_graph(sess_data))
        self.active_sessions.clear()
        return graphs