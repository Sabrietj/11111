import math
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
from collections import deque

try:
    from river import drift
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

# ==========================================
# 核心数学工具
# ==========================================
def log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

# ==========================================
# 检测器基类
# ==========================================
class BaseDriftDetector(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.total_samples = 0

        self.projection_matrix = None
        self.running_mean = 0.0
        self.running_var = 1.0
        self.stats_frozen = False

    def preprocess(self, features: torch.Tensor) -> float:
        device = features.device
        if features.dim() == 1:
            features = features.unsqueeze(0)

        feature_dim = features.shape[-1]

        if self.projection_matrix is None:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(self.config.get('seed', 2026))
            self.projection_matrix = torch.randn(feature_dim, 1, generator=g_cpu).to(device)
            self.projection_matrix = self.projection_matrix / torch.norm(self.projection_matrix)
        elif self.projection_matrix.device != device:
            self.projection_matrix = self.projection_matrix.to(device)

        val = torch.matmul(features, self.projection_matrix).item()

        if not self.stats_frozen:
            self.total_samples += 1
            delta = val - self.running_mean
            self.running_mean += delta / self.total_samples
            delta2 = val - self.running_mean
            self.running_var += delta * delta2

        if self.total_samples < 2:
            return 0.0

        std = math.sqrt(self.running_var / (self.total_samples - 1)) + 1e-8
        return (val - self.running_mean) / std

    def freeze_stats(self):
        self.stats_frozen = True

    def reset_stats(self):
        """
        【⭐ 核心修复】重置统计量。
        漂移发生后，旧的均值和方差已经失效，必须解冻并重新计算新分布的均值和方差。
        防止后续的新数据经过 norm.cdf() 后全部饱和为 0 或 1。
        """
        self.total_samples = 0
        self.running_mean = 0.0
        self.running_var = 1.0
        self.stats_frozen = False

    @abstractmethod
    def update(self, val: float) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

# ==========================================
# BNDM (Polya Tree) 实现
# ==========================================
class PTNode:
    def __init__(self, level, alpha):
        self.level = level
        self.alpha = alpha
        self.n_ref_L, self.n_ref_R = 0, 0
        self.n_cur_L, self.n_cur_R = 0, 0
        self.cached_log_B = 0.0
        self.left, self.right = None, None

    def compute_log_bayes_factor(self):
        n_ref = self.n_ref_L + self.n_ref_R
        n_cur = self.n_cur_L + self.n_cur_R

        if n_ref == 0 and n_cur == 0:
            self.cached_log_B = 0.0
            return 0.0

        n_tot_L = self.n_ref_L + self.n_cur_L
        n_tot_R = self.n_ref_R + self.n_cur_R

        log_ev_H0 = log_beta(self.alpha + n_tot_L, self.alpha + n_tot_R) - log_beta(self.alpha, self.alpha)
        log_ev_H1_ref = log_beta(self.alpha + self.n_ref_L, self.alpha + self.n_ref_R) - log_beta(self.alpha, self.alpha)
        log_ev_H1_cur = log_beta(self.alpha + self.n_cur_L, self.alpha + self.n_cur_R) - log_beta(self.alpha, self.alpha)

        self.cached_log_B = log_ev_H0 - (log_ev_H1_ref + log_ev_H1_cur)
        return self.cached_log_B

class BNDMDetector(BaseDriftDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_level = config.get('max_level', 6)
        self.alpha_scale = config.get('alpha_scale', 0.1)
        self.window_size = config.get('window_size', 1000)

        threshold_val = config.get('threshold', 0.01)
        if threshold_val <= 0: threshold_val = 0.01
        self.threshold = math.log(threshold_val)

        self.ref_window = deque(maxlen=self.window_size)
        self.cur_window = deque(maxlen=self.window_size)
        self.root = self._build_tree(0)

        from scipy.stats import norm
        self.norm_cdf = norm.cdf
        self.is_initialized = False

    def _build_tree(self, level):
        alpha = self.alpha_scale * ((level + 1) ** 2)
        node = PTNode(level, alpha)
        if level < self.max_level:
            node.left = self._build_tree(level + 1)
            node.right = self._build_tree(level + 1)
        return node

    def _update_tree_sliding(self, old_val, new_val, window_type):
        cdf_old = self.norm_cdf(old_val)
        cdf_new = self.norm_cdf(new_val)

        node = self.root
        low, high = 0.0, 1.0
        level = 0

        while level < self.max_level:
            mid = (low + high) / 2
            go_left_old = cdf_old < mid
            go_left_new = cdf_new < mid

            if go_left_old == go_left_new:
                if go_left_new:
                    node = node.left
                    high = mid
                else:
                    node = node.right
                    low = mid
                level += 1
            else:
                break

        if level == self.max_level:
            return

        if go_left_old:
            if window_type == 'ref': node.n_ref_L -= 1
            else: node.n_cur_L -= 1
        else:
            if window_type == 'ref': node.n_ref_R -= 1
            else: node.n_cur_R -= 1

        if go_left_new:
            if window_type == 'ref': node.n_ref_L += 1
            else: node.n_cur_L += 1
        else:
            if window_type == 'ref': node.n_ref_R += 1
            else: node.n_cur_R += 1

        node.compute_log_bayes_factor()

        old_node = node.left if go_left_old else node.right
        old_low, old_high = (low, mid) if go_left_old else (mid, high)
        self._update_single_path(old_node, cdf_old, old_low, old_high, window_type, -1, level + 1)

        new_node = node.left if go_left_new else node.right
        new_low, new_high = (low, mid) if go_left_new else (mid, high)
        self._update_single_path(new_node, cdf_new, new_low, new_high, window_type, 1, level + 1)

    def _update_single_path(self, node, cdf, low, high, window_type, delta, current_level):
        for _ in range(current_level, self.max_level):
            mid = (low + high) / 2
            if cdf < mid:
                if window_type == 'ref': node.n_ref_L += delta
                else: node.n_cur_L += delta
                node.compute_log_bayes_factor()
                node = node.left
                high = mid
            else:
                if window_type == 'ref': node.n_ref_R += delta
                else: node.n_cur_R += delta
                node.compute_log_bayes_factor()
                node = node.right
                low = mid

    def _update_tree(self, val, window_type, delta=1):
        cdf = self.norm_cdf(val)
        node = self.root
        low, high = 0.0, 1.0

        for _ in range(self.max_level):
            mid = (low + high) / 2
            if cdf < mid:
                if window_type == 'ref': node.n_ref_L += delta
                else: node.n_cur_L += delta
                node.compute_log_bayes_factor()
                node = node.left
                high = mid
            else:
                if window_type == 'ref': node.n_ref_R += delta
                else: node.n_cur_R += delta
                node.compute_log_bayes_factor()
                node = node.right
                low = mid

    def _get_total_bf(self):
        total = 0.0
        q = [self.root]
        while q:
            node = q.pop(0)
            total += node.cached_log_B
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        return total

    def update(self, val: float) -> bool:
        if not self.is_initialized:
            self.cur_window.append(val)
            if len(self.cur_window) >= self.window_size:
                for v in self.cur_window:
                    self.ref_window.append(v)
                    self._update_tree(v, 'ref', 1)
                self.freeze_stats()
                self.cur_window.clear()
                self.is_initialized = True
            return False

        if len(self.cur_window) == self.window_size:
            old_val = self.cur_window.popleft()
            self._update_tree_sliding(old_val, val, 'cur')
            self.cur_window.append(val)
        else:
            self.cur_window.append(val)
            self._update_tree(val, 'cur', 1)

        if len(self.cur_window) < 50:
            return False

        log_bf = self._get_total_bf()
        return log_bf < self.threshold

    def reset(self):
        self.root = self._build_tree(0)
        self.ref_window.clear()
        self.cur_window.clear()
        self.is_initialized = False
        self.reset_stats()

# ==========================================
# ADWIN 和 KSWIN (按需保留)
# ==========================================
class ADWINDetector(BaseDriftDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.adwin = drift.ADWIN(delta=config.get('delta', 0.002))

    def update(self, val: float) -> bool:
        self.adwin.update(val)
        return self.adwin.drift_detected

    def reset(self):
        self.adwin = drift.ADWIN(delta=self.config.get('delta', 0.002))
        self.reset_stats()

class KSWINDetector(BaseDriftDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.kswin = drift.KSWIN(alpha=config.get('alpha', 0.001), window_size=config.get('window_size', 5120), stat_size=config.get('stat_size', 1000))

    def update(self, val: float) -> bool:
        self.kswin.update(val)
        return self.kswin.drift_detected

    def reset(self):
        self.kswin = drift.KSWIN(alpha=self.config.get('alpha', 0.001), window_size=self.config.get('window_size', 5120), stat_size=self.config.get('stat_size', 1000))
        self.reset_stats()