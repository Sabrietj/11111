from typing import List, Dict, Optional, Tuple, Any
import ast
import logging
import os, sys
from collections import Counter

# 添加../utils目录到Python搜索路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
# 导入配置管理模块
from logging_config import setup_preset_logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.DEBUG)    

verbose = False

BENIGN_PREFIXES = ("benign", "normal", "legitimate")  # 可根据实际数据集调整

def normalize_label(label: str) -> str:
    if label is None:
        return ""
    return str(label).strip().lower()        
class SessionParser:
    def __init__(self, flow_node_builder, session_label_id_map=None):
        self.flow_node_builder = flow_node_builder
        self.session_label_id_map = session_label_id_map or {}
        # dominant 攻击阈值（多攻击 session 使用）
        # 阈值：你可以写进 config        
        self.dominant_ratio_threshold = 0.8
    
    def extract_flow_uid_list(self, session_row) -> List[str]:
        """从会话行中提取流UID列表"""
        if 'flow_uid_list' not in session_row:
            return []
        
        try:
            return ast.literal_eval(session_row['flow_uid_list'])
        except (ValueError, SyntaxError):
            return []

    @staticmethod
    def is_malicious(raw_label: str) -> bool:
        # CIC-IoMT-2024: benign_xxx
        # CIC-IDS-2017 / 2018: benign
        norm_label = normalize_label(raw_label)
        if norm_label.startswith(BENIGN_PREFIXES):
            return False
        return True

    # 辅助函数：基于配置的标签映射进行前缀匹配
    def match_configured_label(self, raw_label_norm: str) -> Optional[str]:
        """
        Map a flow-level attack label to a configured session-level attack category.
        Only called when the flow is already known to be malicious.
        """
        raw_label_norm = raw_label_norm.strip().lower()

        for configured_label in self.session_label_id_map.keys():
            configured_lower = configured_label.lower()

            # 精确匹配
            if raw_label_norm == configured_lower:
                return configured_label

            # 前缀 / 子串匹配（保留你的需求）
            # “substring matching is intentional to support heterogeneous label styles across datasets”
            if configured_lower in raw_label_norm:
                return configured_label

        return None

    def aggregate_session_label(self, flow_uid_list):
        label_name = self.aggregate_session_label_without_label_id(flow_uid_list)

        # ⭐【关键】统一做 label 规范化（大小写 / 空格）
        label_name = normalize_label(label_name)

        is_malicious = SessionParser.is_malicious(label_name)

        # mixed / unknown 直接跳过
        if label_name == "mixed":
            return label_name, -1, is_malicious

        label_id = self.session_label_id_map.get(label_name)
        if label_id is None:
            logger.error(
                f"Session label '{label_name}' not found in session_label_id_map. "
                f"Available labels: {list(self.session_label_id_map.keys())}"
            )
            return label_name, -1, is_malicious

        return label_name, label_id, is_malicious

    def aggregate_session_label_without_label_id(self, flow_uid_list):
        """
        基于 flow 级标签聚合 session 级标签（不依赖 label_id）。

        聚合策略说明：
        1️⃣ 不包含任何恶意 flow：
            - 若所有 flow 均为同一 benign 类型，返回该 benign 类型；
            - 若存在多种 benign 类型（如 CIC-IoMT-2024 场景），
            返回统一的 benign 标签（如 'benign_unknown'）。

        2️⃣ 仅包含一种攻击类型：
            - 直接返回该攻击类型作为 session 标签。

        3️⃣ 包含多种攻击类型：
            - 采用“主导攻击（dominant attack）”规则：
            若某一攻击类型在恶意 flow 中占比 ≥ dominant_ratio_threshold，
            则将该攻击视为 session 的主导攻击，返回该攻击类型；
            - 否则，认为该 session 为真正的混合攻击行为，标记为 'mixed'。

        设计说明：
        - 该 dominant 规则用于处理真实网络中的复杂攻击场景，
        例如 DDoS session 中可能同时包含多种 DoS 子类型。
        - 被标记为 'mixed' 的 session 表示不存在明确主导攻击模式，
        通常在建图或下游任务中单独处理或直接丢弃。

        Fail-fast 策略：
        - 若遇到无法识别的 flow 标签，立即抛出异常，
        防止标签配置错误导致的隐式数据污染。
        """
        benign_counter = Counter()
        attack_counter = Counter()
        
        for flow_uid in flow_uid_list:
            flow_record = self.flow_node_builder.get_flow_record(flow_uid)
            if flow_record is None:
                continue

            raw_label = str(flow_record.get("label", ""))

            # ---------- 1. class_type 判断 ----------
            # 统一走 class_type 映射逻辑
            class_type = self.match_configured_label(raw_label)
            if class_type is None:
                raise ValueError(
                    f"[SessionLabelError] Unrecognized class label '{raw_label}' "
                    f"(flow_uid={flow_uid}). "
                    f"Please update session_label_id_map."
                )

            # ---------- 2. attack和benign类别统计  ----------
            if SessionParser.is_malicious(raw_label):
                attack_counter[class_type] += 1
            else:
                benign_counter[class_type] += 1
                

        # ---------- 聚合规则 ----------
        # 1️⃣ 没有恶意 flow        
        if sum(attack_counter.values()) == 0: 
            if len(benign_counter) == 1:
                return next(iter(benign_counter))
            else:
                # CIC-IoMT-2024的数据集有多种benign类型
                # 'benign_unknown', 'benign_active', 'benign_broker', 'benign_idle', 'benign_interaction', 'benign_power',
                label_id = self.session_label_id_map.get("benign_unknown")
                if label_id is not None:
                    return "benign_unknown"
                else:
                    return "benign"

        # 2️⃣ 只有一种攻击类型
        if len(attack_counter) == 1:
            return next(iter(attack_counter))

        # 3️⃣ 多种攻击类型 → dominant 判定
        total_attack = sum(attack_counter.values())
        dominant_label, dominant_count = attack_counter.most_common(1)[0]
        dominant_ratio = dominant_count / total_attack

        # ⭐ 阈值判定
        if dominant_ratio >= self.dominant_ratio_threshold:
            logger.info(
                f"[SessionLabelDominant] dominant={dominant_label} "
                f"ratio={dominant_ratio:.3f} "
                f"threshold={self.dominant_ratio_threshold} "
                f"counts={dict(attack_counter)}"
            )
            return dominant_label

        # 4️⃣ 真的混合（谁也不占主导）
        logger.warning(
            f"[SessionLabelMixed] counts={dict(attack_counter)} "
            f"benign={dict(benign_counter)}"
        )
        return "mixed"
