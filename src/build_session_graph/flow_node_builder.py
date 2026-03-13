import tqdm
import os, sys
import pandas as pd
import numpy as np
import json
import ast
from typing import Union, List
import logging
from collections import Counter
from transformers import BertTokenizer
from types import MappingProxyType
from typing import List, Tuple
import re
import mmap
import csv
import sys

def set_csv_field_size_limit():
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

set_csv_field_size_limit()

# 导入配置管理模块
try:
    # 添加../utils目录到Python搜索路径
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
    sys.path.insert(0, utils_path)
    from zeek_columns import conn_columns, http_columns
    from zeek_columns import conn_numeric_columns, conn_categorical_columns, conn_textual_columns
    from zeek_columns import flowmeter_numeric_columns, flowmeter_categorical_columns, flowmeter_textual_columns
    from zeek_columns import ssl_numeric_columns, ssl_categorical_columns, ssl_textual_columns
    from zeek_columns import dns_numeric_columns, dns_categorical_columns, dns_textual_columns
    from zeek_columns import x509_numeric_columns, x509_categorical_columns, x509_textual_columns
    from zeek_columns import max_x509_cert_chain_len, dtype_dict_in_flow_csv
    from config_manager import read_session_label_id_map, read_text_encoder_config
    from logging_config import setup_preset_logging
    # 使用统一的日志配置
    logger = setup_preset_logging(log_level=logging.DEBUG)    
except ImportError:
    # 如果导入失败，提供一个默认实现
    def read_session_label_id_map():
        return {'benign': 0, 'background': 1, 'mixed': 2, 'malicious': 3}
    
class FlowNodeBuilder:
    """处理将网络流数据构造成图节点的类，负责加载和预处理流特征"""
    def __init__(self, flow_csv_path, 
                 session_label_id_map, 
                 max_packet_sequence_length, 
                 text_encoder_name, 
                 max_text_length, 
                 thread_count=1, 
                 enabled_views=None,
                 exclude_ports=None, 
                 exclude_services=None, 
                 storage_mode = "dict",
        ):
        """
        storage_mode 有 "dict" 或 "offset" 两种：
        * "dict" mode 会把整个all_embedded_flow.csv文件加载到内存中，加快扫描速度；
        * "offset" mode 避免加载整个all_embedded_flow.csv文件，只是需要的时候到外存取数据访问。

        实验中发现 “offset” mode可以构建完 categorical vocab，也可以计算完 numeric stats。
        但是，进入 process_sessions_parallel阶段就会卡死，原因是 session 中的 flow_uid 是无序的。这意味着：
        * 线程 1 → seek 到 1GB 位置
        * 线程 2 → seek 到 200MB 位置
        * 线程 3 → seek 到 1.5GB 位置
        * 线程 4 → seek 到 20MB 位置
        ...
        硬盘疯狂来回跳。
        * 如果是机械硬盘 → 基本等于锁死
        * 如果是 SATA SSD → IOPS 被打爆
        * 即使 NVMe → 20 线程 + Python 解析 CSV 也会炸        
        """
        self.mtu_normalize = 1500   # bytes
        self.max_packet_sequence_length = max_packet_sequence_length
        self.text_encoder_name = text_encoder_name
        self.max_text_length = max_text_length
        self.text_tokenizer, self.max_text_length = load_text_tokenizer(
            model_name=self.text_encoder_name,
            max_text_length=self.max_text_length
        )

        self.thread_count = thread_count
        self.enabled_views = enabled_views or {
            "flow_numeric_features": True,
            "flow_categorical_features": True,
            "flow_textual_features": True,
            "packet_len_seq": True,
            "packet_iat_seq": True,
            "domain_probs": True,
            "ssl_numeric_features": True,
            "ssl_categorical_features": True,
            "ssl_textual_features": True,
            "x509_numeric_features": True,
            "x509_categorical_features": True,
            "x509_textual_features": True,
            "dns_numeric_features": True,
            "dns_categorical_features": True,
            "dns_textual_features": True,
        }

        # 读取会话标签映射配置并计算类别数量
        self.session_label_id_map = session_label_id_map
        self.num_classes = len(set(self.session_label_id_map.values()))
        logger.info(f"Loaded session label string-to-id mapping: len={self.num_classes}, mapping={self.session_label_id_map}")

        self.exclude_ports = set(int(p) for p in (exclude_ports or []))
        self.exclude_services = set(s.lower() for s in (exclude_services or []))
        self.excluded_flow_uid_set = set()
        self.excluded_flow_metadata = {}
        self.protocol_filtered_count = 0

        self.storage_mode = storage_mode
        self.load_all_flows(flow_csv_path)

        self.categorical_vocabulary_group = self.scan_all_flows_for_categorical_topk_vocab_group()
        self.global_node_feature_dims, self.numeric_feature_stats = self.scan_all_flows_for_node_feature_dims_and_numeric_stats(
            enabled_views = self.enabled_views,
            max_text_length = self.max_text_length,
            text_tokenizer = self.text_tokenizer,
            max_packet_sequence_length = self.max_packet_sequence_length,
            categorical_vocabulary_group = self.categorical_vocabulary_group,
            num_classes = self.num_classes,
        )
        self.categorical_vocabulary_group = MappingProxyType(self.categorical_vocabulary_group)        
        self.global_node_feature_dims = MappingProxyType(self.global_node_feature_dims)
        self.numeric_feature_stats = MappingProxyType(self.numeric_feature_stats)

        logger.info("✅ Global node feature dimension summary (enabled views):")
        for view_name, dim in self.global_node_feature_dims.items():
            enabled = self.enabled_views.get(view_name, False)
            status = "ON " if enabled else "OFF"
            logger.info(f"  - [{status}] {view_name}: {dim}")


    def load_all_flows(self, flow_csv_path):
        if self.storage_mode == "dict":
            return self.load_all_flows_by_dict(flow_csv_path)
    
        elif self.storage_mode == "offset":
            return self.load_all_flows_by_offset(flow_csv_path)
        
        else:
            raise ValueError(f"Unsupported storage_mode: {self.storage_mode}")


    def load_all_flows_by_dict(self, flow_csv_path):
        # logger.info("Loading flow CSV file as a pandas dataframe...")
        # flow_df = read_large_csv_with_progress(flow_csv_path)

        logger.info("Counting total number of lines in flow CSV...")

        with open(flow_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            flow_df_len = sum(1 for _ in f) - 1  # 减去header

        logger.info(f"Total flow rows: {flow_df_len}")        
        
        # 构建 flow uid -> record的字典结构，方便后续从session row基于flow_uid_list的检索归属于session的网络流。
        self.flow_dict = {}
        logger.info("Building flow dictionary using the 'uid' fields as keys...")
        processed_rows = 0
        with tqdm.tqdm(total=flow_df_len, desc="Building flow_dict", unit="flow") as pbar:
            for chunk in pd.read_csv(flow_csv_path, chunksize=200_000, low_memory=False):
                for _, row in chunk.iterrows():
                    flow_uid = row['uid']
                    excluded, reason = self._should_exclude_flow(row)
                    if excluded:
                        self.excluded_flow_uid_set.add(flow_uid)
                        self.excluded_flow_metadata[flow_uid] = {
                            "orig_h": row.get("conn.id.orig_h"),
                            "orig_p": row.get("conn.id.orig_p"),
                            "resp_h": row.get("conn.id.resp_h"),
                            "resp_p": row.get("conn.id.resp_p"),
                            "proto": row.get("conn.proto"),
                            "service": row.get("conn.service"),
                            "label": row.get("label"),
                            "reason": reason,
                        }
                        continue   # ⭐ 核心：根据配置的网络流过滤功能，直接跳过需要过滤的flows

                    # ===== 在这里一次性规范化 flow_record =====
                    flow_record = row.to_dict()
                    flow_record['uid'] = flow_uid
                    try:
                        flow_record['ts'] = float(flow_record['conn.ts']) # 每个record有时间戳，方便后续建图
                    except Exception:
                        flow_record['ts'] = 0.0

                    self.flow_dict[flow_uid] = MappingProxyType(flow_record) # 把 flow_record 冻结为只读

                processed_rows += len(chunk)
                pbar.update(len(chunk))

                # 4️⃣ 显式释放 chunk，节约内存
                del chunk
                import gc
                gc.collect()

        
        logger.info(f"[Protocol Filter] filtered_count = {self.protocol_filtered_count}")
        if flow_df_len > 0:
            logger.info(f"[Protocol Filter] ratio = {self.protocol_filtered_count / flow_df_len:.4f}")


    def load_all_flows_by_offset(self, flow_csv_path):
        logger.info("Building uid -> file_offset index...")

        self.flow_csv_path = flow_csv_path
        self.uid_to_offset = {}
        self.excluded_flow_uid_set = set()

        with open(flow_csv_path, 'r', encoding='utf-8', newline='') as flow_csv_file:
            # 读取 header
            header_offset = flow_csv_file.tell()
            header_line = flow_csv_file.readline()
            self.csv_columns = next(csv.reader([header_line]))
            expected_cols = len(self.csv_columns)
            uid_idx = self.csv_columns.index("uid")

            while True:
                record_offset = flow_csv_file.tell()
                if record_offset == os.fstat(flow_csv_file.fileno()).st_size:
                    break

                row = self.read_complete_csv_record(flow_csv_file, expected_cols)

                if len(row) != expected_cols:
                    logger.warning("[OFFSET_PARSE_ERROR] malformed record")
                    continue

                flow_uid = row[uid_idx]

                flow_record = dict(zip(self.csv_columns, row))
                flow_record["uid"] = flow_uid
                try:
                    flow_record["ts"] = float(flow_record["conn.ts"])
                except:
                    flow_record["ts"] = 0.0

                excluded, reason = self._should_exclude_flow(flow_record)
                if excluded:
                    self.excluded_flow_uid_set.add(flow_uid)
                    continue

                self.uid_to_offset[flow_uid] = record_offset

        # reopen file for future random access
        self._flow_file = open(flow_csv_path, 'r', encoding='utf-8', newline='')

    def read_complete_csv_record(self, flow_csv_file, expected_cols):
        buffer = ""
        row = None

        while True:
            line = flow_csv_file.readline()
            if not line:
                break

            buffer += line

            try:
                reader = csv.reader([buffer])
                row_candidate = next(reader)
            except Exception:
                continue

            if len(row_candidate) == expected_cols:
                row = row_candidate
                break

        if row is None:
            return []

        return row

    def close_flow_file(self):
        if hasattr(self, "_flow_file_mmap"):
            self._flow_file_mmap.close()
        if hasattr(self, "_flow_file"):
            self._flow_file.close()

    def __del__(self):
        self.close_flow_file()
                    
    def get_flow_record(self, flow_uid):
        if self.storage_mode == "dict":
            return self.get_flow_record_by_dict(flow_uid)
        
        elif self.storage_mode == "offset":
            return self.get_flow_record_by_offset(flow_uid)
        
        else:
            raise ValueError(f"Unsupported storage_mode: {self.storage_mode}")

    def get_flow_record_by_dict(self, flow_uid, verbose=False):
        """获取指定UID的流记录"""
        record = self.flow_dict.get(flow_uid)
        if record is None: 
            if flow_uid in self.excluded_flow_uid_set:
                metadata = self.excluded_flow_metadata.get(flow_uid, {})
                if verbose:
                    logger.warning(
                        f"[FILTERED_FLOW_ACCESS] "
                        f"uid={flow_uid} "
                        f"{metadata.get('orig_h')}:{metadata.get('orig_p')} -> "
                        f"{metadata.get('resp_h')}:{metadata.get('resp_p')} "
                        f"proto={metadata.get('proto')} "
                        f"service={metadata.get('service')} "
                        f"label={metadata.get('label')} "
                        f"reason={metadata.get('reason')}"
                    )
            else:
                logger.warning(f"[MISSING_FLOW] uid={flow_uid} not found in flow_dict and not filtered")

        return record
    
    def get_flow_record_by_offset(self, flow_uid):
        offset = self.uid_to_offset.get(flow_uid)
        if offset is None:
            return None

        self._flow_file.seek(offset)
        row = self.read_complete_csv_record(self._flow_file, len(self.csv_columns))

        if len(row) != len(self.csv_columns):
            return None

        flow_record = dict(zip(self.csv_columns, row))
        flow_record["uid"] = flow_uid
        try:
            flow_record["ts"] = float(flow_record["conn.ts"])
        except:
            flow_record["ts"] = 0.0

        return flow_record

    def get_all_flow_uids(self):
        """获取所有流UID"""
        if self.storage_mode == "dict":
            return self.get_all_flow_uids_by_dict()
        
        elif self.storage_mode == "offset":
            return self.get_all_flow_uids_by_offset()
        
        else:
            raise ValueError(f"Unsupported storage_mode: {self.storage_mode}")

    def get_all_flow_uids_by_dict(self):
        return list(self.flow_dict.keys())

    def get_all_flow_uids_by_offset(self):
        return list(self.uid_to_offset.keys())
    
    def get_num_classes(self):
        """获取类别数量"""
        return self.num_classes
    
    def _should_exclude_flow(self, row) -> Tuple[bool, str]:
        # -------- port ----------
        if self.exclude_ports:
            try:
                resp_p = int(row.get("conn.id.resp_p", -1))
                if resp_p in self.exclude_ports:
                    return (True, "port")
            except Exception:
                pass

        # -------- service ----------
        if self.exclude_services:
            service = row.get("conn.service")
            if isinstance(service, str):
                service = service.strip().lower()
                if service in self.exclude_services:
                    return (True, "service")

        # -------- label-driven protocol consistency filter ----------
        label = row.get("label")
        if isinstance(label, str):
            label_lower = label.lower()
            required_proto = self._label_requires_proto(label_lower)

            if required_proto is not None:
                flow_uid = str(row.get("uid", "")).lower().strip()
                flow_proto = str(row.get("conn.proto", "")).lower().strip()
                flow_service = str(row.get("conn.service", "")).lower().strip()

                # ===== MQTT 特殊处理 =====
                if required_proto == "mqtt":
                    # MQTT 必须是 TCP + service = mqtt
                    if not (flow_proto == "tcp" and flow_service == "mqtt"):
                        self.protocol_filtered_count += 1
                        if self.protocol_filtered_count < 50:
                            logger.debug(f"Proto mismatch: {flow_uid}, label={label}, proto={flow_proto}, service={flow_service}")
                        return (True, "proto_mismatch")

                # ===== 普通传输层协议 =====
                else:
                    if flow_proto != required_proto:
                        self.protocol_filtered_count += 1
                        if self.protocol_filtered_count < 50:
                            logger.debug(f"Proto mismatch: {flow_uid}, label={label}, proto={flow_proto}, service={flow_service}")
                        return (True, "proto_mismatch")
                
        return (False, None)

    def _label_requires_proto(self, label_lower: str):
        """
        根据攻击标签推断该标签所要求的传输层协议类型。

        ----------------------------------------------------------------------
        设计背景说明（非常重要）
        ----------------------------------------------------------------------
        在部分数据集（例如 CIC-IoMT-2024）中，攻击标签并不是
        在 flow 级别精细标注的，而是基于“场景目录名”统一赋值的。

        例如目录名可能为：
            "malicious_TCP_IP-DDoS-TCP"
            "malicious_TCP_IP-DDoS-UDP"

        该目录下的所有 flow 都会被赋予相同标签，
        不论这些 flow 的真实 conn.proto 是 tcp / udp / icmp。

        这会导致标签污染问题，例如：
            - 标记为 "DDoS-TCP" 的目录中，仍然存在 UDP 流量（如 DNS 53）
            - 标记为 "DDoS-UDP" 的目录中，可能混入 TCP 流量
            - 背景服务流量与攻击流量混在一起

        为缓解这种“场景级标签噪声”问题，我们引入
        “标签启发的协议一致性约束”：

            如果标签中显式包含 TCP / UDP / ICMP / MQTT 等信息，
            则仅保留 conn.proto 与标签语义一致的 flow。

        本函数的作用是：
            从标签文本中识别是否隐含了必须的协议类型。

        ----------------------------------------------------------------------
        返回值说明
        ----------------------------------------------------------------------
            'tcp'   -> 标签表明是 TCP 类攻击
            'udp'   -> 标签表明是 UDP 类攻击
            'icmp'  -> 标签表明是 ICMP 类攻击
            'mqtt'  -> 标签表明是 MQTT 类攻击
            None    -> 标签未指定传输层协议（不进行过滤）

        ----------------------------------------------------------------------
        重要说明
        ----------------------------------------------------------------------
        1. 该函数主要为 CIC-IoMT-2024 数据集设计。
        2. 不会影响 CIC-IDS-2017 / CIC-IDS-2018 数据集，
        因为这些数据集的标签（如 DoS Hulk、PortScan、Bot）
        并未在标签中编码传输层协议信息。
        3. 匹配忽略大小写，支持以下形式：
            DDoS-TCP, DDoS_TCP, TCP-DDoS, TCP_DDoS,
            DoS-UDP, UDP_DoS 等。
        """
        if not isinstance(label_lower, str):
            return None

        patterns = {
            "tcp":  r"\b(ddos|dos)[-_]?tcp\b|\btcp[-_]?(ddos|dos)\b",
            "udp":  r"\b(ddos|dos)[-_]?udp\b|\budp[-_]?(ddos|dos)\b",
            "icmp": r"\b(ddos|dos)[-_]?icmp\b|\bicmp[-_]?(ddos|dos)\b",
            "mqtt": r"\b(ddos|dos)[-_]?mqtt\b|\bmqtt[-_]?(ddos|dos)\b",
        }

        for proto, pattern in patterns.items():
            if re.search(pattern, label_lower):
                return proto

        return None

    def scan_all_flows_for_categorical_topk_vocab_group(self):
        """
        仅基于 flow_dict 构建 categorical 特征的 vocabulary（高效版，仅扫描一次 flow_dict）。
        flow_dict: { uid -> flow_record(dict) }

        返回:
            vocab_group = {
                col_name: { token -> index }
            }
        """

        top_k_cat = 500  # 可调
        top_k_map = {
            # ---------------- SSL ----------------
            "ssl.cipher": 50,
            "ssl.curve": 10,
            "ssl.version": 6,
            "ssl.next_protocol": 20,
            "ssl.client_signature_algorithms": 50,
            "ssl.server_signature_algorithms": 50,
            "ssl.client_key_exchange_groups": 20,
            "ssl.server_key_exchange_groups": 20,
            "ssl.client_supported_versions": 10,
            "ssl.server_supported_versions": 10,

            # ---------------- DNS ----------------
            "dns.qtype": 40,
            "dns.qclass": 10,
            "dns.rcode_name": 20,
            "dns.qtype_name": 40,
            "dns.qclass_name": 10,
            "dns.rcode": 10,

            # ---------------- conn ----------------
            "conn.proto": 10,
            "conn.service": 50,
            "conn.conn_state": 20,
            "conn.history": 30,
            "conn.local_orig": 3,
            "conn.local_resp": 3,

            # ---------------- flowmeter ----------------
            # Flowmeter categorical（只有 proto）
            "flowmeter.proto": 10,
        }
        
        # 每种类型的列名，加上前缀 → 真实 DataFrame 列名
        conn_cat_cols_prefixed      = [f"conn.{c}"      for c in conn_categorical_columns]
        flowmeter_cat_cols_prefixed = [f"flowmeter.{c}" for c in flowmeter_categorical_columns]
        flow_cat_cols_prefixed = conn_cat_cols_prefixed + flowmeter_cat_cols_prefixed
        ssl_cat_cols_prefixed       = [f"ssl.{c}"       for c in ssl_categorical_columns]
        x509_cat_cols_prefixed = []
        for n in [0, 1, 2]:
            x509_cat_cols_prefixed += [f"x509.cert{n}.{c}" for c in x509_categorical_columns]
        dns_cat_cols_prefixed       = [f"dns.{c}"       for c in dns_categorical_columns]
        categorical_columns = (
            flow_cat_cols_prefixed +
            ssl_cat_cols_prefixed +
            x509_cat_cols_prefixed +
            dns_cat_cols_prefixed
        )

        # Counter 初始化
        categorical_vocab_counter = {col: Counter() for col in categorical_columns}

        # 🔥 只扫描一次 flow_dict（高效）
        all_flow_uids = self.get_all_flow_uids()
        for flow_uid in tqdm.tqdm(all_flow_uids, 
                                total=len(all_flow_uids),
                                desc="[1st PASS] Scanning categorical vocab", 
                                unit="flow"):
            flow_record = self.get_flow_record(flow_uid)

            for col in categorical_columns:
                raw = flow_record[col]

                if raw is None:
                    token = "<OOV>"
                else:
                    token = str(raw).strip() or "<OOV>"

                categorical_vocab_counter[col][token] += 1

        # 构建最终 vocab_group
        categorical_vocab_group = {}
        for col in categorical_columns:
            counter = categorical_vocab_counter[col]

            if not counter:
                categorical_vocab_group[col] = {"<OOV>": 0}
                continue

            # top-k
            this_top_k = next((v for k, v in top_k_map.items() if col.startswith(k)), top_k_cat)
            most = counter.most_common(this_top_k)

            values = [v for v, _ in most]
            mapping = {v: i+1 for i, v in enumerate(values)}
            mapping["<OOV>"] = 0

            categorical_vocab_group[col] = mapping

        return categorical_vocab_group
    
    def scan_all_flows_for_node_feature_dims_and_numeric_stats(self, enabled_views, max_text_length, text_tokenizer, 
                                                               max_packet_sequence_length, categorical_vocabulary_group, num_classes):
        """计算全局的数值型+类别型节点特征维度"""
        global_node_feature_dims = {
            "flow_numeric_features": 0,
            "flow_categorical_features": 0,
            "packet_len_seq": 0,
            "packet_iat_seq": 0,
            "domain_probs": 0,
            "ssl_numeric_features": 0,
            "ssl_categorical_features": 0,
            "x509_numeric_features": 0,
            "x509_categorical_features": 0,
            "dns_numeric_features": 0,
            "dns_categorical_features": 0,
        }

        numeric_feature_stats = {}
        
        for view_name in [
            "flow_numeric_features",
            "ssl_numeric_features",
            "x509_numeric_features",
            "dns_numeric_features",
        ]:
            if enabled_views.get(view_name, False):
                numeric_feature_stats[view_name] = {
                    "count": 0,
                    "sum": None,
                    "sum_of_squares": None,
                }

        logger.info("Calculating global node feature dimensions and numeric features' statistics from flow_dict...")

        all_flow_uids = self.get_all_flow_uids()
        for flow_uid in tqdm.tqdm(
            all_flow_uids,
            total=len(all_flow_uids),
            desc="[2nd PASS] Calc global node feature dims and numeric features' statistics",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5            
        ):
            flow_record = self.get_flow_record(flow_uid)
            try:
                # 提取网络流的各类特征向量
                if enabled_views.get("flow_numeric_features", False) \
                    or enabled_views.get("flow_categorical_features", False) \
                    or enabled_views.get("flow_textual_features", False):
                    flow_numeric_features, flow_categorical_features, flow_textual_features = extract_conn_and_flowmeter_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

                # 计算网络流的数值型+类别型特征维度
                if enabled_views.get("flow_numeric_features", False):
                    global_node_feature_dims['flow_numeric_features'] = max(
                        global_node_feature_dims['flow_numeric_features'], len(flow_numeric_features) if len(flow_numeric_features) > 0 else 1)
                    
                    vec = np.array(flow_numeric_features, dtype=np.float64)
                    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):                        
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0) # 修复NaN和Inf值

                    stats = numeric_feature_stats["flow_numeric_features"]
                    if stats["sum"] is None:
                        stats["sum"] = np.zeros_like(vec)
                        stats["sum_of_squares"] = np.zeros_like(vec)

                    stats["sum"] += vec
                    stats["sum_of_squares"] += vec * vec
                    stats["count"] += 1
                    
                if enabled_views.get("flow_categorical_features", False):
                    global_node_feature_dims['flow_categorical_features'] = max(
                        global_node_feature_dims['flow_categorical_features'], len(flow_categorical_features) if len(flow_categorical_features) > 0 else 1)
                    
                # 计算数据包时间序列特征向量维度              
                if enabled_views.get("packet_len_seq", False):
                    global_node_feature_dims["packet_len_seq"] = max_packet_sequence_length
                else:
                    global_node_feature_dims["packet_len_seq"] = 0

                if enabled_views.get("packet_iat_seq", False):
                    global_node_feature_dims["packet_iat_seq"] = max_packet_sequence_length
                else:
                    global_node_feature_dims["packet_iat_seq"] = 0
                    
                # 提取基于domain-app共现概率的域名嵌入特征向量
                if enabled_views.get("domain_probs", False):
                    domain_probs = extract_domain_name_probabilities(flow_record, num_classes)
                # 计算基于domain-app共现概率的域名嵌入特征向量维度
                if enabled_views.get("domain_probs", False):
                    global_node_feature_dims['domain_probs'] = max(
                        global_node_feature_dims['domain_probs'], len(domain_probs) if len(domain_probs) > 0 else 1)

                # 提取SSL的各类特征向量
                if enabled_views.get("ssl_numeric_features", False) \
                    or enabled_views.get("ssl_categorical_features", False) \
                    or enabled_views.get("ssl_textual_features", False):
                    ssl_numeric_features, ssl_categorical_features, ssl_textual_features = extract_ssl_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

                # 计算SSL的数值型+类别型特征向量的维度
                if enabled_views.get("ssl_numeric_features", False):
                    global_node_feature_dims['ssl_numeric_features'] = max(
                        global_node_feature_dims['ssl_numeric_features'], len(ssl_numeric_features) if len(ssl_numeric_features) > 0 else 1)
                    
                    vec = np.array(ssl_numeric_features, dtype=np.float64)
                    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):                        
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0) # 修复NaN和Inf值
                    
                    stats = numeric_feature_stats["ssl_numeric_features"]
                    if stats["sum"] is None:
                        stats["sum"] = np.zeros_like(vec)
                        stats["sum_of_squares"] = np.zeros_like(vec)

                    stats["sum"] += vec
                    stats["sum_of_squares"] += vec * vec
                    stats["count"] += 1
                
                if enabled_views.get("ssl_categorical_features", False):
                    global_node_feature_dims['ssl_categorical_features'] = max(
                        global_node_feature_dims['ssl_categorical_features'], len(ssl_categorical_features) if len(ssl_categorical_features) > 0 else 1)
                    
                # 提取X509的各类特征向量
                if enabled_views.get("x509_numeric_features", False) \
                    or enabled_views.get("x509_categorical_features", False) \
                    or enabled_views.get("x509_textual_features", False):
                    x509_numeric_features, x509_categorical_features, x509_textual_features = extract_x509_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

                # 计算X509的数值型+类别型特征向量的维度
                if enabled_views.get("x509_numeric_features", False):
                    global_node_feature_dims['x509_numeric_features'] = max(
                        global_node_feature_dims['x509_numeric_features'], len(x509_numeric_features) if len(x509_numeric_features) > 0 else 1)
                    
                    vec = np.array(x509_numeric_features, dtype=np.float64)
                    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):                        
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0) # 修复NaN和Inf值

                    stats = numeric_feature_stats["x509_numeric_features"]
                    if stats["sum"] is None:
                        stats["sum"] = np.zeros_like(vec)
                        stats["sum_of_squares"] = np.zeros_like(vec)

                    stats["sum"] += vec
                    stats["sum_of_squares"] += vec * vec
                    stats["count"] += 1
                
                if enabled_views.get("x509_categorical_features", False):
                    global_node_feature_dims['x509_categorical_features'] = max(
                        global_node_feature_dims['x509_categorical_features'], len(x509_categorical_features) if len(x509_categorical_features) > 0 else 1)

                # 提取DNS的各类特征向量
                if enabled_views.get("dns_numeric_features", False) \
                    or enabled_views.get("dns_categorical_features", False) \
                    or enabled_views.get("dns_textual_features", False):
                    dns_numeric_features, dns_categorical_features, dns_textual_features = extract_dns_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

                # 计算DNS的数值型+类别型特征向量的维度                
                if enabled_views.get("dns_numeric_features", False):
                    global_node_feature_dims['dns_numeric_features'] = max(
                        global_node_feature_dims['dns_numeric_features'], len(dns_numeric_features) if len(dns_numeric_features) > 0 else 1)
                    
                    vec = np.array(dns_numeric_features, dtype=np.float64)
                    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0) # 修复NaN和Inf值

                    stats = numeric_feature_stats["dns_numeric_features"]
                    if stats["sum"] is None:
                        stats["sum"] = np.zeros_like(vec)
                        stats["sum_of_squares"] = np.zeros_like(vec)

                    stats["sum"] += vec
                    stats["sum_of_squares"] += vec * vec
                    stats["count"] += 1
                
                if enabled_views.get("dns_categorical_features", False):
                    global_node_feature_dims['dns_categorical_features'] = max(
                        global_node_feature_dims['dns_categorical_features'], len(dns_categorical_features) if len(dns_categorical_features) > 0 else 1)
                    
            except Exception as e:
                logger.error(f"Flow {flow_uid} 特征提取错误: {e}")
                continue
        
        logger.info(f"Global feature dimensions: {global_node_feature_dims}, with max_packet_sequence_length = {max_packet_sequence_length}")
        
        # 添加调试信息：显示Flow、SSL、X509、和DNS特征的实际维度
        logger.info(f"Flow feature dimension breakdown: numeric={global_node_feature_dims['flow_numeric_features']}, categorical={global_node_feature_dims['flow_categorical_features']}")
        logger.info(f"SSL feature dimension breakdown: numeric={global_node_feature_dims['ssl_numeric_features']}, categorical={global_node_feature_dims['ssl_categorical_features']}")
        logger.info(f"X509 feature dimension breakdown: numeric={global_node_feature_dims['x509_numeric_features']}, categorical={global_node_feature_dims['x509_categorical_features']}")
        logger.info(f"DNS feature dimension breakdown: numeric={global_node_feature_dims['dns_numeric_features']}, categorical={global_node_feature_dims['dns_categorical_features']}")

        for k, stats in numeric_feature_stats.items():
            count = stats["count"]
            # ⭐ 核心防护
            if count == 0 or stats["sum"] is None:
                logger.warning(
                    f"scan_flow_dict_for_node_feature_dims_and_numeric_stats(): [NUMERIC-STATS] Skip {k}: count={count}, sum is None"
                )
                stats["mean"] = []
                stats["std"] = []

            else:
                mean = stats["sum"] / count
                var = stats["sum_of_squares"] / count - mean * mean
                std = np.sqrt(np.maximum(var, 1e-12))

                stats["mean"] = mean.tolist()
                stats["std"] = std.tolist()
            
        return global_node_feature_dims, numeric_feature_stats

    def get_global_node_feature_dims(self, key):
        assert hasattr(self, 'global_node_feature_dims'), \
            "global_node_feature_dims must be initialized in __init__"
            
        return self.global_node_feature_dims[key]
    
    def build_node_features(self, flow_uids):
        """为指定的流UID构建节点特征"""
        enabled_views = self.enabled_views
        max_text_length = self.max_text_length
        categorical_vocabulary_group = self.categorical_vocabulary_group
        text_tokenizer = self.text_tokenizer
        numeric_feature_stats = self.numeric_feature_stats
        mtu_normalize = self.mtu_normalize
        max_packet_sequence_length = self.max_packet_sequence_length
        num_classes = self.num_classes

        if not flow_uids:
            return

        logger.debug("build_node_features(): begin")
        def is_nan_or_inf(x):
            """更全面的NaN/Inf检查"""
            if x is None:
                return True
            try:
                # 处理numpy类型
                if hasattr(x, 'dtype'):
                    return np.isnan(x) or np.isinf(x)
                # 处理Python数值类型
                elif isinstance(x, (int, float, np.number)):
                    return np.isnan(x) or np.isinf(x)
                return False
            except (TypeError, ValueError):
                return False
            
        nodes = []
        # 提取每个节点的特征
        for flow_uid in flow_uids:
            flow_record = self.get_flow_record(flow_uid)
            if flow_record is None:
                continue

            node = {
                'uid': flow_record['uid'],
                'ts': flow_record['ts'],
            }
            nodes.append(node)
        
        if not nodes or len(nodes) == 0:
            return []
        
        logger.debug("build_node_features(): nodes list construction is ok")

        # 构建节点特征
        for n in nodes:
            flow_record = self.get_flow_record(n['uid'])
            if flow_record is None:
                continue

            if (enabled_views.get("flow_numeric_features", False)
                or enabled_views.get("flow_categorical_features", False)
                or enabled_views.get("flow_textual_features", False)
            ):
                flow_numeric_features, flow_categorical_features, flow_textual_features = extract_conn_and_flowmeter_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

            if enabled_views.get("flow_numeric_features", False):
                n['flow_numeric_features'] = flow_numeric_features                   
                if numeric_feature_stats["flow_numeric_features"]["count"] > 1:
                    vec = n['flow_numeric_features']
                    # 标准化前检查NaN
                    if any(is_nan_or_inf(x) for x in vec):
                        # logger.warning(f"Flow {flow_uid} 标准化前发现NaN值")
                        vec = [0.0 if is_nan_or_inf(x) else x for x in vec]                    
                    mean = numeric_feature_stats["flow_numeric_features"]["mean"]
                    std  = numeric_feature_stats["flow_numeric_features"]["std"]
                    assert len(mean) == len(vec), f"Flow numeric feature length inconsistent: mean={len(mean)}, vec={len(vec)}, uid={flow_uid}"
                    assert len(std) == len(vec), f"Flow numeric feature length inconsistent: mean={len(std)}, vec={len(vec)}, uid={flow_uid}"
                    n['flow_numeric_features'] = [(x - m) / s for x, m, s in zip(vec, mean, std)]

                max_flow_numeric_len = self.get_global_node_feature_dims('flow_numeric_features')
                # 填充或裁剪 flow_numeric_features
                if len(n['flow_numeric_features']) < max_flow_numeric_len:
                    n['flow_numeric_features'] += [0.0] * (max_flow_numeric_len - len(n['flow_numeric_features']))
                else:
                    n['flow_numeric_features'] = n['flow_numeric_features'][:max_flow_numeric_len]

            if enabled_views.get("flow_categorical_features", False):
                n['flow_categorical_features'] = flow_categorical_features
                max_flow_categorical_len = self.get_global_node_feature_dims('flow_categorical_features')
                # 填充或裁剪 flow_categorical_features
                if len(n['flow_categorical_features']) < max_flow_categorical_len:
                    n['flow_categorical_features'] += [0] * (max_flow_categorical_len - len(n['flow_categorical_features']))
                else:
                    n['flow_categorical_features'] = n['flow_categorical_features'][:max_flow_categorical_len]

            if enabled_views.get("flow_textual_features", False):
                # ✅ 直接保存 dict，不做任何长度处理
                assert isinstance(flow_textual_features, dict)
                assert flow_textual_features["input_ids"].dim() == 2                
                n['flow_textual_features'] = flow_textual_features

            if (enabled_views.get("packet_len_seq", False)
                or enabled_views.get("packet_iat_seq", False)
            ):
                packet_len_seq, packet_iat_seq, packet_seq_mask = extract_flowmeter_packet_level_features(flow_record, mtu_normalize, max_packet_sequence_length)

            if enabled_views.get("packet_len_seq", False):
                n['packet_len_seq'] = packet_len_seq

            if enabled_views.get("packet_iat_seq", False):
                n['packet_iat_seq'] = packet_iat_seq

            n['packet_seq_mask'] = packet_seq_mask

            if enabled_views.get("domain_probs", False):
                domain_probs = extract_domain_name_probabilities(flow_record, num_classes)
                n['domain_probs'] = domain_probs
                max_domain_prob_len = self.get_global_node_feature_dims('domain_probs')
                # 填充或裁剪 domain_probs
                if len(n['domain_probs']) < max_domain_prob_len:
                    n['domain_probs'] += [0.0] * (max_domain_prob_len - len(n['domain_probs']))
                else:
                    n['domain_probs'] = n['domain_probs'][:max_domain_prob_len]

            if (enabled_views.get("ssl_numeric_features", False)
                or enabled_views.get("ssl_categorical_features", False)
                or enabled_views.get("ssl_textual_features", False)
            ):
                ssl_numeric_features, ssl_categorical_features, ssl_textual_features = extract_ssl_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

            if enabled_views.get("ssl_numeric_features", False):
                n['ssl_numeric_features'] = ssl_numeric_features
                if numeric_feature_stats["ssl_numeric_features"]["count"] > 1:
                    vec = n['ssl_numeric_features']
                    # 标准化前检查NaN
                    if any(is_nan_or_inf(x) for x in vec):
                        # logger.warning(f"Flow {flow_uid} 标准化前发现NaN值")
                        vec = [0.0 if is_nan_or_inf(x) else x for x in vec]                    
                    mean = numeric_feature_stats["ssl_numeric_features"]["mean"]
                    std  = numeric_feature_stats["ssl_numeric_features"]["std"]
                    assert len(mean) == len(vec), f"SSL numeric feature length inconsistent: mean={len(mean)}, vec={len(vec)}, uid={flow_uid}"
                    assert len(std) == len(vec), f"SSL numeric feature length inconsistent: mean={len(std)}, vec={len(vec)}, uid={flow_uid}"
                    n['ssl_numeric_features'] = [(x - m) / s for x, m, s in zip(vec, mean, std)]

                max_ssl_numeric_len = self.get_global_node_feature_dims('ssl_numeric_features')
                # 填充或裁剪 ssl_numeric_features
                if len(n['ssl_numeric_features']) < max_ssl_numeric_len:
                    n['ssl_numeric_features'] += [0.0] * (max_ssl_numeric_len - len(n['ssl_numeric_features']))
                else:
                    n['ssl_numeric_features'] = n['ssl_numeric_features'][:max_ssl_numeric_len]

            if enabled_views.get("ssl_categorical_features", False):
                n['ssl_categorical_features'] = ssl_categorical_features
                max_ssl_categorical_len = self.get_global_node_feature_dims('ssl_categorical_features')
                if len(n['ssl_categorical_features']) < max_ssl_categorical_len:
                    n['ssl_categorical_features'] += [0] * (max_ssl_categorical_len - len(n['ssl_categorical_features']))
                else:
                    n['ssl_categorical_features'] = n['ssl_categorical_features'][:max_ssl_categorical_len]

            if enabled_views.get("ssl_textual_features", False):
                # ✅ 直接保存 dict，不做任何长度处理
                assert isinstance(ssl_textual_features, dict)
                assert ssl_textual_features["input_ids"].dim() == 2                
                n['ssl_textual_features'] = ssl_textual_features

            if (enabled_views.get("x509_numeric_features", False)
                or enabled_views.get("x509_categorical_features", False)
                or enabled_views.get("x509_textual_features", False)
            ):
                x509_numeric_features, x509_categorical_features, x509_textual_features = extract_x509_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

            if enabled_views.get("x509_numeric_features", False):
                n['x509_numeric_features'] = x509_numeric_features
                if numeric_feature_stats["x509_numeric_features"]["count"] > 1:
                    vec = n['x509_numeric_features']
                    # 标准化前检查NaN
                    if any(is_nan_or_inf(x) for x in vec):
                        # logger.warning(f"Flow {flow_uid} 标准化前发现NaN值")
                        vec = [0.0 if is_nan_or_inf(x) else x for x in vec]                    
                    mean = numeric_feature_stats["x509_numeric_features"]["mean"]
                    std  = numeric_feature_stats["x509_numeric_features"]["std"]
                    assert len(mean) == len(vec), f"X509 numeric feature length inconsistent: mean={len(mean)}, vec={len(vec)}, uid={flow_uid}"
                    assert len(std) == len(vec), f"X509 numeric feature length inconsistent: mean={len(std)}, vec={len(vec)}, uid={flow_uid}"
                    n['x509_numeric_features'] = [(x - m) / s for x, m, s in zip(vec, mean, std)]

                max_x509_numeric_len = self.get_global_node_feature_dims('x509_numeric_features')
                # 填充或裁剪 x509_features
                if len(n['x509_numeric_features']) < max_x509_numeric_len:
                    n['x509_numeric_features'] += [0.0] * (max_x509_numeric_len - len(n['x509_numeric_features']))
                else:
                    n['x509_numeric_features'] = n['x509_numeric_features'][:max_x509_numeric_len]
                
            if enabled_views.get("x509_categorical_features", False):
                n['x509_categorical_features'] = x509_categorical_features
                max_x509_categorical_len = self.get_global_node_feature_dims('x509_categorical_features')
                if len(n['x509_categorical_features']) < max_x509_categorical_len:
                    n['x509_categorical_features'] += [0] * (max_x509_categorical_len - len(n['x509_categorical_features']))
                else:
                    n['x509_categorical_features'] = n['x509_categorical_features'][:max_x509_categorical_len]

            if enabled_views.get("x509_textual_features", False):
                # ✅ 直接保存 dict，不做任何长度处理
                assert isinstance(x509_textual_features, dict)
                assert x509_textual_features["input_ids"].dim() == 2
                n['x509_textual_features'] = x509_textual_features

            if (
                enabled_views.get("dns_numeric_features", False)
                or enabled_views.get("dns_categorical_features", False)
                or enabled_views.get("dns_textual_features", False)
            ):
                dns_numeric_features, dns_categorical_features, dns_textual_features = extract_dns_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

            if enabled_views.get("dns_numeric_features", False):
                n['dns_numeric_features'] = dns_numeric_features
                if numeric_feature_stats["dns_numeric_features"]["count"] > 1:
                    vec = n['dns_numeric_features']
                    # 标准化前检查NaN
                    if any(is_nan_or_inf(x) for x in vec):
                        # logger.warning(f"Flow {flow_uid} 标准化前发现NaN值")
                        vec = [0.0 if is_nan_or_inf(x) else x for x in vec]                    
                    mean = numeric_feature_stats["dns_numeric_features"]["mean"]
                    std  = numeric_feature_stats["dns_numeric_features"]["std"]
                    assert len(mean) == len(vec), f"DNS numeric feature length inconsistent: mean={len(mean)}, vec={len(vec)}, uid={flow_uid}"
                    assert len(std) == len(vec), f"DNS numeric feature length inconsistent: mean={len(std)}, vec={len(vec)}, uid={flow_uid}"
                    n['dns_numeric_features'] = [(x - m) / s for x, m, s in zip(vec, mean, std)]

                max_dns_numeric_len = self.get_global_node_feature_dims('dns_numeric_features')
                # 填充或裁剪 dns_features
                if len(n['dns_numeric_features']) < max_dns_numeric_len:
                    n['dns_numeric_features'] += [0.0] * (max_dns_numeric_len - len(n['dns_numeric_features']))
                else:
                    n['dns_numeric_features'] = n['dns_numeric_features'][:max_dns_numeric_len]
                
            if enabled_views.get("dns_categorical_features", False):
                n['dns_categorical_features'] = dns_categorical_features
                max_dns_categorical_len = self.get_global_node_feature_dims('dns_categorical_features')
                if len(n['dns_categorical_features']) < max_dns_categorical_len:
                    n['dns_categorical_features'] += [0] * (max_dns_categorical_len - len(n['dns_categorical_features']))
                else:
                    n['dns_categorical_features'] = n['dns_categorical_features'][:max_dns_categorical_len]

            if enabled_views.get("dns_textual_features", False):
                # ✅ 直接保存 dict，不做任何长度处理
                assert isinstance(dns_textual_features, dict)
                assert dns_textual_features["input_ids"].dim() == 2
                n['dns_textual_features'] = dns_textual_features

        logger.debug("build_node_features() ends: node feature extraction is ok, and max feature lengths are determined")

        return nodes


def parse_list_field(field_value):
    """终极修正版列表解析函数"""
    if field_value is None or pd.isna(field_value):
        return []
    
    if isinstance(field_value, (list, np.ndarray)):
        return list(field_value)
    
    if isinstance(field_value, str):
        value = field_value.strip()
        if not value or value.lower() in ['nan', 'none', 'null', '[]', '{}']:
            return []
        
        # 尝试自动修复不完整括号
        if value.count('[') != value.count(']'):
            # 情况1：缺少闭合括号
            if value.startswith('[') and not value.endswith(']'):
                value += ']'  # 尝试自动补全
            # 情况2：多余闭合括号
            elif not value.startswith('[') and value.endswith(']'):
                value = '[' + value
            # 其他情况保持原样
        
        # 解析优先级：JSON > Python字面量 > 逗号分隔
        for parser in [json.loads, ast.literal_eval]:
            try:
                parsed = parser(value)
                if isinstance(parsed, (list, tuple)):
                    return [int(x) if isinstance(x, float) and x.is_integer() else x 
                           for x in parsed]
                return [parsed]
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
        
        # 处理纯逗号分隔字符串（无括号）
        if ',' in value:
            parts = []
            for part in value.split(','):
                part = part.strip()
                if not part:
                    continue
                try:
                    num = float(part)
                    parts.append(int(num) if num.is_integer() else num)
                except ValueError:
                    parts.append(part)
            return parts
        
        return [value]
    
    return [field_value]


def normalize_packet_direction(d):
    # 返回 1 表示 客户端->服务端，返回 -1 表示 服务器->客户端
    if isinstance(d, (int, float, np.integer, np.floating)):
        try:
            v = int(d)
            return 1 if v == 1 else -1
        except:
            return 1
    if isinstance(d, str):
        ds = d.strip().lower()
        if ds in ('1', 'true', 't', 'c2s', 'client', '->', '>'):
            return 1
        if ds in ('0','false','f','s2c','server','<-','<'):
            return -1
        # 尝试识别 -1
        if ds.startswith('-'):
            return -1
        return 1
    if isinstance(d, bool):
        return 1 if d else -1
    return 1


def extract_flowmeter_packet_level_features(
    flow_record,
    mtu_normalize=1500,
    max_packet_sequence_length: int | None = None,
    pad_value: float = 0.0,
) -> Tuple[List[float], List[float], List[int]]:
    """
    提取 packet-level 特征：
      ✔ 方向增强 + MTU归一化 payload
      ✔ 方向增强 + 分段 log 缩放 IAT
      ❗ 序列长度不一致 → 直接抛异常，阻断图构建
    """
    flow_uid = flow_record['uid']
    packet_dir_vector = parse_list_field(
        flow_record['flowmeter.packet_direction_vector']
    )
    packet_len_vector = parse_list_field(
        flow_record['flowmeter.packet_payload_size_vector']
    )
    packet_ts_vector = parse_list_field(
        flow_record['flowmeter.packet_timestamp_vector']
    )
    raw_packet_iat_vector = parse_list_field(
        flow_record['flowmeter.packet_iat_vector']
    )
    # μs → s
    raw_packet_iat_vector = [float(x) / 1_000_000.0 for x in raw_packet_iat_vector]    

    # 后续逻辑都在保证长度大于0的前提下执行
    if len(packet_dir_vector) == 0:
        if max_packet_sequence_length is None:
            return [], [], []
        else:
            logger.debug(f"[EmptyPacketFlow] uid={flow_uid}")
            return (
                [pad_value] * max_packet_sequence_length,
                [pad_value] * max_packet_sequence_length,
                [0] * max_packet_sequence_length,
            )
            
    packet_iat_vector = [0.0] + raw_packet_iat_vector  # ⚠ IAT 前补一个0，第一个 packet 没有 IAT
    
    # 解析各bulk的长度序列
    bulk_lengths = parse_list_field(
        flow_record['flowmeter.bulk_length_vector']
    )

    # 解析批量数据包传输的索引序列
    bulk_packet_indices = parse_list_field(
        flow_record['flowmeter.bulk_packet_index_vector']
    )

    # -------- 直接强校验长度 --------
    # 后续逻辑都在保证长度一致前提下执行    
    if not (
        len(packet_dir_vector) ==
        len(packet_len_vector) ==
        len(packet_ts_vector) ==
        len(packet_iat_vector)
    ):
        raise ValueError(
            f"[SeqLenError] packet-level特征长度不一致:"
            f" packet_dir_vector={len(packet_dir_vector)},"            
            f" packet_len_vector={len(packet_len_vector)},"
            f" packet_ts_vector={len(packet_ts_vector)},"            
            f" packet_iat_vector={len(packet_iat_vector)},"
            f" uid={flow_record['uid']}"
        )

    # bulk_lengths 与 bulk_packet_indices 长度一致
    if sum(bulk_lengths) != len(bulk_packet_indices):
        logger.warning(f"[BulkInvalid] bulk info invalid, fallback to packet-level. flow_uid={flow_uid}")
        bulk_lengths = []
        bulk_packet_indices = []

    if any(int(idx) < 0 or int(idx) >= len(packet_dir_vector) for idx in bulk_packet_indices):
        logger.warning(f"[BulkInvalid] bulk index out of range, fallback to packet-level. flow_uid={flow_uid}")
        bulk_lengths = []
        bulk_packet_indices = []
    
    # 后续逻辑都在保证长度大于0的前提下执行
    # dir_vec_len = len(packet_dir_vector)
    # if dir_vec_len == 0:
    #     if max_packet_sequence_length is None:
    #         return [], [], []
    #     else:
    #         return (
    #             [pad_value] * max_packet_sequence_length,
    #             [pad_value] * max_packet_sequence_length,
    #             [0] * max_packet_sequence_length,
    #         )
        
    msg_seq_result = _extract_normalized_directed_msg_seq(flow_uid,
                                                    packet_dir_vector, 
                                                    packet_len_vector, 
                                                    packet_ts_vector,
                                                    bulk_lengths, 
                                                    bulk_packet_indices,
                                                    mtu_normalize,
                                                    max_packet_sequence_length,
                                                    pad_value)
    
    if msg_seq_result is not None:
        message_len_seq, message_iat_seq, message_seq_mask = msg_seq_result
        assert all(x >= 0 for x in message_iat_seq)
        return message_len_seq, message_iat_seq, message_seq_mask

    pkt_seq_result = _extract_normalized_directed_pkt_seq(packet_dir_vector, 
                                                        packet_len_vector, 
                                                        packet_iat_vector,
                                                        mtu_normalize,
                                                        max_packet_sequence_length,
                                                        pad_value)
    
    if pkt_seq_result is not None:
        packet_len_seq, packet_iat_seq, packet_seq_mask = pkt_seq_result
        assert all(x >= 0 for x in packet_iat_seq)
        return packet_len_seq, packet_iat_seq, packet_seq_mask


def _safe_log_scale_normalize(value: float, eps=1e-6, scale=1.0, signed=False) -> float:
    """
    Log-scale normalization to suppress long-tail effects.
    Applicable to:
    - time intervals (seconds)
    - payload sizes (bytes)
    No physical scale assumption (e.g., MTU).
    如果输入数值是时间，那么对以「秒」为单位的时间间隔进行稳健的 log 缩放，用于抑制超长时间间隔对模型训练的不稳定影响。
    如果输入数值是消息大小，那么对以「字节」为单位的消息大小进行稳健的 log 缩放，用于抑制超大消息载荷对模型训练的不稳定影响。
    """
    if value is None or abs(value) < eps:
        return 0.0
    
    if signed:
        sign = 1.0 if value > 0 else -1.0
        abs_val = abs(value) + eps

        return sign * np.log1p(abs_val / scale)
    else:
        value = max(value, 0.0) + eps
        return np.log1p(value / scale)
    
def _extract_normalized_directed_pkt_seq(packet_dir_vector, packet_len_vector, packet_iat_vector, 
                                         mtu_normalize, max_packet_sequence_length, pad_value):
    packet_len_seq = []
    packet_iat_seq = []

    for dir_vec, len_vec, iat_vec in zip(packet_dir_vector, packet_len_vector, packet_iat_vector):
        sign_vec = normalize_packet_direction(dir_vec)

        # Payload 归一化
        # norm_payload = float(len_vec) / float(mtu_normalize)
        norm_payload = _safe_log_scale_normalize(len_vec)
        # 不要CLIP，有的流量数据集可能会有超大的数据包载荷长度，因为不在网关侧采集，而是在端侧用tcpdump采集。
        # norm_payload = max(-1.0, min(1.0, norm_payload))  # clip
        # 数据包传输方向只作用在 payload / size 上。
        # 方向注入：
        # - payload / size 使用 ±1 表示通信方向
        # - iat / duration 保持为非负时间量
        packet_len_seq.append(sign_vec * norm_payload) 
        
        # IAT 缩放，注意👉 IAT 永远是非负量，不要乘方向
        scaled_iat_seq = _safe_log_scale_normalize(float(iat_vec))
        packet_iat_seq.append(scaled_iat_seq)

    # ===== truncate + pad（如果配置了 max_packet_sequence_length） =====
    orig_packet_sequence_length = len(packet_len_seq)

    if max_packet_sequence_length is not None:
        # truncate
        packet_len_seq = packet_len_seq[:max_packet_sequence_length]
        packet_iat_seq = packet_iat_seq[:max_packet_sequence_length]

        # pad
        if orig_packet_sequence_length < max_packet_sequence_length:
            pad_len = max_packet_sequence_length - orig_packet_sequence_length
            packet_len_seq.extend([pad_value] * pad_len)
            packet_iat_seq.extend([pad_value] * pad_len)

    # ✅ 构造 mask：真实位置为 1，padding 为 0
    if max_packet_sequence_length is None:
        # 不截断、不 padding
        packet_seq_mask = [1] * len(packet_len_seq)
    else:
        valid_len = min(orig_packet_sequence_length, max_packet_sequence_length)
        packet_seq_mask = [1] * valid_len
        if orig_packet_sequence_length < max_packet_sequence_length:
            packet_seq_mask.extend([0] * (max_packet_sequence_length - orig_packet_sequence_length))

    return packet_len_seq, packet_iat_seq, packet_seq_mask


def _extract_normalized_directed_msg_seq(flow_uid, packet_dir_vector, packet_len_vector, packet_ts_vector,
                                         bulk_lengths, bulk_packet_indices,
                                         mtu_normalize, max_packet_sequence_length, pad_value):
    '''
    把packet序列变换成single packet 或者 bulk 的信息序列
    '''
    pkt_to_bulk_idx = build_pkt_to_bulk_idx_map(bulk_lengths, bulk_packet_indices)

    message_dir_seq = [] 
    message_len_seq = []
    message_iat_seq = []

    prev_msg_timestamp = None
    current_bulk_direction = None
    current_bulk_bytes = []
    current_bulk_timestamps = []
    current_bulk_idx = -1
    
    for pkt_idx in range(len(packet_len_vector)):
        pkt_direction = packet_dir_vector[pkt_idx] if pkt_idx < len(packet_dir_vector) else 0
        pkt_payload_size = packet_len_vector[pkt_idx] if pkt_idx < len(packet_len_vector) else 0            
        pkt_timestamp = packet_ts_vector[pkt_idx] if pkt_idx < len(packet_ts_vector) else 0
        
        # if pkt_payload_size == 0:
        #     # 这里可以考虑忽略零payload bytes的载荷包，因为：
        #     # 零载荷包（如 ACK / 控制包）不作为 message，
        #     # 避免高频控制包干扰 message-level 行为建模。
        #     # 也可以选择将其作为单包消息处理。
        #     continue

        if pkt_payload_size == 0:
            # Flowmeter插件中，零载荷包不参与 bulk 传输的划分
            # 因此，直接将其视为不属于任何 bulk，但作为单包消息处理
            bulk_idx_of_current_pkt = None  
        else:
            bulk_idx_of_current_pkt = pkt_to_bulk_idx.get(pkt_idx, None)

        if bulk_idx_of_current_pkt is None or bulk_idx_of_current_pkt != current_bulk_idx:
            # 如果当前包不属于任何 bulk 或者其所属于的bulk_idx不同于当前bulk_idx，那么先结束当前的 bulk（如果有）
            if current_bulk_idx >= 0:
                # bulk 发生切换（或进入 / 离开 bulk）
                # 创建一个 multiple-packet message entry
                prev_msg_timestamp = _create_multiple_pkt_msg_return_msg_timestamp(
                                        current_bulk_direction, current_bulk_bytes, current_bulk_timestamps, prev_msg_timestamp, 
                                        message_dir_seq, message_len_seq, message_iat_seq, mtu_normalize)
                # 通过重置 current bulk info，结束当前的 bulk
                current_bulk_direction = None
                current_bulk_bytes = []
                current_bulk_timestamps = []
                current_bulk_idx = -1

        if bulk_idx_of_current_pkt is None: 
            # Zeek Flowmeter插件创建一个Bulk时，要求bulk内至少包含5个数据包。查看https://gitee.com/seu-csqjxiao/zeek-flowmeter
            # FlowMeter::bulk_min_length: The minimal number of data packets which have to be in 
            #                             a bulk transmission for it to be considered a bulk transmission. 
            #                             The default value is 5 packets.
            # FlowMeter::bulk_timeout: The maximal allowed inter-arrival time between two data packets 
            #                          so they are considered to be part of the same bulk transmission. 
            #                          The default value is 1s.
            # 因此，如果当前包不属于任何 bulk，则创建单包消息。
            prev_msg_timestamp = _create_single_pkt_msg_return_msg_timestamp(
                                    pkt_direction, pkt_timestamp, pkt_payload_size, prev_msg_timestamp, 
                                    message_dir_seq, message_len_seq, message_iat_seq, mtu_normalize)
        else:
            if current_bulk_idx == -1:
                # 开始一个新的 bulk，其方向由第一个包决定，该bulk后续的包必须方向一致
                current_bulk_idx = bulk_idx_of_current_pkt
                current_bulk_direction = pkt_direction

            assert pkt_direction == current_bulk_direction, \
                f"检测到 bulk 内方向不一致，flow_uid={flow_uid}, bulk_idx={current_bulk_idx}"
            current_bulk_bytes.append(pkt_payload_size)
            current_bulk_timestamps.append(pkt_timestamp)

    
    if current_bulk_direction is not None:
        # 已经扫描完数据包序列。如果当前bulk还没有结束，创建一个 multiple-packet message
        prev_msg_timestamp = _create_multiple_pkt_msg_return_msg_timestamp(
                                    current_bulk_direction, current_bulk_bytes, current_bulk_timestamps, prev_msg_timestamp, 
                                    message_dir_seq, message_len_seq, message_iat_seq, mtu_normalize)
        # 通过重置 current bulk info，结束当前的 bulk
        current_bulk_direction = None
        current_bulk_bytes = []
        current_bulk_timestamps = []
        current_bulk_idx = -1

    # 方向注入：
    # - payload / size 使用 ±1 表示通信方向
    # - iat / duration 保持为非负时间量
    # ===== 方向注入：统一使用 ±1，与 packet-level 语义一致 =====
    for i in range(len(message_dir_seq)):
        # 数据包传输方向只作用在 payload / size 上。
        message_len_seq[i] *= message_dir_seq[i]
        # 👉 IAT 永远是非负量，不要乘方向
        # message_iat_seq[i] *= message_dir_seq[i]

    # -------------------------------
    # 统一截断到 max_seq_length
    # -------------------------------
    orig_message_seq_len = len(message_dir_seq)
    # message 的 original_length 可能为 0，如果：
    # 全是 payload_size == 0 的包，或者 bulk 被忽略
    # 这种情况下，返回 None，表示没有有效的消息序列，直接回退 packet-level sequence 处理
    if orig_message_seq_len == 0:
        return None 

    message_seq_mask = [1] * orig_message_seq_len

    if max_packet_sequence_length is not None:
        if orig_message_seq_len > max_packet_sequence_length:
            valid_message_seq_len = max_packet_sequence_length

            message_len_seq = message_len_seq[:max_packet_sequence_length]
            message_iat_seq = message_iat_seq[:max_packet_sequence_length]
            message_seq_mask = [1] * max_packet_sequence_length

        else:
            valid_message_seq_len = orig_message_seq_len
            pad_len = max_packet_sequence_length - valid_message_seq_len
            if pad_len > 0:
                message_len_seq.extend([pad_value] * pad_len)      
                message_iat_seq.extend([pad_value] * pad_len)
                message_seq_mask.extend([0] * pad_len)

    return message_len_seq, message_iat_seq, message_seq_mask

def build_pkt_to_bulk_idx_map(
    bulk_lengths,
    bulk_packet_indices,
):
    """
    构建 packet_index -> bulk_idx 的映射字典。

    说明：
    - bulk_length_vector 记录的是 data_size > 0 的数据包数量，
    packet index 在 flow 中可能是不连续的；
    - bulk_packet_indices 已按 bulk 顺序拼接，仅包含 data_size > 0 的包；
    - 因此，通过 bulk_length_vector 对 bulk_packet_indices 顺序切分，
    可以准确恢复每个 bulk 内的 packet index 集合。
    """
    pkt_to_bulk_idx = {}

    offset = 0
    for bulk_idx, bulk_len in enumerate(bulk_lengths):
        # 取属于该 bulk 的 data packets（不要求 index 连续）
        bulk_pkts = bulk_packet_indices[offset : offset + bulk_len]

        for pkt_idx in bulk_pkts:
            pkt_to_bulk_idx[pkt_idx] = bulk_idx

        offset += bulk_len

    return pkt_to_bulk_idx

def _parse_direction_value(direction_val):
        """统一解析方向值"""
        if direction_val is None:
            return 1
            
        if isinstance(direction_val, str):
            direction_str = str(direction_val).lower().strip()
            if direction_str in ['true', '1', 'forward', 'fwd']:
                return 1
            elif direction_str in ['false', '0', 'backward', 'bwd']:
                return -1
            else:
                # 尝试数值转换
                try:
                    num_val = float(direction_val)
                    return 1 if num_val > 0 else -1
                except:
                    return 1  # 默认值
        
        elif isinstance(direction_val, bool):
            return 1 if direction_val else -1
        
        elif isinstance(direction_val, (int, float)):
            return 1 if direction_val > 0 else -1
        
        else:
            return 1  # 默认值

def _create_single_pkt_msg_return_msg_timestamp(pkt_direction, pkt_timestamp, pkt_payload_size, prev_msg_timestamp, 
                                                message_dir_seq, message_len_seq, message_iat_seq, mtu_normalize):
    msg_direction_value = _parse_direction_value(pkt_direction)
    message_dir_seq.append(msg_direction_value)

    # msg_total_payload_size = pkt_payload_size / float(mtu_normalize)  # 归一化到0-1
    msg_total_payload_size = _safe_log_scale_normalize(pkt_payload_size)
    message_len_seq.append(msg_total_payload_size)

    # Zeek Flowmeter插件里面，pkt_timestamp的时间单位是秒，无需转换。
    # packet_timestamp_vector 以Unix时间戳格式记录flow中各数据包的到达时间。
    # 查看 https://gitee.com/seu-csqjxiao/zeek-flowmeter/tree/seu-devel
    if prev_msg_timestamp is None:
        # 第一个消息，时间间隔为0
        msg_iat_time = 0.0
    else:
        # message 到达时间定义为该消息中第一个 packet 的时间戳
        msg_iat_time = pkt_timestamp - prev_msg_timestamp
        msg_iat_time = _safe_log_scale_normalize(msg_iat_time)
    
    message_iat_seq.append(msg_iat_time)

    # message_pkt_num_seq.append(1)  # 单包消息，包数为1

    # message_avg_payload_size_seq.append(msg_total_payload_size)  # 单包消息，平均载荷大小等于载荷大小

    # message_duration_seq.append(0)  # 单包消息，持续时间为0秒

    return pkt_timestamp
    
def _create_multiple_pkt_msg_return_msg_timestamp(current_bulk_direction, current_bulk_bytes, current_bulk_timestamps, prev_msg_timestamp, 
                                                    message_dir_seq, message_len_seq, message_iat_seq, mtu_normalize):
    msg_direction_value = _parse_direction_value(current_bulk_direction)
    message_dir_seq.append(msg_direction_value)

    # msg_total_payload_size = sum(current_bulk_bytes) / float(mtu_normalize)  # 归一化到0-1
    msg_total_payload_size = _safe_log_scale_normalize(sum(current_bulk_bytes))
    message_len_seq.append(msg_total_payload_size)

    # Zeek Flowmeter插件里面，pkt_timestamp的时间单位是秒，无需转换。
    # packet_timestamp_vector 以Unix时间戳格式记录flow中各数据包的到达时间。
    # 查看 https://gitee.com/seu-csqjxiao/zeek-flowmeter/tree/seu-devel
    if prev_msg_timestamp is None:
        # 第一个消息，时间间隔为0
        msg_iat_time = 0.0
    else:
        # message 的到达时间定义为 bulk 中第一个数据包的时间戳
        msg_iat_time = current_bulk_timestamps[0] - prev_msg_timestamp
        msg_iat_time = _safe_log_scale_normalize(msg_iat_time)
    
    message_iat_seq.append(msg_iat_time)

    # message_pkt_num_seq.append(len(current_bulk_bytes))  # 多包消息，包数为 bulk 内包数

    # # 多包消息，平均载荷大小
    # avg_payload_size = sum(current_bulk_bytes) / len(current_bulk_bytes) / float(mtu_normalize)
    # # avg_payload_size = _safe_log_scale_normalize(sum(current_bulk_bytes) / len(current_bulk_bytes))
    # message_avg_payload_size_seq.append(avg_payload_size)  

    # # 多包消息，持续时间设定为 bulk 内最后一个包的时间戳 - 第一个包的时间戳，时间单位是秒
    # message_duration = current_bulk_timestamps[-1] - current_bulk_timestamps[0]
    # message_duration_seq.append(message_duration)

    # 返回该消息的时间戳（即 bulk 中第一个数据包的时间戳）
    return current_bulk_timestamps[0] 

def extract_domain_name_probabilities(flow_record, num_classes, num_domain_name_hierarchy_levels = 5):
    """从DNS和TLS域名嵌入特征中提取多层级嵌入向量，严格校验维度"""
    domain_probs = []    
    for level in range(num_domain_name_hierarchy_levels): # 默认层级数量：0~4
        for proto in ['ssl', 'dns']:
            if proto == 'ssl':
                embed_col = f'{proto}.server_name{level}_freq'
            elif proto == 'dns':
                embed_col = f'{proto}.query{level}_freq'
            else:
                raise ValueError(f"extract_domain_name_probabilities(): unsupported protocol or domain name hierarchical level.")
            
            embed_value = flow_record[embed_col]

            # 🔹如果列不存在或值为空 → 填充全零
            if embed_value is None or pd.isna(embed_value) or embed_value == "":
                domain_probs.extend([0.0] * num_classes)
                continue

            try:
                # 🔹确保转换为 Python list 或 numpy array
                if isinstance(embed_value, str):
                    embed_vector = ast.literal_eval(embed_value)
                elif isinstance(embed_value, (list, np.ndarray)):
                    embed_vector = list(embed_value)
                else:
                    raise TypeError(f"Unsupported type for {embed_col}: {type(embed_value)}")

                # 🔹确保是可迭代的浮点向量
                embed_vector = [float(x) for x in embed_vector]

                # 严格维度校验
                if len(embed_vector) != num_classes:
                    raise ValueError(
                        f"extract_domain_name_probabilities(): [DimError] {embed_col} 维度错误: "
                        f"expected={num_classes}, got={len(embed_vector)}, value={embed_vector}"
                    )

                domain_probs.extend(embed_vector)

            except Exception as e:
                # ❌任何解析失败 → 抛出明确异常，用于定位数据问题
                raise ValueError(
                    f"[ParseError] 域名嵌入解析失败: {embed_col}={embed_value}, error={e}"
                )

    return domain_probs


def to_str_safe(val):
    """把 val 安全地变为 str 并 strip；对于 None / NaN 返回空字符串。"""
    if val is None:
        return ''
    try:
        # pandas 的 NaN / None 检测
        if pd.isna(val):
            return ''
    except Exception:
        pass
    if isinstance(val, str):
        return val.strip()
    # 其他类型（float/int/list/..）都转成字符串并 strip
    return str(val).strip()

def encode_text(text: str, text_tokenizer, max_text_length):
    """
    将任意 textual 字段编码为长度固定 max_text_length 的 token 序列（LongTensor）
    """
    if not isinstance(text, str):
        text = ""  # 非字符串统一处理为空字符串

    encoded = text_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )

    # ✅ Sanity check（强烈推荐）
    assert encoded["input_ids"].dim() == 2, \
        f"encode_text expects input_ids to be 2D [1, L], got shape {encoded['input_ids'].shape}"
    assert encoded["attention_mask"].dim() == 2, \
        f"encode_text expects attention_mask to be 2D [1, L], got shape {encoded['attention_mask'].shape}"

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

def extract_conn_and_flowmeter_features(flow_record, categorical_vocab_group, text_tokenizer, max_text_length):
    """从flowmeter记录中提取统计特征"""
    numeric = []
    categorical = []
    textual_fields = []

    # ---------- 数值特征 ----------
    for col in conn_numeric_columns:
        full = f"conn.{col}"
        value = flow_record[full]
        try:
            numeric.append(float(value) if value not in (None, "") else 0.0)
        except:
            logger.error(
                f"[DEBUG] BAD NUMERIC in conn_numeric_columns: "
                f"col={col}, value='{value}'"
            )            
            numeric.append(0.0)

    for col in flowmeter_numeric_columns:
        full = f"flowmeter.{col}"
        value = flow_record[full]
        try:
            numeric.append(float(value) if value not in (None, "") else 0.0)
        except:
            logger.error(
                f"[DEBUG] BAD NUMERIC in flowmeter_numeric_columns: "
                f"col={col}, value='{value}'"
            )            
            numeric.append(0.0)

    # ---------- 类别特征 ----------
    for col in conn_categorical_columns:
        full = f"conn.{col}"
        value = flow_record[full]
        vocab = categorical_vocab_group.get(full)
        if vocab is None:
            categorical.append(0)
        else:
            categorical.append(vocab.get(value, 0))        

    for col in flowmeter_categorical_columns:
        full = f"flowmeter.{col}"
        value = flow_record[full]
        vocab = categorical_vocab_group.get(full)
        if vocab is None:
            categorical.append(0)
        else:
            categorical.append(vocab.get(value, 0))

    # ---------- 文本特征 ----------
    for col in conn_textual_columns:
        full = f"conn.{col}"
        value = flow_record[full]
        textual_fields.append(to_str_safe(value))

    for col in flowmeter_textual_columns:
        full = f"flowmeter.{col}"
        value = flow_record[full]
        textual_fields.append(to_str_safe(value))

    # 合并成一个字符串，可选，也可分多字段编码
    combined_text = " ".join([str(x) for x in textual_fields if isinstance(x, str)])
    encoded_text = encode_text(combined_text, text_tokenizer, max_text_length)

    return numeric, categorical, encoded_text

def extract_ssl_features(flow_record, categorical_vocab_group, text_tokenizer, max_text_length):
    """提取 SSL 的 numeric / categorical / textual 特征（严格使用 zeek_columns 定义）"""
    numeric = []
    categorical = []
    textual_fields = []

    # ---------- 数值特征 ----------
    for col in ssl_numeric_columns:
        full = f"ssl.{col}"
        value = flow_record[full]
        try:
            numeric.append(float(value) if value not in (None, "") else 0.0)
        except:
            logger.error(
                f"[DEBUG] BAD NUMERIC in ssl_numeric_columns: "
                f"col={col}, value='{value}'"
            )            
            numeric.append(0.0)

    # ---------- 类别特征 ----------
    for col in ssl_categorical_columns:
        full = f"ssl.{col}"
        value = flow_record[full]
        vocab = categorical_vocab_group.get(full)
        if vocab is None:
            categorical.append(0)
        else:
            categorical.append(vocab.get(value, 0))

    # ---------- 文本特征 ----------
    for col in ssl_textual_columns:
        full = f"ssl.{col}"
        value = flow_record[full]
        textual_fields.append(to_str_safe(value))

    # 合并成一个字符串，可选，也可分多字段编码
    combined_text = " ".join([str(x) for x in textual_fields if isinstance(x, str)])
    encoded_text = encode_text(combined_text, text_tokenizer, max_text_length)

    return numeric, categorical, encoded_text

def extract_x509_features(flow_record, categorical_vocab_group, text_tokenizer, max_text_length):
    numeric = []
    categorical = []
    textual_fields = []

    for idx in range(max_x509_cert_chain_len):
        prefix = f"x509.cert{idx}"

        # 如果该证书不存在 → 填零占位（保持对齐）
        exists = any(k.startswith(prefix) for k in flow_record.keys())

        # ---------- numeric ----------
        for col in x509_numeric_columns:
            full = f"{prefix}.{col}"
            value = flow_record[full]
            if not exists:
                numeric.append(0.0)
                continue
            try:
                numeric.append(float(value) if value not in (None, "") else 0.0)
            except:
                numeric.append(0.0)

        # ---------- categorical ----------
        for col in x509_categorical_columns:
            full = f"{prefix}.{col}"
            value = flow_record[full]
            vocab = categorical_vocab_group.get(full)
            if vocab is None:
                categorical.append(0)
            else:
                categorical.append(vocab.get(value, 0))

        # ---------- textual ----------
        for col in x509_textual_columns:
            full = f"{prefix}.{col}"
            value = flow_record[full]
            textual_fields.append(to_str_safe(value) if exists else "")

    # 合并成一个字符串，可选，也可分多字段编码
    combined_text = " ".join([str(x) for x in textual_fields if isinstance(x, str)])
    encoded_text = encode_text(combined_text, text_tokenizer, max_text_length)

    return numeric, categorical, encoded_text


def extract_dns_features(flow_record, categorical_vocab_group, text_tokenizer, max_text_length):
    numeric = []
    categorical = []
    textual_fields = []

    # ---------- numeric ----------
    for col in dns_numeric_columns:
        full = f"dns.{col}"
        value = flow_record[full]
        try:
            numeric.append(float(value) if value not in (None, "") else 0.0)
        except:
            logger.error(
                f"[DEBUG] BAD NUMERIC in dns_numeric_columns: "
                f"col={col}, value='{value}'"
            )            
            numeric.append(0.0)

    # ---------- categorical ----------
    for col in dns_categorical_columns:
        full = f"dns.{col}"
        value = flow_record[full]
        vocab = categorical_vocab_group.get(full)
        if vocab is None:
            categorical.append(0)
        else:
            categorical.append(vocab.get(value, 0))

    # ---------- textual ----------
    for col in dns_textual_columns:
        full = f"dns.{col}"
        value = flow_record[full]
        textual_fields.append(to_str_safe(value))

    # 合并成一个字符串，可选，也可分多字段编码
    combined_text = " ".join([str(x) for x in textual_fields if isinstance(x, str)])
    encoded_text = encode_text(combined_text, text_tokenizer, max_text_length)

    return numeric, categorical, encoded_text


def get_project_root(start_path: str = None):
    import os, subprocess

    if start_path is None:
        start_path = os.path.abspath(os.path.dirname(__file__))

    # ① 尝试通过 Git
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
        return root
    except Exception:
        pass

    # ② 尝试查找关键文件
    markers = ("pyproject.toml", "setup.py", "requirements.txt", ".git")
    cur = start_path
    while True:
        if any(os.path.exists(os.path.join(cur, m)) for m in markers):
            return cur
        parent = os.path.abspath(os.path.join(cur, os.pardir))
        if parent == cur:
            break
        cur = parent

    # ③ fallback：使用 VSCode 工作路径
    return os.environ.get("PWD", os.getcwd())

def load_text_tokenizer(model_name="bert-base-uncased", max_text_length=64):
    """
    加载 BERT tokenizer。
    支持：
      ✔ 先尝试从本地 models_hub 求解
      ✔ 找不到则自动在线加载
    返回：
      tokenizer（BertTokenizer）
      max_text_len（int）
    """
    project_root = get_project_root()
    model_path = os.path.join(project_root, 'models_hub', model_name)

    try:
        logger.info(f"尝试从本地缓存加载 BERT tokenizer: {model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_path)
    except Exception as e:
        logger.warning(f"本地 tokenizer 不存在: {e}")
        logger.warning("尝试从 HuggingFace 在线下载...")
        tokenizer = BertTokenizer.from_pretrained(model_name)

    logger.info(f"BERT tokenizer 加载成功，max_text_len={max_text_length}")
    return tokenizer, max_text_length

def read_large_csv_with_progress(filepath, description="读取数据到pandas dataframe", verbose=True):
    """带进度条的大型CSV文件读取函数"""
    if verbose:
        logger.info(f"{description}，从路径 {filepath}...")
        file_size = os.path.getsize(filepath) / (1024 * 1024 * 1024)  # GB
        logger.info(f"文件大小: {file_size:.2f}GB")
    
    # 先读取前几行获取列信息
    sample_df = pd.read_csv(filepath, nrows=5)
    columns = sample_df.columns.tolist()
    
    # 分块读取
    chunks = []
    chunk_size = 100000  # 每次读取10万行

    if verbose:
        logger.info(f"检测到 {len(columns)} 列，开始每{chunk_size}行分块读取...")
    
    # 获取总行数（不读取全部内容）
    with open(filepath, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # 减去标题行
    
    if verbose:
        # 使用position=0确保进度条在同一行更新
        pbar = tqdm.tqdm(total=total_rows, desc=description, position=0, leave=True)
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
        if verbose:
            pbar.update(len(chunk))
    
    if verbose:
        pbar.close()
    
    # 合并所有块
    df = pd.concat(chunks, ignore_index=True)
    
    if verbose:
        logger.info(f"{description}完成! 数据形状: {df.shape}")
    
    return df

# def main():
#     """测试函数：验证flow和session数据读取及特征提取，并计算全局特征维度"""
#     # 测试数据路径
#     flow_csv_path = "processed_data/CIC-AndMal2017/SMSMalware/jifake-flow.csv"
#     session_csv_path = "processed_data/CIC-AndMal2017/SMSMalware/jifake-session.csv"
    
#     try:
#         # 1. 读取flow数据并构建flow_dict（uid到flow记录的映射）
#         logger.info(f"读取flow数据: {flow_csv_path}")
#         flow_df = pd.read_csv(
#             flow_csv_path,
#             dtype=dtype_dict_in_flow_csv,
#             parse_dates=False  # 避免自动解析日期导致格式问题
#         )
#         flow_dict = {row['uid']: row.to_dict() for _, row in flow_df.iterrows()}
#         logger.info(f"成功加载 {len(flow_dict)} 条flow记录")

#         # 2. 读取session数据
#         logger.info(f"读取session数据: {session_csv_path}")
#         session_df = pd.read_csv(session_csv_path)
#         logger.info(f"成功加载 {len(session_df)} 条session记录")

#         # 3. 初始化全局特征维度统计
#         global_node_feature_dims = {
#             "flow_numeric_features": 0,
#             "flow_categorical_features": 0,
#             "flow_textual_features": 0,
#             "packet_len_seq": 0,
#             "packet_iat_seq": 0,
#             "domain_probs": 0,
#             "ssl_numeric_features": 0,
#             "ssl_categorical_features": 0,
#             "ssl_textual_features": 0,
#             "x509_numeric_features": 0,
#             "x509_categorical_features": 0,
#             "x509_textual_features": 0,
#             "dns_numeric_features": 0,
#             "dns_categorical_features": 0,
#             "dns_textual_features": 0,
#         }

#         # 获取默认的类别数量（用于测试）
#         num_classes = len(set(read_session_label_id_map().values()))
#         logger.info(f"类别数量: {num_classes}")

#         categorical_vocabulary_group = FlowNodeBuilder.scan_flow_dict_for_categorical_topk_vocab_group(flow_dict)

#         text_encoder_name, max_text_length = read_text_encoder_config()
#         text_tokenizer, max_text_length = load_text_tokenizer(
#             model_name=text_encoder_name,
#             max_text_length=max_text_length
#         )
            
#         # 4. 遍历所有会话和流，计算全局特征维度
#         logger.info("计算全局特征维度...")
#         for _, session_row in tqdm.tqdm(session_df.iterrows(), total=len(session_df), desc="处理会话"):
#             # 解析session中的flow列表
#             if 'flow_uid_list' not in session_row:
#                 continue

#             flow_uid_list = ast.literal_eval(session_row['flow_uid_list'])

#             # 遍历每个flow
#             for flow_uid in flow_uid_list:
#                 if flow_uid not in flow_dict:
#                     continue

#                 flow_record = flow_dict[flow_uid]

#                 try:
#                     # 提取加密流量基本特征
#                     flow_numeric_features, flow_categorical_features, flow_textual_features = extract_conn_and_flowmeter_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)
#                     mtu_normalize = 1500    # bytes
#                     max_packet_sequence_length = 512    # tokens
#                     packet_len_seq, packet_iat_seq, packet_seq_mask = extract_flowmeter_packet_level_features(flow_record, mtu_normalize, max_packet_sequence_length)
#                     logger.debug("!!! flow_uid = {flow_uid}, with flow_numeric_features = " + str(flow_numeric_features)
#                                  + ", flow_categorical_features = " + str(flow_categorical_features)
#                                  + ", flow_textual_features = " + str(flow_textual_features)
#                                  + ", packet_len_seq = " + str(packet_len_seq)
#                                  + ", packet_iat_seq = " + str(packet_iat_seq)
#                                  + ", packet_seq_mask = " + str(packet_seq_mask)
#                             )

#                     # 提取明文部分的载荷特征
#                     domain_probs = extract_domain_name_probabilities(flow_record, num_classes)
#                     ssl_numeric_features, ssl_categorical_features, ssl_textual_features = extract_ssl_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)
#                     x509_numeric_features, x509_categorical_features, x509_textual_features = extract_x509_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)
#                     dns_numeric_features, dns_categorical_features, dns_textual_features = extract_dns_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)
#                     logger.debug("!!! flow_uid = {flow_uid}, with domain_probs = " + str(domain_probs) 
#                                  + ", ssl_numeric_features = " + str(ssl_numeric_features) 
#                                  + ", ssl_categorical_features = " + str(ssl_categorical_features)
#                                  + ", ssl_textual_features = " + str(ssl_textual_features)
#                                  + ", x509_numeric_features = " + str(x509_numeric_features) 
#                                  + ", x509_categorical_features = " + str(x509_categorical_features)
#                                  + ", x509_textual_features = " + str(x509_textual_features)
#                                  + ", dns_numeric_features = " + str(dns_numeric_features) 
#                                  + ", dns_categorical_features = " + str(dns_categorical_features)
#                                  + ", dns_textual_features = " + str(dns_textual_features)
#                             )

#                     # 更新全局维度统计
#                     global_node_feature_dims['flow_numeric_features'] = max(
#                         global_node_feature_dims['flow_numeric_features'], 
#                         len(flow_numeric_features) if len(flow_numeric_features) > 0 else 1
#                     )
#                     global_node_feature_dims['flow_categorical_features'] = max(
#                         global_node_feature_dims['flow_categorical_features'], 
#                         len(flow_categorical_features) if len(flow_categorical_features) > 0 else 1
#                     )
#                     global_node_feature_dims['flow_textual_features'] = max(
#                         global_node_feature_dims['flow_textual_features'], 
#                         len(flow_textual_features) if len(flow_textual_features) > 0 else 1
#                     )                    
#                     global_node_feature_dims['packet_len_seq'] = max_packet_sequence_length
#                     global_node_feature_dims['packet_iat_seq'] = max_packet_sequence_length
#                     global_node_feature_dims['domain_probs'] = max(
#                         global_node_feature_dims.get('domain_probs', 0), 
#                         len(domain_probs) if len(domain_probs) > 0 else 1
#                     )

#                     # 计算SSL的各类特征向量的维度
#                     global_node_feature_dims['ssl_numeric_features'] = max(
#                         global_node_feature_dims['ssl_numeric_features'], 
#                         len(ssl_numeric_features) if len(ssl_numeric_features) > 0 else 1)
#                     global_node_feature_dims['ssl_categorical_features'] = max(
#                         global_node_feature_dims['ssl_categorical_features'], 
#                         len(ssl_categorical_features) if len(ssl_categorical_features) > 0 else 1)
#                     global_node_feature_dims['ssl_textual_features'] = max(
#                         global_node_feature_dims['ssl_textual_features'], 
#                         len(ssl_textual_features) if len(ssl_textual_features) > 0 else 1)

#                     # 计算X509的各类特征向量的维度
#                     global_node_feature_dims['x509_numeric_features'] = max(
#                         global_node_feature_dims['x509_numeric_features'], 
#                         len(x509_numeric_features) if len(x509_numeric_features) > 0 else 1)
#                     global_node_feature_dims['x509_categorical_features'] = max(
#                         global_node_feature_dims['x509_categorical_features'], 
#                         len(x509_categorical_features) if len(x509_categorical_features) > 0 else 1)
#                     global_node_feature_dims['x509_textual_features'] = max(
#                         global_node_feature_dims['x509_textual_features'], 
#                         len(x509_textual_features) if len(x509_textual_features) > 0 else 1)

#                     # 计算DNS的各类特征向量的维度                
#                     global_node_feature_dims['dns_numeric_features'] = max(
#                         global_node_feature_dims['dns_numeric_features'], 
#                         len(dns_numeric_features) if len(dns_numeric_features) > 0 else 1)
#                     global_node_feature_dims['dns_categorical_features'] = max(
#                         global_node_feature_dims['dns_categorical_features'], 
#                         len(dns_categorical_features) if len(dns_categorical_features) > 0 else 1)
#                     global_node_feature_dims['dns_textual_features'] = max(
#                         global_node_feature_dims['dns_textual_features'], 
#                         len(dns_textual_features) if len(dns_textual_features) > 0 else 1)

#                 except Exception as e:
#                     logger.error(f"Flow {flow_uid} 特征提取错误: {str(e)}")
#                     continue

#         # 5. 确保最小维度为1
#         for key in global_node_feature_dims:
#             global_node_feature_dims[key] = max(1, global_node_feature_dims[key])

#         # 6. 输出全局特征维度
#         logger.info("全局特征维度统计:")
#         for key, dim in global_node_feature_dims.items():
#             logger.info(f"  {key}: {dim}")

#         logger.info("测试完成")

#     except FileNotFoundError as e:
#         logger.error(f"错误: 未找到测试文件 - {str(e)}")
#     except KeyError as e:
#         logger.error(f"错误: 数据中缺少必要字段 - {str(e)}")
#     except Exception as e:
#         logger.error(f"测试过程出错: {str(e)}")


# if __name__ == "__main__":
#     main()
    