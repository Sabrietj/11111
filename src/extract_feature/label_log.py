# -*- coding: utf-8 -*-
import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import csv
import re
from datetime import datetime
from collections import Counter

verbose = False

# IANA 协议号映射表
PROTO_STR2NUM = {
    "icmp": 1,
    "tcp": 6,
    "udp": 17,
    "gre": 47,
    "esp": 50,
    "ah": 51,
    # 可根据需要补充
}

def normalize_proto(proto):
    """
    将 proto 转为统一字符串数字形式
    支持字符串协议名或数字协议
    """
    if isinstance(proto, str):
        proto = proto.strip().lower()
        # 如果是数字字符串，直接返回
        if proto.isdigit():
            return proto
        # 如果是协议名，转换为数字字符串
        return str(PROTO_STR2NUM.get(proto, proto))
    else:
        return str(proto)

    
def find_flow_label_file_names(path_to_folder):
    '''
    binetflow 文件是一种 基于 NetFlow/IPFIX 记录的流量表示格式，常见于 CTU-13 恶意流量数据
    集（以及后续的一些恶意流量数据集，如 CTU-Malware-Capture 系列）。它本质上是一个 CSV 文件，
    包含流量的五元组信息及其统计特征，并且附带了标签（标记是否是恶意流量）。    
    '''
    print("<< Searching for flow label files in ", path_to_folder, " >>\n")
    # 搜索 binetflow 和 cicflowmeter生成的csv 文件
    candidate_files = glob.glob(os.path.join(path_to_folder, "*.binetflow")) \
                      + glob.glob(os.path.join(path_to_folder, "*.csv"))
    print(f"Found {len(candidate_files)} candidate flow label files: " + str(candidate_files))

    # 仅保留 Label/label 列的文件
    valid_files = []
    for file in candidate_files:
        try:
            # 只读取表头，不加载全部数据
            if check_flow_label_file_contain_label(file):
                valid_files.append(file)
        except Exception as e:
            # 文件读取失败则跳过
            continue

    print(f"Found {len(valid_files)} valid flow label files:" + str(valid_files))
    return valid_files


def detect_log_format(filename):
    """
    检测 Zeek {conn_filename} 的格式（json 或 text）。
    读取第一条非空且非注释行，若能被 json.loads() 解析则视为 JSON，否则视为 TEXT。
    """
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            # 尝试解析
            return "json" if line.startswith('{') and _is_json(line) else "text"
    return "text"


def _is_json(line: str) -> bool:
    """轻量 JSON 检测函数"""
    try:
        json.loads(line)
        return True
    except Exception:
        return False

def extract_5tuple_from_row(row, fmt="cicflowmeter"):
    if fmt == "binetflow":
        return (
            row.get("SrcAddr", "").strip(),
            row.get("Sport", "").strip(),
            row.get("DstAddr", "").strip(),
            row.get("Dport", "").strip(),
            normalize_proto(row.get("Proto", "")),
        )
    elif fmt == "cicflowmeter":
        return (
            row.get("Source IP", "").strip(),
            row.get("Source Port", "").strip(),
            row.get("Destination IP", "").strip(),
            row.get("Destination Port", "").strip(),
            normalize_proto(row.get("Protocol", "")),
        )
    elif fmt == "zeek":
        return (
            row.get("id.orig_h", "").strip(),
            str(row.get("id.orig_p", "")).strip(),
            row.get("id.resp_h", "").strip(),
            str(row.get("id.resp_p", "")).strip(),
            normalize_proto(row.get("proto", "")),
        )
    return ("", "", "", "", "")


def attach_label_to_conn_json_record(rec, path_to_dataset, flow_label_dict):
    # 从 {conn_filename} 的 JSON 记录里取五元组
    src_ip, src_port, dst_ip, dst_port, proto = extract_5tuple_from_row(rec, fmt="zeek")

    rec["label"] = "Background"  # 先默认背景流量
    for dtime in get_possible_local_datetimes_from_zeek_ts(rec.get("ts"), path_to_dataset):
        key = (src_ip, src_port, dst_ip, dst_port, proto, dtime)
        matched_rows = flow_label_dict.get(key)
        if matched_rows:
            for row in matched_rows:
                label_val = row.get("Label") or row.get("label")
                if label_val:
                    rec["label"] = label_val
                    break

    if verbose and rec["label"] == "Background":
        print("No match for conn entry (CIC-IDS-2017):", 
                f"{src_ip}-{src_port}-{dst_ip}-{dst_port}-{proto}-{rec.get('ts')} or {dst_ip}-{dst_port}-{src_ip}-{src_port}-{proto}-{rec.get('ts')}")                        


def check_conn_label_in_json(file_name, path_to_dataset, flow_array, 
                             infected_ips_list=None, normal_ips_list=None, 
                             flow_label_dict=None,
                             use_folder_as_label=False, folder_name=None):
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        malicious_label = 0
        normal_label = 0
        background = 0
        total, label_matched = 0, 0
        protocol_filtered_count = 0
        for i, line in enumerate(f):
            line_strip = line.strip()
            if not line_strip:
                continue
            if line_strip.startswith('#'):
                continue
            try:
                flow_record = json.loads(line_strip)
            except Exception as e:
                print(f"Warning: skip invalid json line {i+1}: {e}")
                continue

            if use_folder_as_label:
                flow_record["label"] = folder_name
            elif flow_label_dict is not None:
                attach_label_to_conn_json_record(flow_record, path_to_dataset, flow_label_dict)                        
            elif infected_ips_list is not None and normal_ips_list is not None:
                src_address = flow_record.get("id.orig_h", "").strip()
                if src_address in normal_ips_list:
                    flow_record["label"] = "Normal"
                    normal_label += 1
                elif src_address in infected_ips_list:
                    flow_record["label"] = "Malicious"
                    malicious_label += 1
                else:
                    flow_record["label"] = "Background"
                    background += 1

            total += 1
            label_str = flow_record["label"]
            if label_str != "Background":
                label_lower = label_str.strip().lower()
                required_proto = _label_requires_proto(label_lower)

                # 只append成功匹配标签的流记录
                if required_proto is None:
                    flow_array.append(flow_record)
                    label_matched += 1

                else:
                    flow_uid = str(flow_record.get("uid", "")).lower().strip()
                    flow_proto = str(flow_record.get("proto", "")).lower().strip()
                    flow_service = str(flow_record.get("service", "")).lower().strip()

                    # ===== MQTT 特殊处理 =====
                    if required_proto == "mqtt":
                        # MQTT 必须是 TCP + service = mqtt
                        if flow_proto == "tcp" and flow_service == "mqtt":
                            flow_array.append(flow_record) # 需要的协议匹配了，才append
                            label_matched += 1
                        else:
                            protocol_filtered_count += 1
                            if protocol_filtered_count < 50:
                                print(f"Proto mismatch: {flow_uid}, label={label_str}, proto={flow_proto}, service={flow_service}")

                    # ===== 普通传输层协议 =====
                    else:
                        if flow_proto == required_proto:
                            flow_array.append(flow_record) # 需要的协议匹配了，才append
                            label_matched += 1
                        else:
                            protocol_filtered_count += 1
                            if protocol_filtered_count < 50:
                                print(f"Proto mismatch: {flow_uid}, label={label_str}, proto={flow_proto}, service={flow_service}")

    print(f"[Protocol Filter] removed {protocol_filtered_count} flows")
    print(f"[Summary] Matched: {label_matched}/{total} ({label_matched/total:.2%})")


def check_conn_label(path_to_dataset, conn_filename="conn.log", infected_ips_list=None, normal_ips_list=None, 
                     flow_label_dict=None,
                     use_folder_as_label=False, two_level_dataset_folder=False):
    '''
    读取 {conn_filename} 文件，检查并添加 label 列。返回处理后的行列表和文件格式（"json" 或 "text"）。
    该程序支持下面的三种打标签的方式。

    * 如果 use_folder_as_label 为 True，那么使用文件夹名作为 label。

    * 如果flow_label_dict字典不为空，那么可以采用flow_label对五元组打标签。
        **flow_label 文件本质**
        * 每一行代表一个流（flow），它是由一个扩展的 五元组
        (SrcAddr, SrcPort, DstAddr, DstPort, Proto)来标识的：
        * 再加上一些统计特征（持续时间、字节数、包数等）和标签（Label）。
        * 所以 flow_label_dict 通常就是以五元组作为键，存储该流的特征和标签。

    * 如果infected_ips_list和infected_ips_list不为空，那么根据源 IP 地址在 
    infected_ips_list 和 normal_ips_list 中的存在与否来设置 label。注意CTU-13僵尸网络数据的标注方法如下：
        * CTU-13 数据集的标注机制
        CTU-13 每个场景（Scenario）会给出一个 capture20110810.binetflow 这样的流量文件。
        同时还会提供一个 *.ips 文件（或 ips.txt），里面列出了：
        僵尸主机（infected IPs）
        正常主机（normal IPs）
        研究人员会用这些 IP 列表来给 binetflow 文件打标签。

        * 生成标签的方式
        在 CTU-13 原始论文和官方说明中，标注是基于源/目的 IP 是否出现在 infected_ips_list 里完成的：
        如果流的源 IP 或目的 IP 属于 infected IPs → Botnet 标签
        如果流的源 IP 和目的 IP 都属于 正常 IPs → Normal 标签
        其他情况（比如未知 IP 或未在列表中的） → Background 标签
    '''
    print("--------- Checking conn file -------------\n")

    malicious_label = 0
    normal_label = 0
    background = 0

    flow_array = []
    file_path = os.path.join(path_to_dataset, conn_filename)
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return flow_array, "text"

    folder_name = ""
    if use_folder_as_label:
        # 根据two_level_dataset_folder变量，获取双层或者单层文件夹名称，后续作为流量标签
        if two_level_dataset_folder:
            parent_dir = os.path.basename(os.path.dirname(os.path.normpath(path_to_dataset)))
            current_dir = os.path.basename(os.path.normpath(path_to_dataset))
            normalized_parent_dir = normalize_folder_label(parent_dir)
            folder_name = f"{normalized_parent_dir}_{current_dir}"
        else:
            folder_name = os.path.basename(os.path.normpath(path_to_dataset))

    normalized_folder_name = normalize_folder_label(folder_name)
    if use_folder_as_label:
        print(f"<< Using folder name '{normalized_folder_name}' as label >>\n")
        if normalized_folder_name == "":
            print("Warning: folder name is empty string.")
            raise ValueError("folder name is empty string.")

    file_format = detect_log_format(file_path)

    if file_format == "json":
        check_conn_label_in_json(file_path, path_to_dataset, flow_array, 
                                infected_ips_list, normal_ips_list, 
                                flow_label_dict,
                                use_folder_as_label, normalized_folder_name)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 保留原始行（不去掉注释行）
                if line.startswith('#'):
                    # 对于 fields/types 注释行，追加 label/类型信息（兼容原来行为）
                    if 'fields' in line:
                        flow_array.append(line.rstrip('\n') + '\t' + 'label' + '\n')
                    elif 'types' in line:
                        flow_array.append(line.rstrip('\n') + '\t' + 'string' + '\n')
                    else:
                        flow_array.append(line)
                    continue

                # 正常数据行
                split = line.rstrip('\n').split('\t')
                if use_folder_as_label:
                    flow_array.append(line.rstrip('\n') + '\t' + normalized_folder_name + '\n')
                    continue

                # 如果字段不足，直接保留原行并跳过计数
                if len(split) <= 2:
                    flow_array.append(line)
                    continue

                if use_folder_as_label:
                    flow_array.append(line.rstrip('\n') + '\t' + normalized_folder_name + '\n')
                elif infected_ips_list is not None and normal_ips_list is not None:
                    src_address = split[2].strip()
                    if src_address in normal_ips_list:
                        normal_label += 1
                        flow_array.append(line.rstrip('\n') + '\t' + "Normal" + '\n')
                    elif src_address in infected_ips_list:
                        malicious_label += 1
                        flow_array.append(line.rstrip('\n') + '\t' + "Malicious" + '\n')
                    else:
                        background += 1
                        flow_array.append(line.rstrip('\n') + '\t' + "Background" + '\n')

    if not use_folder_as_label and infected_ips_list is not None and normal_ips_list is not None:
        print("Malicious:", malicious_label)
        print("Normal:", normal_label)
        print("Background:", background)

    return flow_array, file_format


def write_conn_label(path, flow_array, file_format, output_filename="conn_label.log"):
    print(f"<< Writing {output_filename} --------------\n")
    index = 0
    output_path = os.path.join(path, output_filename)
    
    if file_format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            for rec in flow_array:
                f.write(json.dumps(rec) + "\n")
                index += 1
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in flow_array:
                f.write(line)
                index += 1

    print("     << Number of lines:", index)
    print(f"<< New file {output_filename} was successfully created.")


def check_flow_label_file_contain_label(path_to_csv):
    import csv
    label_i = None
    found_header = False

    with open(path_to_csv, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for cells in reader:
            cells = [c.strip() for c in cells]  # ★ 关键修正：去掉字段空格

            if not cells or cells[0].startswith('#'):
                continue

            if not found_header and 'Timestamp' in cells:
                for candidate in ('Label', 'label'):
                    if candidate in cells:
                        label_i = cells.index(candidate)
                        found_header = True
                        break
                continue

            if found_header and label_i is not None:
                if len(cells) > label_i and cells[label_i].strip():
                    return True
            else:
                if cells and cells[-1].strip():
                    return True

    return False


def detect_csv_format(path_to_csv):
    """
    判断一个 CSV 文件是 CTU-13 binetflow 格式还是 CICFlowMeter 格式
    """

    with open(path_to_csv, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)  # 读取表头
        header = [h.strip() for h in header]

    # 典型字段集合
    ctu13_keys = {"StartTime", "Dur", "Proto", "SrcAddr", "Sport", "DstAddr", "Dport", "State", "TotPkts", "TotBytes", "SrcBytes", "Label"}
    cic_keys = {"Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Flow Duration", "Label"}

    # 判断是否为 CTU-13 binetflow
    if ctu13_keys.intersection(header) and "SrcAddr" in header and "DstAddr" in header:
        return "CTU-13 binetflow"

    # 判断是否为 CICFlowMeter
    if cic_keys.intersection(header) and "Source IP" in header and "Destination IP" in header:
        return "CICFlowMeter"

    return "Unknown format"


def get_possible_local_datetimes_from_zeek_ts(utc_timestamp, path_to_dataset=None):
    """
    将 Zeek 的 UTC timestamp 转换为可能对应的 CICFlowMeter 本地时间字符串列表。
    
    CIC-IDS-2017 CSV 使用加拿大东部时间（Eastern Time，包含 EST/EDT）：
      - EST (UTC-5) 对应偏移 -23 小时（用于匹配 CSV 的时间格式）
      - EDT (UTC-4) 对应偏移 -11 小时（用于匹配 CSV 的时间格式）
    
    由于 CSV 时间可能存在 ±1~3 分钟的偏差，因此生成多种可能的时间。
    
    返回：
        List[str]：格式为 "日/月/年 小时:分钟" 的时间字符串
    """
    results = []

    # 需要匹配的偏移（单位：秒）
    minute_variations = [0, 60, -60, 120, -120, 180, -180]  # ±1~3 分钟误差
    timezone_offsets = [0*3600]
    if path_to_dataset and "CIC-IDS-2017" in os.path.normpath(path_to_dataset):
        timezone_offsets = [-23*3600, -11*3600]  # EST, EDT（与 CSV 对齐）
        
    for offset in timezone_offsets:
        for delta in minute_variations:
            ts_local = float(utc_timestamp) + offset + delta
            dt = datetime.fromtimestamp(ts_local)
            formatted_time = normalize_timestamp_str(dt.strftime("%d/%m/%Y %H:%M"))
            results.append(formatted_time)

    return results


def normalize_timestamp_str(ts):
    """
    仅当 ts 明显是日期时间格式时（如 6/7/2017 3:30 或 06/07/2017 00:03:01），
    才执行去前导零等标准化操作。
    """
    if not ts:
        return ts

    # 匹配常见的 datetime 格式，如 7/7/2017 3:30 或 07/07/2017 03:30:00
    datetime_pattern = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(:\d{2})?\s*$")

    if datetime_pattern.match(ts):
        ts = ts.strip()
        # 把原始时间字符串 ts 规范化，使日期和时间部分的格式更统一。
        # 原始时间 (ts)	处理后 (timestamp)	说明
        # "08/05/2025 09:03:01"	"8/5/2025 9:03:01"	去掉日期和小时的多余0
        # "09/01/2025 07:05:00"	"9/1/2025 7:05:00"	把 /0 改成 /， " 0" 改成 " "
        ts = ts.lstrip("0").replace("/0", "/").replace(" 0", " ")
        return ts
    else:
        # 不是标准日期时间格式，则原样返回
        return ts

def _label_requires_proto(label_lower: str):
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

def build_flow_label_dict(path_to_csv, csv_type="cicflowmeter"):
    """
    通用函数，根据 csv_type 自动解析 CSV 构建 flow_label_dict
    key: (SrcIP, SrcPort, DstIP, DstPort, Proto, UTC日期)
    value: list，包含该 flow 的所有字段（字典形式）
    
    csv_type: "cicflowmeter" 或 "binetflow"
    """
    flow_label_dict = {}

    # 自动选择编码
    encoding = 'utf-8-sig' if csv_type == "cicflowmeter" else 'utf-8'

    with open(path_to_csv, 'r', encoding=encoding, errors='ignore') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]

        for row in reader:
            # --- Step 1. 针对 CIC-IDS-2017 等不同数据集的专有字段修复 ---
            if "CIC-IDS-2017" in os.path.normpath(path_to_csv):
                ts = row.get("Timestamp", "").strip()
                if len(ts) >= 16:  # 例如 "06/07/2017 00:03:01"
                    ts = ts[:16]   # 保留到分钟
                row["Timestamp"] = ts

                label = row.get("Label") or row.get("label")
                if label == "Web Attack � Brute Force":
                    row["Label"] = "Web Attack – Brute Force"
                elif label == "Web Attack � XSS":
                    row["Label"] = "Web Attack – XSS"
                elif label == "Web Attack � Sql Injection":
                    row["Label"] = "Web Attack – Sql Injection"

            # --- Step 2. 提取五元组字段 ---
            src_ip, src_port, dst_ip, dst_port, proto = extract_5tuple_from_row(row, fmt=csv_type)

            # --- Step 3. 时间标准化 ---
            timestamp = normalize_timestamp_str(row.get("Timestamp", ""))

            # --- Step 4. 构建 key ---
            key = (src_ip, src_port, dst_ip, dst_port, proto, timestamp)
            if not all(key[:5]):  # 前5个字段必须完整
                continue

            if key not in flow_label_dict:
                flow_label_dict[key] = []
            flow_label_dict[key].append(row)

    # --- Step 5. 清洗多标签冲突 ---
    new_dict = {}
    for k, v in flow_label_dict.items():
        labels = [r.get("Label") or r.get("label") for r in v]
        # 删除空标签：过滤掉空值（或 None / 空字符串）的标签
        labels = [lbl for lbl in labels if lbl]
        if not labels:
            continue
        most_common, _ = Counter(labels).most_common(1)[0]
        new_dict[k] = [{"Label": most_common}]
    flow_label_dict = new_dict

    print(f"[build_flow_label_dict] {os.path.basename(path_to_csv)} loaded: {len(flow_label_dict)} unique flows")
    return flow_label_dict


def process_given_ip(path):
    normal_ips_list = set()
    infected_ips_list = set()
    label = None
    ipadr_path = os.path.join(path, "IPadr.txt")
    if not os.path.exists(ipadr_path):
        return infected_ips_list, normal_ips_list

    with open(ipadr_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low == 'normal':
                label = 'Normal'
                continue
            if low == 'malicious':
                label = 'Malicious'
                continue
            # 非 label 行，应为 IP
            if label == 'Normal':
                normal_ips_list.add(line)
            elif label == 'Malicious':
                infected_ips_list.add(line)
    return infected_ips_list, normal_ips_list


def normalize_folder_label(folder_name: str) -> str:
    """
    Normalize dataset folder name into label:
    - remove leading train_ / test_
    - remove trailing _train / _test
    - remove trailing index suffix like:
        _1, _3a, -2b, 1a, 12, etc.
    - remove data format indicators like PCAPs / Logs / Raw
    """
    if not folder_name:
        return folder_name

    name = folder_name

    # 0. remove leading train_ / test_
    name = re.sub(r'^(train|test)_', '', name, flags=re.IGNORECASE)

    # 1. remove trailing _train / _test
    name = re.sub(r'_(train|test)$', '', name, flags=re.IGNORECASE)

    # 2. remove trailing index suffix
    # patterns covered:
    #   _1, _3a, -2b, 1a, 12
    name = re.sub(r'([_-]?\d+[a-zA-Z]?)$', '', name)

    # 3. 移除数据格式标识（PCAPs / Logs / Raw）
    name = re.sub(r'[-_](pcaps?|logs?|raw)$', '', name, flags=re.IGNORECASE)

    # 4. remove trailing index suffix
    name = re.sub(r'[-_]+$', '', name)

    return name


def label_conn_log(path, two_level_dataset_folder=False, conn_filename="conn.log"):
    print(">>>------------------------------------------------------------<<<")
    use_folder_as_label = False
    flow_label_dict = None
    infected_ips_list, normal_ips_list = None, None
    print(path)
    flow_label_file_paths = find_flow_label_file_names(path)
    if len(flow_label_file_paths) > 0:
        for f in flow_label_file_paths:
            csv_format = detect_csv_format(f)
            if csv_format == "CTU-13 binetflow":
                f_dict = build_flow_label_dict(f, csv_type="binetflow")
                print("Detected CTU-13 binetflow format, and built flow_label_dict with", len(f_dict), "entries from ", f)
            elif csv_format == "CICFlowMeter":
                f_dict = build_flow_label_dict(f, csv_type="cicflowmeter")
                print("Detected CICFlowMeter format, and built flow_label_dict with", len(f_dict), "entries from ", f)
            else:
                print("Warning: Unknown CSV format, cannot build flow_label_dict.")
            
            if flow_label_dict is None:
                flow_label_dict = f_dict
            else:
                flow_label_dict.update(f_dict)  # 合并多个文件的内容
        print(f"Total flow_label_dict size from {len(flow_label_file_paths)} files: num_entries = {len(flow_label_dict)}")
    elif os.path.exists(os.path.join(path, "IPadr.txt")):
        infected_ips_list, normal_ips_list = process_given_ip(path)
    else: # 没有flow_label_file或者Ipadr.txt标签文件的时候，使用目录名作为标签
        if two_level_dataset_folder:
            parent_dir = os.path.basename(os.path.dirname(os.path.normpath(path)))
            current_dir = os.path.basename(os.path.normpath(path))
            normalized_parent_dir = normalize_folder_label(parent_dir)
            folder_name = f"{normalized_parent_dir}_{current_dir}"
        else:
            folder_name = os.path.basename(os.path.normpath(path))

        normalized_folder_name = normalize_folder_label(folder_name)

        print(f"No *.binetflow or *.csv file for flow labeling or IPadr.txt found in {path}, so use normalized folder name '{normalized_folder_name}' as label")
        use_folder_as_label = True

    if infected_ips_list:
        print("Infected ip list: ", infected_ips_list)
    else:
        print("Infected ip list is empty")
    if normal_ips_list:
        print("Normal ip list: ", normal_ips_list)
    else:
        print("Normal ip list is empty")

    # 处理 {conn_filename}，给其打标签
    conn_file_exists = os.path.exists(os.path.join(path, conn_filename))
    if conn_file_exists:
        conn_flow_array, conn_file_format = check_conn_label(path, conn_filename=conn_filename,
                                                             infected_ips_list=infected_ips_list, 
                                                             normal_ips_list=normal_ips_list,
                                                             flow_label_dict=flow_label_dict,
                                                             use_folder_as_label=use_folder_as_label, 
                                                             two_level_dataset_folder=two_level_dataset_folder)
        # 从conn_filename生成输出文件名
        base_name = os.path.splitext(conn_filename)[0]  # 剥离扩展名
        output_filename = f"{base_name}_label.log"
        if conn_flow_array:
            write_conn_label(path, conn_flow_array, conn_file_format, output_filename=output_filename)
        else:
            print(f"Warning: conn_flow_array is empty, skip writing {output_filename}")
    else:
        print(f"No {conn_filename} file found in the directory.")
        
    print('\n\n')