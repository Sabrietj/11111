# -*- coding: utf-8 -*-

from print_manager import __PrintManager__
import os
import csv
import traceback

import sys
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
from zeek_columns import conn_columns, flowmeter_columns, ssl_columns, x509_columns, dns_columns, http_columns, max_x509_cert_chain_len, \
                    ftp_columns, mqtt_connect_columns, mqtt_subscribe_columns, mqtt_publish_columns

def reduce_events_aligned(events, columns):
    """
    events: list[dict]
    columns: zeek_columns (http_columns / ftp_columns / mqtt_columns)

    返回：
    - 单事件 → dict[str, scalar]
    - 多事件 → dict[str, list]（None 对齐）
    """
    if not events:
        return {k: "" for k in columns}

    # 单事件：保持原样（不升级成 list）
    if len(events) == 1:
        ev = events[0]
        return {k: ev.get(k, "") for k in columns}

    # 多事件：字段级对齐 + None
    result = {}
    for k in columns:
        result[k] = [ev.get(k, None) for ev in events]
    
    return result

class EvaluateData(object):
    def __init__(self):
        self.session_tuple = dict()
        self.cert_dict = dict()

    def create_plot_data(self, path, filename):
        print(f"\nCreating plot data for {filename} in {path}")        
        __PrintManager__.evaluate_creating_plot()
        self.create_session_csv(path, filename+"-session"+".csv")
        self.create_flow_list_csv(self.session_tuple, path, filename+"-flow"+".csv")
        __PrintManager__.success_evaluate_data()


    # NOTE:
    # HTTP / FTP / MQTT 字段在 CSV 中可能是：
    #   - scalar（单事件）
    #   - list（多事件，None 对齐）
    # 仅当 allow_1toN=True 且实际存在多事件时才会出现 list。
    def create_flow_list_csv(self, session_tuple_dict, output_path, filename="flow_list.csv"):
        """创建流列表CSV文件，带调试信息"""
        print(f"\n>>> 开始创建流列表CSV: {filename}")
        
        # 特定UID调试
        target_uid = "COIC5J39wSkYdzNLah"
        target_flow_count = 0        
        
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, filename)

        # ===== 新增：如果文件已存在，直接返回 =====
        if os.path.exists(output_file):
            print(f"\n>>> ⚠️ CSV 文件已存在，跳过创建: {output_file}")
            return
        # ===========================================

        # 找到证书链的最大长度
        # 设置了证书链的最大证书数量是3
        # * ​​cert_chain_fps​​：服务器证书链的 SHA1 指纹列表，长度通常为 ​​1-3​​（服务器证书 + 中间 CA）。
        # ​* ​client_cert_chain_fps​​：客户端证书链的 SHA1 指纹列表（仅在双向认证时存在），长度通常为 ​​1-2​​。
        # ​* ​自签名证书​​的证书链长度一般是1
        # * 错误配置​​的情况下，证书链长度可能 > 3。但是很罕见，可能因错误配置发送冗余证书（如重复中间 CA 或无关证书）
        # max_x509_cert_chain_len = max(len(flow.x509_logs) if isinstance(flow.x509_logs, list) else 0 
        #                             for session in session_tuple_dict.values() 
        #                             for flow in session.flow_list)
        headers = ["uid", "label", "is_malicious"] + \
                [f"conn.{col}" for col in conn_columns] + \
                [f"flowmeter.{col}" for col in flowmeter_columns] + \
                [f"dns.{col}" for col in dns_columns] + \
                [f"ssl.{col}" for col in ssl_columns] + \
                [f"x509.cert{i}.{col}" for i in range(max_x509_cert_chain_len) for col in x509_columns] + \
                [f"http.{col}" for col in http_columns] + \
                [f"ftp.{col}" for col in ftp_columns] + \
                [f"mqtt_connect.{col}" for col in mqtt_connect_columns] + \
                [f"mqtt_subscribe.{col}" for col in mqtt_subscribe_columns] + \
                [f"mqtt_publish.{col}" for col in mqtt_publish_columns]

        def _safe_dict(obj, *, name=None, uid=None):
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj
            print(f"⚠️ 非法 {name or 'dict'} 类型:",
                type(obj),
                f"uid={uid}")
            return {}

        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            # writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL, escapechar="\\", line_terminator="\n")
            writer = csv.DictWriter(f, fieldnames=headers, line_terminator="\n")
            writer.writeheader()
            
            total_flows = 0
            ssl_flows = 0            

            for session in session_tuple_dict.values():
                for flow in session.flow_list:
                    total_flows += 1
                    if flow.ssl_log is not None and isinstance(flow.ssl_log, dict) and len(flow.ssl_log) > 0:
                        ssl_flows += 1

                    conn_data = _safe_dict(flow.conn_log, name="conn_log", uid=flow.uid)
                    flowmeter_data = _safe_dict(flow.flowmeter_log, name="flowmeter_log", uid=flow.uid)
                    ssl_data = _safe_dict(flow.ssl_log, name="ssl_log", uid=flow.uid)
                    dns_data = _safe_dict(flow.dns_log, name="dns_log", uid=flow.uid)

                    if isinstance(flow.http_log, list):
                        http_data = reduce_events_aligned(flow.http_log, http_columns)
                    else:
                        http_data = _safe_dict(flow.http_log, name="http_log", uid=flow.uid)

                    if isinstance(flow.ftp_log, list):
                        ftp_data = reduce_events_aligned(flow.ftp_log, ftp_columns)
                    else:
                        ftp_data = _safe_dict(flow.ftp_log, name="ftp_log", uid=flow.uid)

                    # ===== MQTT =====
                    mqtt_log = flow.mqtt_log

                    if flow.mqtt_log is not None and not isinstance(flow.mqtt_log, dict):
                        print("⚠️ 非法 mqtt_log:", type(flow.mqtt_log), flow.uid)
                        mqtt_log = {}

                    def handle_mqtt_sublog(raw, columns):
                        if isinstance(raw, list):
                            return reduce_events_aligned(raw, columns)
                        if isinstance(raw, dict):
                            return {k: raw.get(k, "") for k in columns}
                        return {}
                    
                    mqtt_connect = handle_mqtt_sublog(mqtt_log.get("connect"), mqtt_connect_columns)
                    mqtt_subscribe = handle_mqtt_sublog(mqtt_log.get("subscribe"), mqtt_subscribe_columns)
                    mqtt_publish = handle_mqtt_sublog(mqtt_log.get("publish"), mqtt_publish_columns)
                        
                    # 特定UID调试
                    if flow.uid == target_uid:
                        target_flow_count += 1
                        print(f">>> 🔍 在CSV生成中找到目标UID: {target_uid}")
                        print(f">>>   Flow SSL日志: {flow.ssl_log is not None}")
                        if flow.ssl_log:
                            print(f">>>   SSL版本字段: {flow.ssl_log.get('version', 'N/A')}")
                            ssl_flows += 1
                        
                    # 每个证书链成员
                    x509_logs = flow.x509_logs if isinstance(flow.x509_logs, list) else []
                    x509_flat = {}
                    for i in range(max_x509_cert_chain_len):
                        cert_data = x509_logs[i] if i < len(x509_logs) else {}
                        x509_flat.update({f"x509.cert{i}.{k}": cert_data.get(k, "") for k in x509_columns})

                    flow_record = {
                        "uid": flow.uid,
                        "label": flow.get_label(),
                        "is_malicious": flow.is_malicious(),
                        **{f"conn.{k}": conn_data.get(k, "") for k in conn_columns},
                        **{f"flowmeter.{k}": flowmeter_data.get(k, "") for k in flowmeter_columns},                        
                        **{f"dns.{k}": dns_data.get(k, "") for k in dns_columns},
                        **{f"ssl.{k}": ssl_data.get(k, "") for k in ssl_columns},
                        **x509_flat,
                        **{f"http.{k}": http_data.get(k, "") for k in http_columns},
                        **{f"ftp.{k}": ftp_data.get(k, "") for k in ftp_columns},
                        **{f"mqtt_connect.{k}": mqtt_connect.get(k, "") for k in mqtt_connect_columns},
                        **{f"mqtt_subscribe.{k}": mqtt_subscribe.get(k, "") for k in mqtt_subscribe_columns},
                        **{f"mqtt_publish.{k}": mqtt_publish.get(k, "") for k in mqtt_publish_columns}
                    }
                    
                    try:
                        writer.writerow(flow_record)
                    except Exception:
                        print("\n>>> ❌ 写 CSV 出错")
                        print(">>> uid =", flow.uid)
                        print(">>> conn_log =", type(flow.conn_log))
                        print(">>> dns_log =", type(flow.dns_log))
                        print(">>> ssl_log =", type(flow.ssl_log))
                        print(">>> http_log =", type(flow.http_log))
                        print(">>> ftp_log =", type(flow.ftp_log))
                        print(">>> mqtt_log =", type(flow.mqtt_log))
                        traceback.print_exc()
                        raise
            
            print(f"\n>>> CSV生成完成: 总流数={total_flows}, 含SSL流数={ssl_flows}")
            # print(f">>> 目标UID出现次数: {target_flow_count}")                    
            print("\n<<< dataset file %s-flow.csv successfully created !" % filename)

    def create_session_csv(self, path, filename="session_list.csv"):
        print(f"\nCreating session dataset for {filename}")        
        index = 0
        ssl_flow = 0
        all_flow = 0
        malicious = 0
        normal = 0

        # file header: label feature
        header = [
            'session_index',
            'is_malicious',
            'ssl_version',
            'cipher_suite_server',
            'cert_key_alg',
            'cert_sig_alg',
            'cert_key_type',
            'max_duration',
            'avg_duration',
            'percent_of_std_duration',
            'number_of_flows',
            'ssl_flow_ratio',
            'avg_size',
            'recv_sent_size_ratio',
            'avg_pkts',
            'recv_sent_pkts_ratio',
            'packet_loss',
            'percent_of_established_state',
            'avg_time_diff',
            'std_time_diff',
            'max_time_diff',
            'ssl_tls_ratio',
            'resumed',
            'self_signed_ratio',
            'avg_key_length',
            'avg_cert_valid_day',
            'std_cert_valid_day',
            'percent_of_valid_cert',
            'avg_valid_cert_percent',
            'number_of_cert_serial',
            'number_of_domains_in_cert',
            'avg_cert_path',
            'x509_ssl_ratio',
            'SNI_ssl_ratio',
            'is_SNIs_in_SNA_dns',
            'is_CNs_in_SNA_dns',
            'subject_CN_is_IP',
            'subject_is_com',
            'is_O_in_subject',
            'is_CO_in_subject',
            'is_ST_in_subject',
            'is_L_in_subject',
            'subject_only_CN',
            'issuer_is_com',
            'is_O_in_issuer',
            'is_CO_in_issuer',
            'is_ST_in_issuer',
            'is_L_in_issuer',
            'issuer_only_CN',
            'avg_TTL',
            'avg_domain_name_length',
            'std_domain_name_length',
            'avg_IPs_in_DNS',
            'flow_uid_list'  # 新增的列
        ]

        output_file = os.path.join(path, filename)
        
        with open(output_file, 'w+', newline='', encoding='utf-8') as f:
            # writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar="\\", line_terminator="\n")
            writer = csv.writer(f, line_terminator="\n")
            writer.writerow(header)
            for key in self.session_tuple:
                session = self.session_tuple[key]
                is_malicious = session.is_malicious()
                
                # 构建 flow list 信息
                flow_uid_list = []
                for flow in session.flow_list:
                    flow_uid_list.append(flow.uid)

                session_feature = [\
                str(key),\
                is_malicious,\
                str(session.ssl_version()),\
                str(session.cipher_suite_server()),\
                str(session.cert_key_alg()),\
                str(session.cert_sig_alg()),\
                str(session.cert_key_type()),\
                str(session.max_duration()),\
                str(session.avg_duration()),\
                str(session.percent_of_std_duration()),\
                str(session.number_of_flows()),\
                str(session.ssl_flow_ratio()),\
                str(session.avg_size()),\
                str(session.recv_sent_size_ratio()),\
                str(session.avg_pkts()),\
                str(session.recv_sent_pkts_ratio()),\
                str(session.packet_loss()),\
                str(session.percent_of_established_state()),\
                str(session.avg_time_diff()),\
                str(session.std_time_diff()),\
                str(session.max_time_diff()),\
                str(session.ssl_tls_ratio()),\
                str(session.resumed()),\
                str(session.self_signed_ratio()),\
                str(session.avg_key_length()),\
                str(session.avg_cert_valid_day()),\
                str(session.std_cert_valid_day()),\
                str(session.percent_of_valid_cert()),\
                str(session.avg_valid_cert_percent()),\
                str(session.number_of_cert_serial()),\
                str(session.number_of_domains_in_cert()),\
                str(session.avg_cert_path()),\
                str(session.x509_ssl_ratio()),\
                str(session.SNI_ssl_ratio()),\
                str(session.is_SNIs_in_SNA_dns()),\
                str(session.is_CNs_in_SNA_dns()),\
                str(session.subject_CN_is_IP()),\
                str(session.subject_is_com()),\
                str(session.is_O_in_subject()),\
                str(session.is_CO_in_subject()),\
                str(session.is_ST_in_subject()),\
                str(session.is_L_in_subject()),\
                str(session.subject_only_CN()),\
                str(session.issuer_is_com()),\
                str(session.is_O_in_issuer()),\
                str(session.is_CO_in_issuer()),\
                str(session.is_ST_in_issuer()),\
                str(session.is_L_in_issuer()),\
                str(session.issuer_only_CN()),\
                str(session.avg_TTL()),\
                str(session.avg_domain_name_length()),\
                str(session.std_domain_name_length()),\
                str(session.avg_IPs_in_DNS()),\
                str(flow_uid_list)
                ]
                writer.writerow(session_feature)

        print("\n<<< dataset file %s-session.csv successfully created !" % filename)