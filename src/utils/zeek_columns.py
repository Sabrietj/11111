
# conn.log 标准字段 (Zeek 官方文档 https://docs.zeek.org/en/master/logs/conn.html)
conn_columns = [
    'ts',                # Timestamp of the first packet in the connection
    'uid',               # Unique ID for the connection
    'id.orig_h',         # Source IP
    'id.orig_p',         # Source port
    'id.resp_h',         # Destination IP
    'id.resp_p',         # Destination port
    'proto',             # Transport-layer protocol (TCP/UDP/ICMP)
    'service',           # Detected application protocol/service
    'duration',          # Connection duration in seconds
    'orig_bytes',        # Total bytes sent by originator
    'resp_bytes',        # Total bytes sent by responder
    'conn_state',        # Connection state (e.g., SF, S0, REJ)
    'local_orig',        # True if originator is local
    'local_resp',        # True if responder is local
    'missed_bytes',      # Bytes missed due to capture loss
    'history',           # TCP flags seen in connection
    'orig_pkts',         # Number of packets sent by originator
    'orig_ip_bytes',     # Number of IP-layer bytes sent by originator
    'resp_pkts',         # Number of packets sent by responder
    'resp_ip_bytes',     # Number of IP-layer bytes sent by responder
    'tunnel_parents'     # Parent tunnels if encapsulated
]

conn_numeric_columns = [
    'duration',          # Connection duration in seconds
    'orig_bytes',        # Total bytes sent by originator
    'resp_bytes',        # Total bytes sent by responder
    'missed_bytes',      # Bytes missed due to capture loss
    'orig_pkts',         # Number of packets sent by originator
    'orig_ip_bytes',     # Number of IP-layer bytes sent by originator
    'resp_pkts',         # Number of packets sent by responder
    'resp_ip_bytes',     # Number of IP-layer bytes sent by responder
]

conn_categorical_columns = [
    'proto',        # TCP/UDP/ICMP
    'service',      # http/ssl/dns/ftp/ssh...
    'conn_state',   # SF / S0 / REJ / RSTO...
    'local_orig',   # True/False
    'local_resp',   # True/False
]

conn_textual_columns = [
    'history',        # Flag sequence (S, F, PA, ...)
    'tunnel_parents', # Zeek list-style text
]

# 基本流量统计特征
flowmeter_statistical_basic_columns = [
    'flow_duration',         # 流持续时间（秒）
    'fwd_pkts_tot',          # 正向总包数
    'bwd_pkts_tot',          # 反向总包数
    'fwd_data_pkts_tot',     # 正向有效载荷包数（包含数据的包）
    'bwd_data_pkts_tot',     # 反向有效载荷包数（包含数据的包）
    'fwd_pkts_per_sec',      # 正向包速率（包/秒）
    'bwd_pkts_per_sec',      # 反向包速率（包/秒）
    'flow_pkts_per_sec',     # 总包速率（包/秒）
    'down_up_ratio',         # 下行/上行数据包比（反向/正向包数比）

    'fwd_header_size_tot',   # 正向包头总长度（字节）
    'fwd_header_size_min',   # 正向包头最小长度（字节）
    'fwd_header_size_max',   # 正向包头最大长度（字节）
    'bwd_header_size_tot',   # 反向包头总长度（字节）
    'bwd_header_size_min',   # 反向包头最小长度（字节）
    'bwd_header_size_max',   # 反向包头最大长度（字节）
]

# TCP标志统计特征
flowmeter_statistical_tcp_columns = [
    'flow_FIN_flag_count',   # FIN标志总数（连接终止标志）
    'flow_SYN_flag_count',   # SYN标志总数（连接建立标志）
    'flow_RST_flag_count',   # RST标志总数（连接重置标志）
    'fwd_PSH_flag_count',    # 正向PSH标志总数（推送数据标志）
    'bwd_PSH_flag_count',    # 反向PSH标志总数（推送数据标志）
    'flow_ACK_flag_count',   # ACK标志总数（确认标志）
    'fwd_URG_flag_count',    # 正向URG标志总数（紧急数据标志）
    'bwd_URG_flag_count',    # 反向URG标志总数（紧急数据标志）
    'flow_CWR_flag_count',   # CWR标志总数（拥塞窗口减少标志）
    'flow_ECE_flag_count',   # ECE标志总数（显式拥塞通知回显标志）
]

# 窗口大小特征
flowmeter_statistical_window_columns = [
    'fwd_init_window_size',  # 初始正向窗口大小（字节）
    'bwd_init_window_size',  # 初始反向窗口大小（字节）
    'fwd_last_window_size',  # 最后正向窗口大小（字节）
    'bwd_last_window_size',  # 最后反向窗口大小（字节）
]

# 载荷统计特征（数据包有效载荷）
flowmeter_statistical_payload_columns = [
    'fwd_pkts_payload.min',  # 正向包最小载荷长度（字节）
    'fwd_pkts_payload.max',  # 正向包最大载荷长度（字节）
    'fwd_pkts_payload.tot',  # 正向包载荷总长度（字节）
    'fwd_pkts_payload.avg',  # 正向包平均载荷长度（字节）
    'fwd_pkts_payload.std',  # 正向包载荷长度标准差（字节）
    'bwd_pkts_payload.min',  # 反向包最小载荷长度（字节）
    'bwd_pkts_payload.max',  # 反向包最大载荷长度（字节）
    'bwd_pkts_payload.tot',  # 反向包载荷总长度（字节）
    'bwd_pkts_payload.avg',  # 反向包平均载荷长度（字节）
    'bwd_pkts_payload.std',  # 反向包载荷长度标准差（字节）
    'flow_pkts_payload.min', # 流中所有包的最小载荷长度（字节）
    'flow_pkts_payload.max', # 流中所有包的最大载荷长度（字节）
    'flow_pkts_payload.tot', # 流中所有包载荷总长度（字节）
    'flow_pkts_payload.avg', # 流中所有包平均载荷长度（字节）
    'flow_pkts_payload.std', # 流中所有包载荷长度标准差（字节）
    'payload_bytes_per_second', # 每秒有效载荷字节数（字节/秒）
]

# 时间间隔统计特征（IAT - Inter Arrival Time）
flowmeter_statistical_iat_columns = [
    'fwd_iat.min',          # 正向包最小到达时间间隔（秒）
    'fwd_iat.max',          # 正向包最大到达时间间隔（秒）
    'fwd_iat.tot',          # 正向包总到达时间间隔（秒）
    'fwd_iat.avg',          # 正向包平均到达时间间隔（秒）
    'fwd_iat.std',          # 正向包到达时间间隔标准差（秒）
    'bwd_iat.min',          # 反向包最小到达时间间隔（秒）
    'bwd_iat.max',          # 反向包最大到达时间间隔（秒）
    'bwd_iat.tot',          # 反向包总到达时间间隔（秒）
    'bwd_iat.avg',          # 反向包平均到达时间间隔（秒）
    'bwd_iat.std',          # 反向包到达时间间隔标准差（秒）
    'flow_iat.min',         # 流中所有包最小到达时间间隔（秒）
    'flow_iat.max',         # 流中所有包最大到达时间间隔（秒）
    'flow_iat.tot',         # 流中所有包总到达时间间隔（秒）
    'flow_iat.avg',         # 流中所有包平均到达时间间隔（秒）
    'flow_iat.std',         # 流中所有包到达时间间隔标准差（秒）
]

# Subflow和Bulk流量特征
flowmeter_statistical_subflow_bulk_columns = [
    'fwd_subflow_pkts',     # 正向子流包数
    'bwd_subflow_pkts',     # 反向子流包数
    'fwd_subflow_bytes',    # 正向子流字节数
    'bwd_subflow_bytes',    # 反向子流字节数
    'fwd_bulk_bytes',       # 正向Bulk流量字节数（连续大数据传输）
    'bwd_bulk_bytes',       # 反向Bulk流量字节数（连续大数据传输）
    'fwd_bulk_packets',     # 正向Bulk流量包数
    'bwd_bulk_packets',     # 反向Bulk流量包数
    'fwd_bulk_rate',        # 正向Bulk速率（字节/秒）
    'bwd_bulk_rate',        # 反向Bulk速率（字节/秒）
]

# 活跃和空闲时间统计特征
flowmeter_statistical_active_idle_columns = [
    'active.min',           # 最小活跃时间（秒）- 流中有数据传输的时间段
    'active.max',           # 最大活跃时间（秒）
    'active.tot',           # 总活跃时间（秒）
    'active.avg',           # 平均活跃时间（秒）
    'active.std',           # 活跃时间标准差（秒）
    'idle.min',             # 最小空闲时间（秒）- 流中无数据传输的时间段
    'idle.max',             # 最大空闲时间（秒）
    'idle.tot',             # 总空闲时间（秒）
    'idle.avg',             # 平均空闲时间（秒）
    'idle.std',             # 空闲时间标准差（秒）
]

# 数据包序列向量特征（用于时序分析）
flowmeter_packet_vector_columns = [
    'packet_direction_vector',     # 数据包方向向量（1:正向, -1:反向）
    'packet_timestamp_vector',     # 数据包时间戳向量（相对时间）
    'packet_iat_vector',           # 数据包间隔时间向量（秒）
    'packet_payload_size_vector',  # 数据包载荷大小向量（字节）
    'tcp_packet_ack_vector',       # TCP ACK标志向量（0:无ACK, 1:有ACK）
    'tcp_packet_psh_vector',       # TCP PSH标志向量（0:无PSH, 1:有PSH）
    'tcp_packet_seqno_vector',     # TCP序号向量（相对序列号）
]

# Subflow和Bulk索引特征
flowmeter_subflow_bulk_index_vector_columns = [
    'num_subflows',                 # 子流数量
    'subflow_first_packet_index_vector',  # 子流首包索引向量
    'num_fwd_bulks',                # 正向Bulk数量
    'num_bwd_bulks',                # 反向Bulk数量
    'bulk_first_packet_index_vector',     # Bulk首包索引向量
    'bulk_length_vector',           # Bulk长度向量（字节）
    'bulk_packet_index_vector',     # Bulk包索引向量
]

# 链路层地址特征
flowmeter_l2_addr_columns = [
    'orig_l2_addr',          # 源MAC地址
    'resp_l2_addr',          # 目的MAC地址
]

# flowmeter.log 字段 (根据Zeek FlowMeter插件 https://gitee.com/seu-csqjxiao/zeek-flowmeter)
flowmeter_columns = [
    # 流标识字段（这些不在统计特征中，需要单独添加）
    'ts',                    # 流开始时间戳
    'uid',                   # 流唯一ID
    'id.orig_h',             # 源IP
    'id.orig_p',             # 源端口
    'id.resp_h',             # 目的IP
    'id.resp_p',             # 目的端口
    'proto',                 # 协议类型 (TCP/UDP/ICMP)
] + flowmeter_statistical_basic_columns + flowmeter_statistical_tcp_columns + flowmeter_statistical_payload_columns + flowmeter_statistical_iat_columns + flowmeter_statistical_window_columns + flowmeter_statistical_subflow_bulk_columns + flowmeter_statistical_active_idle_columns + flowmeter_packet_vector_columns + flowmeter_subflow_bulk_index_vector_columns + flowmeter_l2_addr_columns

flowmeter_numeric_columns = (
    flowmeter_statistical_basic_columns +
    flowmeter_statistical_tcp_columns +
    flowmeter_statistical_payload_columns +
    flowmeter_statistical_iat_columns +
    flowmeter_statistical_window_columns +
    flowmeter_statistical_subflow_bulk_columns +
    flowmeter_statistical_active_idle_columns
)

flowmeter_categorical_columns = [
    'proto',             # Transport-layer protocol (TCP/UDP/ICMP)
]

flowmeter_textual_columns = [ # FlowMeter 中没有文本字段    
]


# ssl.log 标准字段 (Zeek官方文档 https://docs.zeek.org/en/master/logs/ssl.html)
ssl_columns = [
    'ts',                        # Timestamp of first packet in the SSL connection
    'uid',                       # Unique ID for the connection
    'id.orig_h',                  # Source IP
    'id.orig_p',                  # Source port
    'id.resp_h',                  # Destination IP
    'id.resp_p',                  # Destination port
    'version',                    # SSL/TLS version
    'cipher',                     # Negotiated cipher
    'curve',                      # Named curve used (if applicable)
    'server_name',                # Server Name Indication (SNI)
    'resumed',                     # True if session resumed
    'established',                 # True if SSL/TLS handshake completed
    'cert_chain_fps',              # Fingerprints of server certificate chain
    'client_cert_chain_fps',       # Fingerprints of client certificate chain
    'subject',                     # Subject of server certificate
    'issuer',                      # Issuer of server certificate
    'ja3',                         # JA3 fingerprint of client hello
    'ja3s',                        # JA3S fingerprint of server hello
    'next_protocol',               # Next protocol negotiated (NPN/ALPN)
    'ssl_history',                 # Record of SSL/TLS handshake messages
    'client_supported_versions',   # TLS versions supported by client
    'server_supported_versions',   # TLS versions supported by server
    'client_key_exchange_groups',  # Client key exchange groups
    'server_key_exchange_groups',  # Server key exchange groups
    'client_signature_algorithms', # Client signature algorithms
    'server_signature_algorithms', # Server signature algorithms
    'client_ec_point_formats',     # Client EC point formats
    'server_ec_point_formats',     # Server EC point formats
    'sni_matches_cert'             # True if SNI matches certificate
]

ssl_identifier_columns = {
    'ts',                        # Timestamp of first packet in the SSL connection
    'uid',                       # Unique ID for the connection
    'id.orig_h',                  # Source IP
    'id.orig_p',                  # Source port
    'id.resp_h',                  # Destination IP
    'id.resp_p',                  # Destination port
}

ssl_numeric_columns = { # Zeek 官方 ssl.log 本身 没有真正意义的 numeric 字段

}

ssl_categorical_columns = {
    'version',                    # SSL/TLS version
    'cipher',                     # Negotiated cipher
    'curve',                      # Named curve used (if applicable)
    'resumed',                     # True if session resumed
    'established',                 # True if SSL/TLS handshake completed    
    'next_protocol',               # Next protocol negotiated (NPN/ALPN)
    'sni_matches_cert'             # True if SNI matches certificate
}

ssl_textual_columns = {
    'server_name',                # Server Name Indication (SNI)
    'subject',                     # Subject of server certificate
    'issuer',                      # Issuer of server certificate
}

# 设置了证书链的最大证书数量是3
# * ​​cert_chain_fps​​：服务器证书链的 SHA1 指纹列表，长度通常为 ​​1-3​​（服务器证书 + 中间 CA）。
# ​* ​client_cert_chain_fps​​：客户端证书链的 SHA1 指纹列表（仅在双向认证时存在），长度通常为 ​​1-2​​。
# ​* ​自签名证书​​的证书链长度一般是1
# * 错误配置​​的情况下，证书链长度可能 > 3。但是很罕见，可能因错误配置发送冗余证书（如重复中间 CA 或无关证书）
max_x509_cert_chain_len = 3

# x509.log 标准字段 (Zeek官方文档 https://docs.zeek.org/en/master/logs/x509.html)
x509_columns = [
    'ts',                                      # Timestamp of certificate capture
    'id',                                      # Unique ID for the connection
    'certificate.version',                     # Certificate version
    'certificate.serial',                      # Certificate serial number
    'certificate.subject',                     # Certificate subject
    'certificate.issuer',                      # Certificate issuer
    'certificate.not_valid_before',            # Validity start
    'certificate.not_valid_after',             # Validity end
    'certificate.key_alg',                     # Public key algorithm
    'certificate.sig_alg',                     # Signature algorithm
    'certificate.key_type',                    # Key type
    'certificate.key_length',                  # Key length
    'certificate.exponent',                    # Public exponent (if RSA)
    'certificate.curve',                       # Named curve (if EC)
    'san.dns',                                 # Subject Alternative Name DNS
    'san.uri',                                 # SAN URI
    'san.email',                               # SAN email
    'san.ip',                                  # SAN IP
    'basic_constraints.ca',                    # CA flag
    'basic_constraints.path_len',              # Max path length
    'certificate.extension.v3_subject_alternative_name',   # v3 SAN extension
    'certificate.extension.v3_basic_constraints',          # v3 Basic Constraints
    'certificate.extension.v3_key_usage',                   # v3 Key Usage
    'certificate.extension.v3_extended_key_usage',          # v3 Extended Key Usage
    'certificate.extension.v3_crl_distribution_points',    # v3 CRL Distribution Points
    'certificate.extension.v3_certificate_policies',       # v3 Certificate Policies
    'certificate.extension.v3_authority_key_identifier',   # v3 Authority Key ID
    'certificate.extension.v3_subject_key_identifier',     # v3 Subject Key ID
    'certificate.extension.v3_issuer_alternative_name',    # v3 Issuer Alt Name
    'certificate.extension.v3_name_constraints',           # v3 Name Constraints
    'certificate.extension.v3_policy_constraints',         # v3 Policy Constraints
    'certificate.extension.v3_inhibit_any_policy',         # v3 Inhibit Any Policy
    'certificate.extension.v3_freshest_crl',               # v3 Freshest CRL
]

x509_identifier_columns = [
    'ts',                                      # Timestamp of certificate capture
    'id',                                      # Unique ID for the connection
    'certificate.serial',                      # Certificate serial number
    'san.ip',                                  # SAN IP
]

x509_numeric_columns = [
    'certificate.key_length',                  # Key length
    'certificate.exponent',                    # Public exponent (if RSA)
]

x509_categorical_columns = [
    'certificate.version',                     # Certificate version    
    'certificate.key_alg',                     # Public key algorithm
    'certificate.sig_alg',                     # Signature algorithm
    'certificate.key_type',                    # Key type
    'basic_constraints.ca',                    # CA flag
]

x509_textual_columns = [
    'certificate.subject',                     # Certificate subject
    'certificate.issuer',                      # Certificate issuer
    'san.dns',                                 # Subject Alternative Name DNS
    'san.uri',                                 # SAN URI
    'san.email',                               # SAN email
]

# dns.log 标准字段 (Zeek官方文档 https://docs.zeek.org/en/master/logs/dns.html)
dns_columns = [
    'ts',                # Timestamp of the query
    'uid',               # Unique ID for the connection
    'id.orig_h',         # Source IP
    'id.orig_p',         # Source port
    'id.resp_h',         # Destination IP
    'id.resp_p',         # Destination port
    'proto',             # Transport-layer protocol (UDP/TCP)
    'trans_id',          # DNS transaction ID
    'rtt',               # Round-trip time
    'query',             # Query name
    'qclass',            # Query class
    'qclass_name',       # Query class name
    'qtype',             # Query type
    'qtype_name',        # Query type name
    'rcode',             # Response code
    'rcode_name',        # Response code name
    'AA',                # Authoritative Answer flag
    'TC',                # Truncated flag
    'RD',                # Recursion Desired flag
    'RA',                # Recursion Available flag
    'Z',                 # Reserved
    'answers',           # Answer RRs
    'TTLs',              # TTLs of answer RRs
    'rejected',          # True if rejected
    'saw_query',         # True if query observed
    'saw_reply',         # True if reply observed
    'query_length',      # Length of query
    'answer_length',     # Length of answer
    'authority_length',  # Length of authority section
    'additional_length', # Length of additional section
    'query_edns_version',# EDNS version in query
    'query_edns_do',     # DO flag in query
    'query_edns_udp_size', # UDP size in query
    'answer_edns_version',  # EDNS version in answer
    'answer_edns_do',       # DO flag in answer
    'answer_edns_udp_size',  # UDP size in answer

    # ✅ 项目自己新增的，非Zeek定义的，用于解析CNAME链
    # 现实 Zeek DNS 中很常见的是：
    # domain.com → CNAME → cdn.xxx.net → A → server IP
    'cname_chain',          # CNAME chain extracted from answers
]


dns_identifier_columns = [
    'ts',                # Timestamp of the query
    'uid',               # Unique ID for the connection
    'id.orig_h',         # Source IP
    'id.orig_p',         # Source port
    'id.resp_h',         # Destination IP
    'id.resp_p',         # Destination port
    'proto',             # Transport-layer protocol (UDP/TCP)
    'trans_id',          # DNS transaction ID
]

dns_numeric_columns = [
    'rtt',               # Round-trip time
    'query_length',      # Length of query
    'answer_length',     # Length of answer
    'authority_length',  # Length of authority section
    'additional_length', # Length of additional section
    'query_edns_udp_size', # UDP size in query    
    'answer_edns_udp_size'  # UDP size in answer
]

dns_categorical_columns = [
    'qclass',            # Query class
    'qtype',             # Query type
    'rcode',             # Response code
    'TC',                # Truncated flag
    'RD',                # Recursion Desired flag
    'RA',                # Recursion Available flag
    'rejected',          # True if rejected
    'saw_query',         # True if query observed
    'saw_reply',         # True if reply observed
]

dns_textual_columns = [
    'query',             # Query name    
    'qclass_name',       # Query class name
    'qtype_name',        # Query type name
    'rcode_name',        # Response code name
    'answers',           # Answer RRs
]

# http.log 标准字段 (Zeek官方文档 https://docs.zeek.org/en/master/logs/http.html)
http_columns = [
    'ts',                # Timestamp of the HTTP request/response
    'uid',               # Unique ID for the connection
    'id.orig_h',         # Source IP
    'id.orig_p',         # Source port
    'id.resp_h',         # Destination IP
    'id.resp_p',         # Destination port
    'trans_depth',       # Transaction depth (number of requests in connection)
    'method',            # HTTP method (GET, POST, etc.)
    'host',              # Host header
    'uri',               # Request URI
    'referrer',          # Referrer header
    'version',           # HTTP version
    'user_agent',        # User-Agent header
    'request_body_len',  # Length of request body in bytes
    'response_body_len', # Length of response body in bytes
    'status_code',       # HTTP response status code
    'status_msg',        # HTTP response status message
    'info_code',         # Informational response code (if any)
    'info_msg',          # Informational response message (if any)
    'tags',              # Zeek-assigned tags (malware, etc.)
    'username',          # Username if present in request
    'password',          # Password if present in request
    'proxied',           # True if request was proxied
    'orig_fuids',        # Originating file unique IDs
    'resp_fuids',        # Responding file unique IDs
    'resp_mime_types'    # MIME types of responses
]

http_identifier_columns = [
    'ts',                # Timestamp of the HTTP request/response
    'uid',               # Unique ID for the connection
    'id.orig_h',         # Source IP
    'id.orig_p',         # Source port
    'id.resp_h',         # Destination IP
    'id.resp_p',         # Destination port
]

http_numeric_columns = [
    'trans_depth',       # Transaction depth (number of requests in connection)
    'request_body_len',  # Length of request body in bytes
    'response_body_len', # Length of response body in bytes
]

http_categorical_columns = [
    'method',            # HTTP method (GET, POST, etc.)
    'version',           # HTTP version
    'status_code',       # HTTP response status code
    'resp_mime_types'    # MIME types of responses
]

http_textual_columns = [
    'host',              # Host header
    'uri',               # Request URI
    'referrer',          # Referrer header
    'user_agent',        # User-Agent header
]

# ftp.log 标准字段 (官方文档 https://docs.zeek.org/en/master/logs/ftp.html)
ftp_columns = [
    "ts",              # FTP 事件时间戳
    "uid",             # Zeek 连接唯一 ID（可关联 conn.log）
    "id.orig_h",       # 客户端 IP
    "id.orig_p",       # 客户端端口
    "id.resp_h",       # FTP 服务器 IP
    "id.resp_p",       # FTP 服务器端口（通常 21）
    "user",            # FTP 用户名
    "password",        # FTP 密码（明文时可见，可能为空）
    "command",         # FTP 命令（USER, PASS, RETR, STOR, LIST 等）
    "arg",             # FTP 命令参数（文件名、路径等）
    "file_size",       # 文件大小（字节，若可知）
    "reply_code",      # FTP 服务器返回的三位数字状态码
    "reply_msg",       # FTP 服务器返回的文本消息
    "mime_type",       # 代表文件被嗅探出的 MIME 类型（可选字段）。官方 ftp.log 会包含这个字段（未必每条都有，但它是官方支持的）

    # 以下是官方 ftp.log 输出的数据通道字段
    "data_channel.passive",  # bool
    "data_channel.orig_h",   # 数据通道源主机
    "data_channel.resp_h",   # 数据通道响应主机
    "data_channel.resp_p",   # 数据通道响应端口

    # 这是官方可能输出的文件 UID 字段（可选）
    "fuid",
]

ftp_identifier_columns = [
    "ts",          # FTP 事件时间戳
    "uid",         # Zeek 连接唯一 ID
    "id.orig_h",   # 客户端 IP
    "id.orig_p",   # 客户端端口
    "id.resp_h",   # FTP 服务器 IP
    "id.resp_p",   # FTP 服务器端口
]

ftp_numeric_columns = [
    "file_size",   # 传输文件大小（字节），注意经常缺失（None）
]

ftp_categorical_columns = [
    "command",                 # FTP 命令（USER, PASS, RETR, STOR 等）
    "mime_type",               # 文件 MIME 类型
    "data_channel.passive",    # 是否被动模式（bool → categorical）
    "reply_code",              # FTP 响应码（3 位整数）    
]

ftp_textual_columns = [
    "user",        # FTP 用户名
    "password",    # FTP 密码（若存在）
    "arg",         # 命令参数（文件名、路径）
    "reply_msg",   # 服务器返回文本
]

# mqtt通用的时间戳 + 五元组标识字段
mqtt_common_columns = [
    "ts",              # CONNECT/PUBLISH/SUBSCRIBE 报文时间戳
    "uid",             # Zeek 连接唯一标识（可关联 conn.log）
    "id.orig_h",       # 客户端 IP
    "id.orig_p",       # 客户端端口
    "id.resp_h",       # Broker IP
    "id.resp_p",       # Broker 端口
]

mqtt_connect_columns = mqtt_common_columns + [
    "proto_name",      # 协议名称（MQTT）
    "proto_version",   # MQTT 协议版本
    "client_id",       # MQTT 客户端 ID
    "connect_status",  # 连接结果
]

mqtt_publish_columns = mqtt_common_columns + [
    "from_client",     # 是否由客户端发送
    "retain",          # MQTT retain 标志
    "qos",             # QoS 等级
    "status",          # 解析状态
    "topic",           # MQTT 主题（业务语义核心）
    "payload",         # 消息内容
    "payload_len",     # payload 字节长度
]

mqtt_subscribe_columns = mqtt_common_columns + [
    "action",             # 控制报文类型（MQTT::SUBSCRIBE）
    "topics",             # 订阅主题列表，List[str]
    "qos_levels",         # 请求的 QoS 列表，List[int]
    "granted_qos_level",  # Broker 授权 QoS
    "ack",                # 是否收到 SUBACK
]

mqtt_log_columns = {
    "connect": mqtt_connect_columns,
    "publish": mqtt_publish_columns,
    "subscribe": mqtt_subscribe_columns,
}


# xxxx-flow.csv文件中，主要混合数据类型列的分类：
# ​​向量/列表字段​​：FlowMeter的各种向量字段，包含列表数据
# ​​证书链字段​​：SSL和X509的证书指纹和SAN字段，包含多个值
# ​​DNS列表字段​​：answers和TTLs字段包含列表数据
# ​​字符串标识字段​​：各种协议版本、密码套件、域名等字符串字段
# 这些字段在CSV中可能以字符串形式存储列表数据（如"[False, True]"），导致Pandas无法确定统一的数据类型。通过明确指定为字符串类型，可以避免警告并确保数据正确读取。

dtype_dict_in_flow_csv = {
    # FlowMeter 向量字段
    'flowmeter.packet_direction_vector': str,
    'flowmeter.packet_timestamp_vector': str,
    'flowmeter.packet_iat_vector': str,
    'flowmeter.packet_payload_size_vector': str,
    'flowmeter.tcp_packet_ack_vector': str,
    'flowmeter.tcp_packet_psh_vector': str,
    'flowmeter.tcp_packet_seqno_vector': str,
    'flowmeter.subflow_first_packet_index_vector': str,
    'flowmeter.bulk_first_packet_index_vector': str,
    'flowmeter.bulk_length_vector': str,
    'flowmeter.bulk_packet_index_vector': str,
    
    # SSL 列表字段
    'ssl.cert_chain_fps': str,
    'ssl.client_cert_chain_fps': str,
    'ssl.client_supported_versions': str,
    'ssl.server_supported_versions': str,
    'ssl.client_key_exchange_groups': str,
    'ssl.server_key_exchange_groups': str,
    'ssl.client_signature_algorithms': str,
    'ssl.server_signature_algorithms': str,
    'ssl.client_ec_point_formats': str,
    'ssl.server_ec_point_formats': str,
    
    # X509 证书字段（多个证书）
    'x509.cert0.san.dns': str,
    'x509.cert0.san.uri': str,
    'x509.cert0.san.email': str,
    'x509.cert0.san.ip': str,
    'x509.cert1.san.dns': str,
    'x509.cert1.san.uri': str,
    'x509.cert1.san.email': str,
    'x509.cert1.san.ip': str,
    'x509.cert2.san.dns': str,
    'x509.cert2.san.uri': str,
    'x509.cert2.san.email': str,
    'x509.cert2.san.ip': str,
    'x509.cert3.san.dns': str,
    'x509.cert3.san.uri': str,
    'x509.cert3.san.email': str,
    'x509.cert3.san.ip': str,
    
    # DNS 列表字段
    'dns.answers': str,
    'dns.TTLs': str,


    # 其他可能包含列表的字段
    'conn.history': str,
    'conn.tunnel_parents': str,
    'ssl.ssl_history': str,
    'ssl.next_protocol': str,
    
    # MAC地址字段
    'flowmeter.orig_l2_addr': str,
    'flowmeter.resp_l2_addr': str,
    
    # 字符串标识字段
    'conn.service': str,
    'ssl.version': str,
    'ssl.cipher': str,
    'ssl.curve': str,
    'ssl.server_name': str,
    'ssl.subject': str,
    'ssl.issuer': str,
    'ssl.ja3': str,
    'ssl.ja3s': str,
    'dns.query': str,
    'dns.qclass_name': str,
    'dns.qtype_name': str,
    'dns.rcode_name': str,
    'x509.cert0.certificate.subject': str,
    'x509.cert0.certificate.issuer': str,
    'x509.cert0.certificate.key_alg': str,
    'x509.cert0.certificate.sig_alg': str,
    'x509.cert0.certificate.key_type': str,
    'x509.cert1.certificate.subject': str,
    'x509.cert1.certificate.issuer': str,
    'x509.cert1.certificate.key_alg': str,
    'x509.cert1.certificate.sig_alg': str,
    'x509.cert1.certificate.key_type': str,
    'x509.cert2.certificate.subject': str,
    'x509.cert2.certificate.issuer': str,
    'x509.cert2.certificate.key_alg': str,
    'x509.cert2.certificate.sig_alg': str,
    'x509.cert2.certificate.key_type': str,
    'x509.cert3.certificate.subject': str,
    'x509.cert3.certificate.issuer': str,
    'x509.cert3.certificate.key_alg': str,
    'x509.cert3.certificate.sig_alg': str,
    'x509.cert3.certificate.key_type': str,
    'x509.cert4.certificate.subject': str,
    'x509.cert4.certificate.issuer': str,
    'x509.cert4.certificate.key_alg': str,
    'x509.cert4.certificate.sig_alg': str,
    'x509.cert4.certificate.key_type': str,
    # =========================
    # HTTP 相关字段（补充）
    # =========================
    'http.host': str,
    'http.uri': str,
    'http.referrer': str,
    'http.user_agent': str,

    'http.tags': str,            # set[str]
    'http.username': str,
    'http.password': str,

    'http.orig_fuids': str,      # set[str]
    'http.resp_fuids': str,      # set[str]
    'http.resp_mime_types': str, # set[str]

    # =========================
    # MQTT 相关字段（新增）
    # =========================
    # MQTT publish
    'mqtt.topic': str,
    'mqtt.payload': str,
    'mqtt.qos': str,
    'mqtt.status': str,
    'mqtt.from_client': str,
    'mqtt.retain': str,

    # MQTT subscribe
    'mqtt.topics': str,        # List[str]
    'mqtt.qos_levels': str,    # List[int]
    'mqtt.action': str,
    'mqtt.ack': str,

    # MQTT connect
    'mqtt.client_id': str,
    'mqtt.proto_name': str,
    'mqtt.proto_version': str,
    'mqtt.connect_status': str,    

    # =========================
    # FTP 相关字段（新增）
    # =========================     
    'ftp.command': str,
    'ftp.arg': str,
    'ftp.user': str,
    'ftp.password': str,
    'ftp.reply_msg': str,
    'ftp.mime_type': str,
    'ftp.fuid': str,

    # data_channel 是结构体，CSV 中必然是字符串
    'ftp.data_channel.passive': str,
    'ftp.data_channel.orig_h': str,
    'ftp.data_channel.resp_h': str,
    'ftp.data_channel.resp_p': str,       
}

def add_prefix(prefix, columns):
    return [f"{prefix}.{col}" for col in columns]

all_zeek_numeric_columns = (
    add_prefix("conn", conn_numeric_columns) +
    add_prefix("flowmeter", flowmeter_numeric_columns) +
    add_prefix("dns", dns_numeric_columns) +
    add_prefix("x509", x509_numeric_columns) +
    add_prefix("http", http_numeric_columns) +
    add_prefix("ftp", ftp_numeric_columns)
)