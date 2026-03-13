import os, sys
import argparse
from flow_node_builder import FlowNodeBuilder
from session_graph_builder import SessionGraphBuilder
import logging

# 添加../utils目录到Python搜索路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)

import config_manager as ConfigManager
from logging_config import setup_preset_logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(
        description="Construct multi-flow session graphs from dataset filepaths defined in src/utils/config.cfg."
    )

    # -------- sampling --------
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        required=True,
        help="Session-level sampling ratio (0,1]. None means no sampling."
    )

    parser.add_argument(
        "--downsample_benign_only",
        type=lambda x: x.lower() == "true",
        required=True
    )

    # -------- flow filtering --------
    # DNS：53端口和dns服务的流量也被过滤掉，因为 DNS 已经通过域名反查特征被充分利用
    # 而且这个数据并不做 DNS 流量分类任务，比如检测 DNS 隧道等。
    # NTP: 属于基础时间同步服务，行为模式固定、周期性强，与攻击流量区分度低。非研究重点协议，且在图中易形成高频噪声节点。
    # DHCP：67/68，广播型、一次性、无行为区分度
    # ARP	二层地址解析，和攻击语义无关
    # mDNS / LLMNR	本地名称解析，流量碎片化
    # NBNS	Windows 广播噪声
    # exclude_ports: [53, 67, 68, 123]
    # exclude_services: ["dns", "dhcp", "ntp", "arp", "mdns", "llmnr", "nbns"]
    parser.add_argument(
        "--exclude-ports",
        type=str,
        default="53,67,68,123",
        help="Comma-separated port list to exclude, e.g. 53,67,68,123"
    )

    parser.add_argument(
        "--exclude-services",
        type=str,
        default="dns,dhcp,ntp,arp,mdns,llmnr,nbns",
        help="Comma-separated service list to exclude, e.g. dns,dhcp,ntp,arp,mdns,llmnr,nbns"
    )
    
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
        help="Global dataset seed. Controls both session-level sampling and random dataset splitting."
    )

    # -------- dataset split --------
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["csv", "random"],
        default="random",
        help="Dataset split mode: csv (use split column in all_split_session.csv) or random"
    )

    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/val/test split ratio, e.g. 0.8,0.1,0.1 (only used when split-mode=random)"
    )    

    try:
        dataset_dir = ConfigManager.read_plot_data_path_config()
        
        # 从配置文件读取线程数
        thread_count = ConfigManager.read_thread_count_config()
        logger.info(f"配置了内核线程数 = {thread_count}")

        session_label_id_map = ConfigManager.read_session_label_id_map()
        logger.info(f"配置了 session label string-to-id mapping: {session_label_id_map}")

        # 从命令行接受如下参数
        args = parser.parse_args()
        if not (0.0 < args.sampling_ratio <= 1.0):
            parser.error("--sampling-ratio must be in (0, 1].")
            return

        sampling_ratio = float(args.sampling_ratio)
        if sampling_ratio > 1.0 - 1e-6:
            sampling_ratio = None

        downsample_benign_only = args.downsample_benign_only

        exclude_ports = (
            [int(p) for p in args.exclude_ports.split(",")]
            if args.exclude_ports else None
        )

        # 注意：
        # conn.service == "-" 表示 Zeek 未识别协议
        # 不应被 exclude_services 过滤
        exclude_services = (
            [s.strip().lower() for s in args.exclude_services.split(",")]
            if args.exclude_services else None
        )

        logger.info(
            f"命令行参数: sampling_ratio={sampling_ratio}, "
            f"exclude_ports={exclude_ports}, "
            f"exclude_services={exclude_services}"
        )

        # parse split ratio
        try:
            split_ratio = tuple(float(x) for x in args.split_ratio.split(","))
            if len(split_ratio) != 3 or abs(sum(split_ratio) - 1.0) > 1e-6:
                raise ValueError
        except Exception:
            parser.error("--split-ratio must be three floats summing to 1.0, e.g. 0.8,0.1,0.1")
            
        # 1. 先处理flow数据
        logger.info("开始处理 Flow 数据文件 ...")
        merged_flow_path = os.path.join(dataset_dir, "all_embedded_flow.csv")
        if not os.path.exists(merged_flow_path):
            raise FileNotFoundError(f"合并后的 Flow 数据文件不存在: {merged_flow_path}")
        else:
            logger.info(f"合并后的 Flow 数据文件 {merged_flow_path} 已经存在!")
        
        enabled_flow_node_views = ConfigManager.read_enabled_flow_node_views_config()
        logger.info(f"Enabled flow node views = {enabled_flow_node_views}")
        max_packet_sequence_length = ConfigManager.read_max_packet_sequence_length()
        logger.info(f"max_packet_sequence_length = {max_packet_sequence_length}，长度不足的packet sequences会做padding，超长的予以截断")
        text_encoder_name, max_text_length = ConfigManager.read_text_encoder_config()        
        logger.info(f"text_encoder_name = {text_encoder_name}, max_text_length = {max_text_length}")
        flow_node_builder = FlowNodeBuilder(flow_csv_path = merged_flow_path,
                                            session_label_id_map = session_label_id_map,
                                            max_packet_sequence_length = max_packet_sequence_length,
                                            text_encoder_name = text_encoder_name,
                                            max_text_length = max_text_length,
                                            thread_count = thread_count,
                                            enabled_views = enabled_flow_node_views,
                                            exclude_ports=exclude_ports,
                                            exclude_services=exclude_services,
                                            storage_mode="dict",
                                            )
        # storage_mode 有 "dict" 或 "offset" 两种：
        # * "dict" mode 会把整个all_embedded_flow.csv文件加载到内存中，加快扫描速度；
        # * "offset" mode 避免加载整个all_embedded_flow.csv文件，只是需要的时候到外存取数据访问。
        #
        # 实验中发现 “offset” mode可以构建完 categorical vocab，也可以计算完 numeric stats。
        # 但是，进入 process_sessions_parallel阶段就会卡死，原因是 session 中的 flow_uid 是无序的。这意味着：
        # * 线程 1 → seek 到 1GB 位置
        # * 线程 2 → seek 到 200MB 位置
        # * 线程 3 → seek 到 1.5GB 位置
        # * 线程 4 → seek 到 20MB 位置
        # ...
        # 硬盘疯狂来回跳。
        # * 如果是机械硬盘 → 基本等于锁死
        # * 如果是 SATA SSD → IOPS 被打爆
        # * 即使 NVMe → 20 线程 + Python 解析 CSV 也会炸

        # 2. 构建 session graph
        logger.info("开始处理 Session 数据文件 ...")
        merged_session_path = os.path.join(dataset_dir, "all_split_session.csv")
        if not os.path.exists(merged_session_path):
            raise FileNotFoundError(f"合并后的 Session 数据文件不存在: {merged_session_path}")
        else: 
            logger.info(f"合并后的 Session 数据文件 {merged_session_path} 已经存在!")

        tag_parts = []
        # sampling
        if sampling_ratio is not None:
            tag_parts.append(f"sampled_p{sampling_ratio}")

        # flow filtering
        if exclude_ports:
            tag_parts.append("port_" + "_".join(map(str, sorted(exclude_ports))))

        if exclude_services:
            tag_parts.append("svc_" + "_".join(sorted(exclude_services)))

        graph_tag = "__".join(tag_parts) if tag_parts and len(tag_parts) > 0 else "full"
        dump_file = os.path.join(dataset_dir, f"all_session_graph__{graph_tag}")
        dump_bin_path = dump_file + ".bin"
        dump_info_path = dump_file + "_info.pkl"

        logger.info(f"正在构建统一的 session graph 到输出文件路径 bin_path={dump_bin_path}， 和 info_path={dump_info_path} ...")
        concurrent_flow_iat_threshold = ConfigManager.read_concurrent_flow_iat_threshold()
        sequential_flow_iat_threshold = ConfigManager.read_sequential_flow_iat_threshold()    
        logger.info(f"使用配置参数: concurrent_flow_iat_threshold={concurrent_flow_iat_threshold}, sequential_flow_iat_threshold={sequential_flow_iat_threshold}")
        max_nodes_per_flow_relation_graph = ConfigManager.read_max_nodes_per_flow_relation_graph()
        logger.info(f"使用配置参数: max_nodes_per_flow_relation_graph={max_nodes_per_flow_relation_graph}")
        assert dump_bin_path.endswith(".bin")
        assert dump_info_path.endswith("_info.pkl")
        builder = SessionGraphBuilder(flow_node_builder, merged_session_path, dump_bin_path, dump_info_path, 
                                      concurrent_flow_iat_threshold= concurrent_flow_iat_threshold,
                                      sequential_flow_iat_threshold=sequential_flow_iat_threshold,
                                      session_label_id_map = session_label_id_map, 
                                      thread_count = thread_count,
                                      sampling_ratio=sampling_ratio,
                                      downsample_benign_only=downsample_benign_only,
                                      seed=args.dataset_seed,
                                      split_mode=args.split_mode,
                                      split_ratio=split_ratio,
                                      max_nodes_per_flow_relation_graph=max_nodes_per_flow_relation_graph,
                                      )

        flow_node_builder.close_flow_file()        
        logger.info(f"[SUCCESS] 成功构建 unified session graph! 输出文件: bin_path={dump_bin_path}， 和 info_path={dump_info_path}")

    except Exception as e:
        import traceback
        logger.error(f"构建 graph 失败: {str(e)}")
        logger.error("详细错误信息:")
        # 使用 traceback.format_exc() 获取完整的堆栈跟踪信息
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
