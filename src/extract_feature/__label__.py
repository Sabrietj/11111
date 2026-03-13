# -*- coding: utf-8 -*-

import label_log
import sys
import os
import argparse

# 添加../utils目录到Python搜索路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
from config_manager import read_dataset_path_config
from logging_config import setup_preset_logging
import logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.INFO)


def attach_label_to_dataset(dataset_path, conn_filename="conn.log"):
    # 检查dataset_path目录下是否有指定的日志文件
    conn_log_path = os.path.join(dataset_path, conn_filename)
    if os.path.exists(conn_log_path):
        print(f"Found {conn_filename} file, start processing {dataset_path}")
        label_log.label_conn_log(dataset_path, conn_filename=conn_filename)

    # 遍历dataset_path的一级子目录
    for dir_name in os.listdir(dataset_path):
        path_to_single = os.path.join(dataset_path, dir_name)
        if os.path.isdir(path_to_single):
            # 检查一级目录下是否有指定的日志文件
            conn_log_path = os.path.join(path_to_single, conn_filename)
            if os.path.exists(conn_log_path):
                print(f"Found {conn_filename} file, start processing {path_to_single}")
                label_log.label_conn_log(path_to_single, conn_filename=conn_filename)
            else:
                # 没有，则遍历dataset_path的二级子目录
                for sub_dir in os.listdir(path_to_single):
                    sub_path = os.path.join(path_to_single, sub_dir)
                    if os.path.isdir(sub_path):
                        sub_conn_log = os.path.join(sub_path, conn_filename)
                        if os.path.exists(sub_conn_log):
                            print(f"Found {conn_filename} file, start processing {sub_path}")
                            label_log.label_conn_log(sub_path, two_level_dataset_folder=True, conn_filename=conn_filename)


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Attach labels to dataset based on connection logs.')
    parser.add_argument('--conn_filename', type=str, default="conn.log",
                        help='Name of the connection log file (default: conn.log)')
    
    args = parser.parse_args()
    
    dataset_path = read_dataset_path_config()
    if dataset_path == -1:
        raise ValueError

    attach_label_to_dataset(dataset_path, args.conn_filename)
    