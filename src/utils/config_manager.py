# -*- coding: utf-8 -*-
#config_manager.py

import os
import glob
from configparser import ConfigParser
import json

def get_config_file():
    # 获取当前文件（config_manager.py）所在的目录 → 即 src/utils
    current_dir = os.path.dirname(__file__)

    # 在当前目录下查找所有 .cfg 文件
    pattern = os.path.join(current_dir, '*.cfg')
    cfg_files = glob.glob(pattern)

    if cfg_files:
        return cfg_files[0]

    # 如果没找到，报错并打印调试信息
    raise FileNotFoundError(
        f"No .cfg file found in {current_dir}\n"
        f"Looked in: {pattern}\n"
        f"Files present: {os.listdir(current_dir) if os.path.exists(current_dir) else 'Directory does not exist'}"
    )

def read_thread_count_config():
    """读取线程数配置"""
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)
    
    # 在[GENERAL]或新建的[THREADING] section中读取线程数
    if config.read(name_of_config):
        try:
            thread_count = config.getint('GENERAL', 'thread_count', fallback=4)
            return thread_count
        except Exception as e:
            try:
                thread_count = config.getint('THREADING', 'thread_count', fallback=4)
            except Exception as e:
                raise ValueError("Config path has bad format") from e
    else:
        raise IOError("Cannot read config file")
    

def read_dataset_path_config():
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)

    # Read the configuration file
    if config.read(name_of_config,encoding="utf-8"):
        try:
            # 检查是否启用了数据集切换
            if 'GENERAL' in config and 'ACTIVE_DATASET' in config['GENERAL']:
                active_dataset = config['GENERAL']['ACTIVE_DATASET']
                # 如果PATH段是AUTO_FILL，从激活的数据集配置中获取
                if config.get('PATH', 'path_to_dataset') == 'AUTO_FILL' and active_dataset in config:
                    return config[active_dataset].get('path_to_dataset', './dataset/default')
            
            dataset_path = config.get('PATH', 'path_to_dataset')
            return dataset_path
        except Exception as e:
            raise ValueError("Config path has bad format") from e
    else:
        raise IOError("Cannot read config file")

def read_plot_data_path_config():
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)

    # Read the configuration file
    if config.read(name_of_config,encoding="utf-8"):
        try:
            # 检查是否启用了数据集切换
            if 'GENERAL' in config and 'ACTIVE_DATASET' in config['GENERAL']:
                active_dataset = config['GENERAL']['ACTIVE_DATASET']
                # 如果PATH段是AUTO_FILL，从激活的数据集配置中获取
                if config.get('PATH', 'plot_data_path') == 'AUTO_FILL' and active_dataset in config:
                    return config[active_dataset].get('plot_data_path', './processed_data/default')
            
            plot_data_path = config.get('PATH', 'plot_data_path')
            return plot_data_path
        except Exception as e:
            raise ValueError("Config path has bad format") from e
    else:
        raise IOError("Cannot read config file")
    

def read_session_tuple_mode():
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)

    # Read the configuration file
    if config.read(name_of_config):
        try:
            # 检查是否启用了数据集切换
            if 'GENERAL' in config and 'ACTIVE_DATASET' in config['GENERAL']:
                active_dataset = config['GENERAL']['ACTIVE_DATASET']
                # 如果SESSION段是AUTO_FILL，从激活的数据集配置中获取
                if config.get('SESSION', 'session_tuple_mode') == 'AUTO_FILL' and active_dataset in config:
                    return config[active_dataset].get('session_tuple_mode', 'srcIP_dstIP_dstPort_proto')
            
            session_tuple_mode = config.get('SESSION', 'session_tuple_mode', fallback='srcIP_dstIP_dstPort_proto')
            return session_tuple_mode
        except Exception as e:
            raise ValueError("Config path has bad format") from e
    else:
        raise IOError("Cannot read config file")


def read_concurrent_flow_iat_threshold():
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)

    # Read the configuration file
    if config.read(name_of_config):
        try:
            # 检查是否启用了数据集切换
            if 'GENERAL' in config and 'ACTIVE_DATASET' in config['GENERAL']:
                active_dataset = config['GENERAL']['ACTIVE_DATASET']
                # 如果SESSION段是AUTO_FILL，从激活的数据集配置中获取
                if config.get('SESSION', 'concurrent_flow_iat_threshold') == 'AUTO_FILL' and active_dataset in config:
                    return float(config[active_dataset].get('concurrent_flow_iat_threshold', '1.0'))
            
            concurrent_flow_iat_threshold = config.getfloat('SESSION', 'concurrent_flow_iat_threshold', fallback=1.0)
            return concurrent_flow_iat_threshold
        except Exception as e:
            raise ValueError("Config path has bad format") from e
    else:
        raise IOError("Cannot read config file")


def read_sequential_flow_iat_threshold():
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)

    # Read the configuration file
    if config.read(name_of_config):
        try:
            # 检查是否启用了数据集切换
            if 'GENERAL' in config and 'ACTIVE_DATASET' in config['GENERAL']:
                active_dataset = config['GENERAL']['ACTIVE_DATASET']
                # 如果SESSION段是AUTO_FILL，从激活的数据集配置中获取
                if config.get('SESSION', 'sequential_flow_iat_threshold') == 'AUTO_FILL' and active_dataset in config:
                    return float(config[active_dataset].get('sequential_flow_iat_threshold', '1.0'))
            
            sequential_flow_iat_threshold = config.getfloat('SESSION', 'sequential_flow_iat_threshold', fallback=1.0)
            return sequential_flow_iat_threshold
        except Exception as e:
            raise ValueError("Config path has bad format") from e
    else:
        raise IOError("Cannot read config file")


def get_config_parser():
    """获取配置解析器实例"""
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)
    config.read(name_of_config, encoding='utf-8')
    return config

    
def read_session_label_id_map(dataset=None):
    """读取会话标签ID映射配置"""
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)

    # 读取配置文件
    if config.read(name_of_config):
        try:
            if dataset is None:
                # 检查是否启用了活跃数据集切换
                if 'GENERAL' in config and 'ACTIVE_DATASET' in config['GENERAL']:
                    active_dataset = config['GENERAL']['ACTIVE_DATASET']
                    # 如果SESSION段是AUTO_FILL，从激活的数据集配置中获取
                    if config.get('SESSION', 'session_label_id_map') == 'AUTO_FILL' and active_dataset in config:
                        label_mapping_str = config[active_dataset].get('session_label_id_map', 'benign:0, background:1, mixed:2, malicious:3')
                    else:
                        # 获取配置字符串
                        label_mapping_str = config.get('SESSION', 'session_label_id_map', 
                                                    fallback='benign:0, background:1, mixed:2, malicious:3')
                else:
                    # 获取配置字符串
                    label_mapping_str = config.get('SESSION', 'session_label_id_map', 
                                                fallback='benign:0, background:1, mixed:2, malicious:3')
            else:
                # 从指定数据集的配置段获取
                if config.get('SESSION', 'session_label_id_map') == 'AUTO_FILL' and dataset in config:
                    label_mapping_str = config[dataset].get('session_label_id_map', 'benign:0, background:1, mixed:2, malicious:3')
                else:
                    raise ValueError(f"Dataset section '{dataset}' not found in config file")
            
            # 解析标签映射
            label_id_map = {}
            for item in label_mapping_str.split(','):
                item = item.strip()
                if ':' in item:
                    label_name, label_id = item.split(':', 1)
                    label_id_map[label_name.strip().lower()] = int(label_id.strip())
            
            return label_id_map
            
        except Exception as e:
            raise ValueError("Config session_label_id_map has bad format") from e
    else:
        raise IOError("Cannot read config file")


def read_max_packet_sequence_length():
    """读取会话标签ID映射配置"""
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)

    # 读取配置文件
    if config.read(name_of_config):
        try:
            # 获取配置字符串
            max_packet_sequence_length = config.getint('MODEL_PARAMS', 'max_packet_sequence_length', 
                                         fallback='512')
            
            return max_packet_sequence_length
            
        except Exception as e:
            raise ValueError("Config max_packet_sequence_length has bad format") from e
    else:
        raise IOError("Cannot read config file")


def read_max_nodes_per_flow_relation_graph():
    """读取会话标签ID映射配置"""
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)

    # 读取配置文件
    if config.read(name_of_config):
        try:
            # 获取配置字符串
            max_nodes_per_flow_relation_graph = config.getint('MODEL_PARAMS', 'max_nodes_per_flow_relation_graph', 
                                         fallback='512')
            
            return max_nodes_per_flow_relation_graph
            
        except Exception as e:
            raise ValueError("Config max_nodes_per_flow_relation_graph has bad format") from e
    else:
        raise IOError("Cannot read config file")


def read_text_encoder_config():
    """读取流量中的明文文本的编码器配置"""
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)

    # 读取配置文件
    if config.read(name_of_config):
        try:
            # 获取配置字符串
            text_encoder_name = config.get("MODEL_PARAMS", "text_encoder_name", fallback="bert-base-uncased")
            max_text_length = config.getint("MODEL_PARAMS", "max_text_length", fallback=64)

            return text_encoder_name, max_text_length
        
        except Exception as e:
            raise ValueError("Config text_encoder_name or max_text_length has bad format") from e        
    else:
        raise IOError("Cannot read config file")

def read_enabled_flow_node_views_config():
    name_of_config = get_config_file()
    config = ConfigParser(allow_no_value=True)

    if config.read(name_of_config, encoding="utf-8"):
        try:
            raw = config.get('MODEL_PARAMS', 'enabled_flow_node_views')

            # 将 JSON 字符串转换为 dict
            views = json.loads(raw)

            if not isinstance(views, dict):
                raise ValueError("enabled_flow_node_views 必须是字典形式")

            return views

        except Exception as e:
            raise ValueError("Config enabled_flow_node_views has bad format: " + str(e))
    else:
        raise IOError("Cannot read config file")
    
    
def get_folders_name(file_path):
    # Return a list of folder names in the given path
    return [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]

# Example usage
if __name__ == '__main__':
    dataset_path = read_dataset_path_config()
    print(f"The dataset path is {dataset_path}")
    plot_data_path = read_plot_data_path_config()
    print(f"The plot data path is {plot_data_path}")
    folder_names = get_folders_name(dataset_path)
    print(f"Folder names under the dataset path: {folder_names}")
    max_packet_sequence_length = read_max_packet_sequence_length()
    print(f"Max number of tokens that can be processed by Transformer: {max_packet_sequence_length}")