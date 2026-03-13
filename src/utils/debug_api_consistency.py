#!/usr/bin/env python3
"""
调试API一致性问题
"""

import configparser
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_integration import get_config_section, refresh_config

def debug_api_consistency():
    """调试API一致性问题"""
    config_path = "config.cfg"
    
    print("调试API一致性问题...")
    
    # 传统方式
    print("\n1. 传统ConfigParser方式:")
    traditional_config = configparser.ConfigParser()
    traditional_config.read(config_path, encoding='utf-8')
    
    print(f"PATH.path_to_dataset: {traditional_config['PATH']['path_to_dataset']}")
    print(f"SESSION.session_tuple_mode: {traditional_config['SESSION']['session_tuple_mode']}")
    
    # 新方式
    print("\n2. 新的config_integration方式:")
    refresh_config()  # 确保最新配置
    
    new_path_config = get_config_section('PATH')
    new_session_config = get_config_section('SESSION')
    
    print(f"PATH.path_to_dataset: {new_path_config['path_to_dataset']}")
    print(f"SESSION.session_tuple_mode: {new_session_config['session_tuple_mode']}")
    
    # 比较结果
    print("\n3. 比较结果:")
    path_match = traditional_config['PATH']['path_to_dataset'] == new_path_config['path_to_dataset']
    session_match = traditional_config['SESSION']['session_tuple_mode'] == new_session_config['session_tuple_mode']
    
    print(f"PATH配置匹配: {path_match}")
    print(f"SESSION配置匹配: {session_match}")
    
    if not path_match:
        print(f"  传统PATH: {traditional_config['PATH']['path_to_dataset']}")
        print(f"  新PATH: {new_path_config['path_to_dataset']}")
    
    if not session_match:
        print(f"  传统SESSION: {traditional_config['SESSION']['session_tuple_mode']}")
        print(f"  新SESSION: {new_session_config['session_tuple_mode']}")

if __name__ == "__main__":
    debug_api_consistency()