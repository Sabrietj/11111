#!/usr/bin/env python3
"""
config.cfg 快速测试用例
用于快速验证配置文件调整后的核心功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_integration import get_config_section, get_global_config, refresh_config

def quick_test():
    """快速测试核心功能"""
    print("🔍 config.cfg 快速测试")
    print("-" * 40)
    
    try:
        # 1. 测试基本配置读取
        print("1. 测试基本配置读取...")
        path_config = get_config_section('PATH')
        session_config = get_config_section('SESSION')
        print(f"   ✓ 数据集路径: {path_config['path_to_dataset']}")
        print(f"   ✓ 会话模式: {session_config['session_tuple_mode']}")
        
        # 2. 测试数据集切换
        print("\n2. 测试数据集切换功能...")
        from config_loader import get_active_dataset_config
        active_config = get_active_dataset_config('config.cfg')
        current_dataset = active_config.get('dataset_name', 'Unknown')
        print(f"   ✓ 当前数据集: {current_dataset}")
        
        # 3. 验证配置完整性
        print("\n3. 验证关键配置...")
        required_keys = {
            'path_to_dataset': path_config.get('path_to_dataset'),
            'session_tuple_mode': session_config.get('session_tuple_mode'),
            'concurrent_flow_iat_threshold': session_config.get('concurrent_flow_iat_threshold'),
            'sequential_flow_iat_threshold': session_config.get('sequential_flow_iat_threshold')
        }
        
        all_present = all(v is not None and v != 'AUTO_FILL' for v in required_keys.values())
        if all_present:
            print("   ✓ 所有关键配置都已正确填充")
        else:
            missing = [k for k, v in required_keys.items() if v is None or v == 'AUTO_FILL']
            print(f"   ✗ 缺少配置: {missing}")
            return False
        
        # 4. 显示可用数据集
        print("\n4. 可用数据集:")
        config = get_global_config()
        dataset_sections = [s for s in config.sections() 
                           if s not in ['GENERAL', 'PATH', 'SESSION', 'DOMAIN_HIERARCHY', 
                                       'MODEL_PARAMS', 'TRAINING_MODES', 'MODEL_ARCHITECTURE',
                                       'TRAINING_PARAMS', 'EXPERIMENT_MANAGER', 'METRICS']]
        for dataset in dataset_sections:
            marker = "👉" if dataset == current_dataset else "  "
            print(f"   {marker} {dataset}")
        
        print("\n✅ config.cfg 测试通过！配置调整成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False

def show_usage_tip():
    """显示使用提示"""
    print("\n💡 使用提示:")
    print("   切换数据集: 编辑 config.cfg 中的 ACTIVE_DATASET 参数")
    print("   例如: ACTIVE_DATASET = USTC-TFC2016")
    print("   重启程序后生效")

if __name__ == "__main__":
    success = quick_test()
    if success:
        show_usage_tip()
    sys.exit(0 if success else 1)