# 数据集切换使用指南

## 概述

新的数据集切换系统允许您通过简单的配置修改来切换不同的数据集，无需修改代码或重新注释/取消注释大量配置项。

## 使用方法

### 1. 切换数据集

只需编辑 `src/utils/config.cfg` 文件，找到 `[GENERAL]` 段中的 `ACTIVE_DATASET` 参数：

```ini
[GENERAL]
# 处理线程数
thread_count = 20

# 数据集激活配置 - 只需修改这里
ACTIVE_DATASET = CIC-IDS-2017
# 其他选择: ACTIVE_DATASET = CIC-AndMal2017
# 其他选择: ACTIVE_DATASET = USTC-TFC2016
# 其他选择: ACTIVE_DATASET = CIC-IoT-2023
# 其他选择: ACTIVE_DATASET = CTU-13
```

### 2. 可用数据集

目前支持的数据集包括：

- **CIC-IDS-2017**: 默认数据集，使用四元组会话模式
- **CIC-AndMal2017**: 恶意软件数据集，使用二元组会话模式  
- **USTC-TFC2016**: 流量分类数据集，使用一元组会话模式
- **CIC-IoT-2023**: IoT设备数据集，使用二元组会话模式
- **CTU-13**: 流量分析数据集，使用二元组会话模式

### 3. 代码中使用

新系统完全兼容现有代码。现有代码无需修改即可使用：

```python
# 现有的代码仍然可以正常工作
from configparser import ConfigParser

config = ConfigParser()
config.read('src/utils/config.cfg')

# 或者使用新的便捷函数
from src.utils.config_integration import get_config_section

path_config = get_config_section('PATH')
session_config = get_config_section('SESSION')

print(f"数据集路径: {path_config['path_to_dataset']}")
print(f"会话模式: {session_config['session_tuple_mode']}")
```

## 工作原理

1. **自动检测**: 系统读取 `ACTIVE_DATASET` 参数
2. **配置查找**: 在配置文件中查找对应的数据集配置段
3. **自动填充**: 将数据集特定配置自动填充到 `PATH` 和 `SESSION` 段
4. **保持兼容**: 现有代码继续使用原有的 `PATH` 和 `SESSION` 段，无需修改

## 新增功能

### 配置便捷函数

```python
from src.utils.config_integration import get_config_section, get_active_dataset_config

# 获取当前激活的数据集完整配置
dataset_config = get_active_dataset_config('config.cfg')
print(f"当前数据集: {dataset_config['dataset_name']}")

# 获取特定配置段
path_config = get_config_section('PATH')
session_config = get_config_section('SESSION')

# 在运行时刷新配置（如果配置文件被修改）
from src.utils.config_integration import refresh_config
refresh_config()
```

### 测试工具

提供了测试脚本来验证配置：

```bash
# 测试基本配置功能
cd src/utils
python test_config.py

# 测试数据集切换功能
python test_dataset_switch.py

# 查看使用示例
python dataset_switch_example.py
```

## 优势

1. **简单易用**: 只需修改一个参数
2. **零代码修改**: 现有代码无需任何改动
3. **避免错误**: 不需要手动注释/取消注释大量配置
4. **易于扩展**: 添加新数据集只需在配置文件中添加新的段
5. **完全兼容**: 保持所有现有API不变

## 添加新数据集

要添加新的数据集，只需在配置文件中添加新的配置段：

```ini
[NEW-DATASET]
dataset_name = NEW-DATASET
path_to_dataset = ./dataset/NEW-DATASET
plot_data_path = ./processed_data/NEW-DATASET
session_tuple_mode = srcIP_dstIP
session_label_id_map = benign:0, malicious:1
concurrent_flow_iat_threshold = 1
sequential_flow_iat_threshold = 1.0
```

然后在 `ACTIVE_DATASET` 处添加注释选项即可使用。

## 注意事项

1. 修改 `ACTIVE_DATASET` 后，重启程序生效
2. 如果在同一进程中需要立即生效，调用 `refresh_config()`
3. 确保数据集路径和文件存在
4. 标签映射格式为 `label1:id1,label2:id2,...`