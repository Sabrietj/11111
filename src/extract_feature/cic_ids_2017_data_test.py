import pandas as pd
import os
import time
from collections import Counter

# ================= 配置区域 =================
# 这里填入你的具体文件路径 (注意前面的 r 不能删，用于处理 Windows 反斜杠)
TARGET_FILE = r"D:\Document\GitHub\code-multiview-network-traffic-classification-model\processed_data\CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto\all_flow.csv"
# ===========================================

def main():
    print(f"准备读取文件: {TARGET_FILE}")
    
    if not os.path.exists(TARGET_FILE):
        print("错误: 找不到文件，请检查路径是否正确。")
        return

    start_time = time.time()
    label_counter = Counter()
    total_rows = 0
    
    print("开始统计 (使用分块读取模式)...")
    print("-" * 60)

    try:
        # 分块读取，防止内存溢出
        # encoding='cp1252' 或 'utf-8'，CIC-IDS-2017通常是utf-8，但有时会有特殊字符
        with pd.read_csv(
            TARGET_FILE, 
            chunksize=100000, 
            low_memory=False, 
            encoding='utf-8', 
            on_bad_lines='skip' 
        ) as reader:
            
            for i, chunk in enumerate(reader):
                # 1. 修复列名空格问题 (" Label" -> "Label")
                chunk.columns = chunk.columns.str.strip()
                
                if 'Label' in chunk.columns:
                    # 2. 统计标签
                    label_counts = chunk['Label'].value_counts().to_dict()
                    label_counter.update(label_counts)
                    
                    current_rows = len(chunk)
                    total_rows += current_rows
                    
                    # 每处理 50万行 打印一次进度
                    if (i + 1) % 5 == 0:
                        print(f"  -> 已扫描 {total_rows:,} 行...")
                else:
                    print("  -> 警告: 当前块未找到 'Label' 列")

    except Exception as e:
        print(f"\n发生错误: {e}")
        return

    # 统计结束，输出结果
    duration = time.time() - start_time
    total_samples = sum(label_counter.values())
    
    print("\n" + "=" * 60)
    print(f"统计完成！耗时: {duration:.2f} 秒")
    print(f"文件路径: {os.path.basename(TARGET_FILE)}")
    print(f"总数据量: {total_samples:,}")
    print("=" * 60)
    print(f"{'Label 名称':<40} {'数量':<15} {'占比':<10}")
    print("-" * 60)
    
    # 打印详细分布
    for label, count in label_counter.most_common():
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"{label:<40} {count:<15,} {percentage:.2f}%")
    
    print("-" * 60)

    # 简单的保存结果到当前目录
    output_csv = "cic2017_stats.csv"
    res_df = pd.DataFrame(label_counter.most_common(), columns=['Label', 'Count'])
    res_df['Percentage'] = (res_df['Count'] / total_samples) * 100
    res_df.to_csv(output_csv, index=False)
    print(f"结果已保存至: {output_csv}")

if __name__ == "__main__":
    main()