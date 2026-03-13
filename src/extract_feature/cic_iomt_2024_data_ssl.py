import pandas as pd
import os
from collections import Counter, defaultdict

# ================= 配置区域 =================
# 指向您的全量 csv 文件
TARGET_FILE = r"E:\QingjunXiao\code-multiview-network-traffic-classification-model\processed_data\CIC-IoMT-2024-session-srcIP_srcPort_dstIP_dstPort_proto\all_embedded_flow.csv" 

# 此数据集特有的列名
LABEL_COL = "label"             # 注意是小写
DNS_COL = "dns.query"           # DNS 列
SNI_COL = "ssl.server_name"     # SNI 列

# 标签标准化逻辑 (针对 IoMT 数据集)
def normalize_label(val):
    s = str(val).strip()
    if s.lower().startswith("benign"): return "Benign"
    if s.startswith("malicious_"): return s.replace("malicious_", "")
    return s
# ===========================================

def main():
    if not os.path.exists(TARGET_FILE):
        print(f"文件不存在: {TARGET_FILE}")
        return

    print("开始分析...")
    # 统计器
    label_dist = Counter()
    dns_stats = defaultdict(Counter)
    sni_stats = defaultdict(Counter)

    # 分块读取
    chunk_size = 100000
    try:
        reader = pd.read_csv(TARGET_FILE, chunksize=chunk_size, low_memory=False)
        
        for i, chunk in enumerate(reader):
            # 1. 标签处理
            if LABEL_COL not in chunk.columns: continue
            chunk['norm_label'] = chunk[LABEL_COL].apply(normalize_label)
            
            # 统计标签
            label_dist.update(chunk['norm_label'].value_counts().to_dict())

            # 2. 统计 DNS
            if DNS_COL in chunk.columns:
                valid_dns = chunk[chunk[DNS_COL].notna() & (chunk[DNS_COL] != '')]
                for label, grp in valid_dns.groupby('norm_label'):
                    dns_stats[label].update(grp[DNS_COL].value_counts().to_dict())

            # 3. 统计 SNI
            if SNI_COL in chunk.columns:
                valid_sni = chunk[chunk[SNI_COL].notna() & (chunk[SNI_COL] != '')]
                for label, grp in valid_sni.groupby('norm_label'):
                    sni_stats[label].update(grp[SNI_COL].value_counts().to_dict())

            if (i+1) % 5 == 0:
                print(f"已处理 { (i+1)*chunk_size } 行...")

    except Exception as e:
        print(f"出错: {e}")

    # === 打印结果 ===
    print("\n" + "="*50)
    print("Top DNS & SNI 统计结果")
    print("="*50)
    
    # 按 Label 打印
    for label in sorted(label_dist.keys()):
        print(f"\n>>> 类别: {label} (样本数: {label_dist[label]})")
        
        # DNS
        print("  [Top 5 DNS]")
        for d, c in dns_stats[label].most_common(5):
            print(f"    {c:<6} {d}")
            
        # SNI
        print("  [Top 5 SNI]")
        for d, c in sni_stats[label].most_common(5):
            print(f"    {c:<6} {d}")

if __name__ == "__main__":
    main()