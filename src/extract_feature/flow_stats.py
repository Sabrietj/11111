#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流量统计脚本 - 分析CSV格式的流量数据文件
支持label字段统计分析（class_family格式）、DNS查询统计和多核并行处理
性能优化版本 - 支持结果导出到CSV
"""

import argparse
import pandas as pd
import sys
import os
import multiprocessing as mp
from collections import Counter, defaultdict
import time
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
from config_manager import read_plot_data_path_config
import traceback

def analyze_dns_statistics_optimized(input_file, chunk_size=50000):
    """分析DNS查询统计数据 - 性能优化版本"""
    print(f"\n开始DNS查询统计分析...")
    start_time = time.time()
    
    # 初始化统计数据结构
    dns_query_stats = defaultdict(lambda: {'classes': set(), 'families': set(), 'count': 0})
    class_dns_stats = defaultdict(lambda: Counter())
    family_dns_stats = defaultdict(lambda: Counter())
    
    total_rows = 0
    non_empty_dns_count = 0
    
    # 单次文件读取，同时检查列名和处理数据
    for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)):
        total_rows += len(chunk)
        
        # 向量化筛选dns.query非空记录
        non_empty_mask = chunk['dns.query'].notna() & (chunk['dns.query'] != '')
        non_empty_dns = chunk[non_empty_mask]
        non_empty_dns_count += len(non_empty_dns)
        
        if len(non_empty_dns) > 0:
            # 批量处理：向量化操作
            dns_queries = non_empty_dns['dns.query'].astype(str).str.strip()
            labels = non_empty_dns['label'].astype(str)
            
            # 批量解析class和family - 使用pandas向量化操作
            class_family_parts = labels.str.split('_', n=2, expand=True)
            class_names = class_family_parts[0].fillna('').str.strip()
            family_names = class_family_parts[1].fillna('').str.strip()
            
            # 批量更新统计信息 - 避免逐行循环
            for i in range(len(non_empty_dns)):
                dns_query = dns_queries.iloc[i]
                class_name = class_names.iloc[i]
                family_name = family_names.iloc[i]
                
                # 更新DNS查询维度统计
                dns_query_stats[dns_query]['classes'].add(class_name)
                dns_query_stats[dns_query]['families'].add(family_name)
                dns_query_stats[dns_query]['count'] += 1
                
                # 更新class级别统计
                if class_name:
                    class_dns_stats[class_name][dns_query] += 1
                
                # 更新family级别统计
                if family_name:
                    family_dns_stats[family_name][dns_query] += 1
        
        # 优化进度显示（减少输出频率）
        if (chunk_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
            print(f"已处理 {total_rows:,} 行数据，非空DNS查询: {non_empty_dns_count:,} (速度: {rows_per_sec:,.0f} 行/秒)")
    
    total_time = time.time() - start_time
    print(f"DNS查询分析完成，耗时: {total_time:.2f} 秒")
    
    return {
        'total_rows': total_rows,
        'non_empty_dns_count': non_empty_dns_count,
        'dns_query_stats': dns_query_stats,
        'class_dns_stats': class_dns_stats,
        'family_dns_stats': family_dns_stats
    }


def analyze_label_statistics_optimized(input_file, chunk_size=50000):
    """分析label字段统计 - 性能优化版本"""
    print(f"\n开始Label字段统计分析...")
    start_time = time.time()
    
    class_counts = Counter()
    total_rows = 0
    
    # 分块读取数据
    for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)):
        total_rows += len(chunk)
        
        # 向量化处理label字段
        valid_labels = chunk['label'].dropna().astype(str)
        class_labels = valid_labels[valid_labels.str.contains('_')]
        
        # 批量提取class名称
        if len(class_labels) > 0:
            class_names = class_labels.str.split('_').str[0].str.strip()
            class_counts.update(class_names.value_counts().to_dict())
        
        # 优化进度显示
        if (chunk_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
            print(f"已处理 {total_rows:,} 行数据 (速度: {rows_per_sec:,.0f} 行/秒)")
    
    total_time = time.time() - start_time
    print(f"Label字段分析完成，耗时: {total_time:.2f} 秒")
    
    return {
        'total_rows': total_rows,
        'class_counts': class_counts
    }


def save_dns_analysis_to_csv(dns_stats, input_filename, plot_data_path):
    """将DNS分析结果保存到CSV文件"""
    base_name = input_filename.replace('.csv', '')
    output_filename = f"{base_name}_label_anal.csv"
    output_filepath = os.path.join(plot_data_path, output_filename)
    
    # 创建DataFrame来存储所有分析结果
    analysis_data = []
    
    # 1. DNS查询维度Top 20统计
    dns_query_stats = dns_stats['dns_query_stats']
    sorted_queries = sorted(dns_query_stats.items(), 
                          key=lambda x: len(x[1]['classes']), 
                          reverse=True)[:20]
    
    for rank, (dns_query, stats) in enumerate(sorted_queries, 1):
        analysis_data.append({
            '分析类型': 'DNS查询维度Top20',
            '排名': rank,
            'DNS查询': dns_query,
            'Class数量': len(stats['classes']),
            'Family数量': len(stats['families']),
            '记录次数': stats['count']
        })
    
    # 2. Class级别DNS查询Top 20统计
    class_dns_stats = dns_stats['class_dns_stats']
    for class_name, dns_counter in class_dns_stats.items():
        top_20_dns = dns_counter.most_common(20)
        for rank, (dns_query, count) in enumerate(top_20_dns, 1):
            analysis_data.append({
                '分析类型': f'Class级别Top20_{class_name}',
                '排名': rank,
                'Class名称': class_name,
                'DNS查询': dns_query,
                '记录次数': count
            })
    
    # 3. Family级别DNS查询Top 20统计
    family_dns_stats = dns_stats['family_dns_stats']
    for family_name, dns_counter in family_dns_stats.items():
        top_20_dns = dns_counter.most_common(20)
        for rank, (dns_query, count) in enumerate(top_20_dns, 1):
            analysis_data.append({
                '分析类型': f'Family级别Top20_{family_name}',
                '排名': rank,
                'Family名称': family_name,
                'DNS查询': dns_query,
                '记录次数': count
            })
    
    # 4. 统计摘要
    analysis_data.append({
        '分析类型': '统计摘要',
        '总记录数': dns_stats['total_rows'],
        '非空DNS查询记录数': dns_stats['non_empty_dns_count'],
        '涉及的Class数量': len(dns_stats['class_dns_stats']),
        '涉及的Family数量': len(dns_stats['family_dns_stats']),
        '唯一DNS查询数量': len(dns_stats['dns_query_stats'])
    })
    
    # 保存到CSV文件
    df_analysis = pd.DataFrame(analysis_data)
    df_analysis.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    
    return output_filepath


def print_dns_query_dimension_top20(dns_query_stats):
    """输出DNS查询维度Top 20统计"""
    print(f"\nDNS查询维度Top 20统计:")
    print("-" * 60)
    print(f"{'排名':<4} {'DNS查询':<30} {'Class数量':<12} {'Family数量':<12}")
    print("-" * 60)
    
    # 按Class数量降序排序，取Top 20
    sorted_queries = sorted(dns_query_stats.items(), 
                          key=lambda x: len(x[1]['classes']), 
                          reverse=True)[:20]
    
    for rank, (dns_query, stats) in enumerate(sorted_queries, 1):
        class_count = len(stats['classes'])
        family_count = len(stats['families'])
        print(f"{rank:<4} {dns_query:<30} {class_count:<12} {family_count:<12}")


def print_class_dns_top20(class_dns_stats):
    """输出Class级别DNS查询Top 20统计"""
    print(f"\nClass级别DNS查询Top 20统计:")
    print("-" * 70)
    print(f"{'#Class':<15} {'#排名':<6} {'#域名':<30} {'#记录次数':<12}")
    print("-" * 70)
    
    for class_name, dns_counter in class_dns_stats.items():
        top_20_dns = dns_counter.most_common(20)
        for rank, (dns_query, count) in enumerate(top_20_dns, 1):
            print(f"{class_name:<15} {rank:<6} {dns_query:<30} {count:<12}")


def print_family_dns_top20(family_dns_stats):
    """输出Family级别DNS查询Top 20统计"""
    print(f"\nFamily级别DNS查询Top 20统计:")
    print("-" * 70)
    print(f"{'#Family':<15} {'#排名':<6} {'#域名':<30} {'#记录次数':<12}")
    print("-" * 70)
    
    for family_name, dns_counter in family_dns_stats.items():
        top_20_dns = dns_counter.most_common(20)
        for rank, (dns_query, count) in enumerate(top_20_dns, 1):
            print(f"{family_name:<15} {rank:<6} {dns_query:<30} {count:<12}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='流量数据统计脚本 - 性能优化版本')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行工作进程数，默认4个线程')
    parser.add_argument('--chunk_size', type=int, default=50000,
                       help='分块大小，默认50000行')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    plot_data_path = read_plot_data_path_config()
    input_file = os.path.join(plot_data_path, "all_flow.csv")
    if not os.path.exists(input_file):
        print(f"错误: 文件 '{input_file}' 不存在")
        sys.exit(1)
    
    try:
        start_time = time.time()
        print(f"正在分析文件: {input_file}")
        
        # 获取文件大小信息
        file_size = os.path.getsize(input_file)
        print(f"文件大小: {file_size / (1024*1024*1024):.2f} GB")
        
        # 只读取文件头部来获取列名
        df_columns = pd.read_csv(input_file, nrows=0)
        
        # 输出列名信息
        print(f"\n列名列表 ({len(df_columns.columns)} 列):")
        print("-" * 50)

        # 打印所有列名，格式为："列名1","列名2","列名3"
        print(",".join([f'"{col}"' for col in df_columns.columns]))
        print("-" * 50)

        print("=====OK======")
        for i, col_name in enumerate(df_columns.columns, 1):
            print(f"{i:2d}. {col_name}")
        print("-" * 50)
        
        # 检查是否包含必要的列
        required_columns = ['dns.query', 'label']
        missing_columns = [col for col in required_columns if col not in df_columns.columns]
        if missing_columns:
            print(f"警告: 缺少必要列: {missing_columns}")
            print("将跳过DNS查询统计分析")
            dns_stats = None
        else:
            # 进行DNS查询统计分析
            dns_stats = analyze_dns_statistics_optimized(input_file, args.chunk_size)
            
            # 输出统计摘要
            print(f"\nDNS查询统计分析摘要:")
            print(f"- 总记录数: {dns_stats['total_rows']:,}")
            print(f"- 非空DNS查询记录数: {dns_stats['non_empty_dns_count']:,}")
            print(f"- 涉及的Class数量: {len(dns_stats['class_dns_stats'])}")
            print(f"- 涉及的Family数量: {len(dns_stats['family_dns_stats'])}")
            print(f"- 唯一DNS查询数量: {len(dns_stats['dns_query_stats'])}")
            
            # 输出各维度统计结果
            print_dns_query_dimension_top20(dns_stats['dns_query_stats'])
            print_class_dns_top20(dns_stats['class_dns_stats'])
            print_family_dns_top20(dns_stats['family_dns_stats'])
            
            # 保存分析结果到CSV文件
            input_filename = os.path.basename(input_file)
            analysis_file = save_dns_analysis_to_csv(dns_stats, input_filename, plot_data_path)
            print(f"\nDNS分析结果已保存到: {analysis_file}")
        
        # 分析label字段的统计信息
        label_stats = analyze_label_statistics_optimized(input_file, args.chunk_size)
        
        print(f"\nLabel字段统计分析:")
        print("-" * 50)
        print(f"总记录数: {label_stats['total_rows']:,}")
        print("\nClass分类统计:")
        print("-" * 30)
        for class_name, count in sorted(label_stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / label_stats['total_rows']) * 100
            print(f"{class_name:<20} {count:>10,} ({percentage:>6.2f}%)")
        
        # 读取前50行数据并保存到文件（可选功能）
        try:
            df_sample = pd.read_csv(input_file, nrows=50)
            input_filename = os.path.basename(input_file)
            output_filename = input_filename.replace('.csv', '_head50.csv')
            output_filepath = os.path.join(plot_data_path, output_filename)
            df_sample.to_csv(output_filepath, index=False)
            print(f"\n前50行数据已保存到: {output_filepath}")
        except Exception as sample_error:
            print(f"\n无法保存前50行数据: {sample_error}")
            traceback.print_exc()

        total_time = time.time() - start_time
        print(f"\n总处理时间: {total_time:.2f} 秒")
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()