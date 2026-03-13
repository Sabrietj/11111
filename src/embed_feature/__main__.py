import os, sys
import argparse
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
#from domain_embedding_serial import DomainEmbeddingProcessor
from domain_embedding import DomainEmbeddingProcessor
from tqdm import tqdm  # 添加tqdm导入
from sklearn.model_selection import train_test_split

# 导入配置管理器和相关模块
try:
    # 添加../utils目录到Python搜索路径
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
    sys.path.insert(0, utils_path)    
    import config_manager as ConfigManager

    from logging_config import setup_preset_logging
    import logging
    # 使用统一的日志配置
    logger = setup_preset_logging(log_level=logging.INFO)
    
except ImportError as e:
    logger.error(f"导入错误: {e}")
    logger.error("请确保项目结构正确，所有依赖模块可用")
    sys.exit(1)

def find_flow_session_csv_pairs(root_dir, verbose=True):
    """
    遍历 root_dir 和一级子目录，寻找 xxxx-flow.csv 与 xxxx-session.csv 配对
    """
    pairs = []
    ignore_files = {'all_flow.csv', 'all_session.csv', 'all_embedded_flow.csv', 'all_split_session.csv'}
    
    for dirpath, _, filenames in os.walk(root_dir):
        # 限制只搜索 root_dir 和一级子目录
        if os.path.relpath(dirpath, root_dir).count(os.sep) > 1:
            continue

        # 过滤掉要忽略的文件
        flow_files = [f for f in filenames 
                     if f.endswith("-flow.csv") and f not in ignore_files]
        session_files = [f for f in filenames 
                        if f.endswith("-session.csv") and f not in ignore_files]

        # 构建文件名前缀到路径的映射
        flow_path_dict = {f.replace("-flow.csv", ""): os.path.join(dirpath, f) 
                    for f in flow_files}
        session_path_dict = {f.replace("-session.csv", ""): os.path.join(dirpath, f) 
                       for f in session_files}

        # 收集匹配的配对
        for key in flow_path_dict:
            if key in session_path_dict:
                pairs.append((flow_path_dict[key], session_path_dict[key]))
    
    if verbose:
        logger.info(f"共找到 {len(pairs)} 对 flow/session CSV 文件")
    
    return pairs


def save_dataframe_with_progress(df, filepath, description="保存数据", verbose=True):
    """带进度提示的数据保存函数"""
    if verbose:
        logger.info(f"{description}到 {filepath}...")
        logger.info(f"数据形状: {df.shape}, 预计需要一些时间...")
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 使用更高效的保存方式
    try:
        # 尝试使用更快的保存方法
        if verbose:
            start_time = time.time()
        
        # 分块保存大文件（如果数据量很大）
        if len(df) > 1000000:  # 如果超过100万行，分块保存
            if verbose:
                logger.info(f"数据量较大 ({len(df)} 行)，使用分块保存...")
            
            # 先保存小样本测试
            test_sample = df.iloc[:1000]
            test_sample.to_csv(filepath, index=False)
            
            # 然后追加剩余数据
            chunks = np.array_split(df.index, 10)  # 分成10块
            for i, chunk_indices in enumerate(chunks):
                if i == 0:
                    # 第一块：跳过前1000行（已经保存过了），保存剩余部分
                    if len(chunk_indices) > 1000:
                        chunk_df = df.loc[chunk_indices[1000:]]
                        chunk_df.to_csv(filepath, mode='a', header=False, index=False)
                else:
                    # 其他块：正常保存
                    chunk_df = df.loc[chunk_indices]
                    chunk_df.to_csv(filepath, mode='a', header=False, index=False)
                
                if verbose:
                    progress = (i + 1) / len(chunks) * 100
                    logger.info(f"{description}: {progress:.1f}%")
        else:
            # 直接保存
            df.to_csv(filepath, index=False)
        
        if verbose:
            end_time = time.time()
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            logger.info(f"{description}完成! 耗时: {end_time - start_time:.2f}秒, 文件大小: {file_size:.2f}MB")
            
    except Exception as e:
        if verbose:
            logger.warning(f"快速保存失败: {e}, 尝试标准保存方式...")
        # 回退到标准保存方式
        df.to_csv(filepath, index=False)
    
    return filepath


def merge_csv_files(flow_session_csv_pairs, dataset_dir, verbose=True):
    """
    合并所有 flow_csv 和 session_csv，添加进度条
    """
    if verbose:
        logger.info(f"开始合并 {len(flow_session_csv_pairs)} 对CSV文件...")
    
    merged_flow_path = os.path.join(dataset_dir, "all_flow.csv")
    # merge_flow_csv_files_in_memory(
    #     flow_session_csv_pairs=flow_session_csv_pairs, 
    #     merged_flow_path=merged_flow_path, 
    #     verbose=verbose)    
    merge_flow_csv_files_in_stream(
        flow_session_csv_pairs=flow_session_csv_pairs, 
        merged_flow_path=merged_flow_path, 
        verbose=verbose)

    merged_session_path = os.path.join(dataset_dir, "all_session.csv")
    # merge_session_csv_files_in_memory(
    #     flow_session_csv_pairs=flow_session_csv_pairs, 
    #     merged_session_path=merged_session_path, 
    #     verbose=verbose)
    merge_session_csv_files_in_stream(
        flow_session_csv_pairs=flow_session_csv_pairs, 
        merged_session_path=merged_session_path, 
        verbose=verbose)

    return merged_flow_path, merged_session_path

def merge_flow_csv_files_in_memory(flow_session_csv_pairs, merged_flow_path, verbose):
    raise RuntimeError(
        "Do NOT use in-memory merge for large datasets. "
        "Use merge_flow_csv_files_in_stream instead."
    )
    
    if not os.path.exists(merged_flow_path):
        # 合并 flow_csv，添加进度条
        all_flow_dfs = []
        if verbose:
            logger.info(f"合并flow数据...")
            pbar = tqdm(flow_session_csv_pairs, desc="合并flow文件", disable=not verbose)
        else:
            pbar = flow_session_csv_pairs
            
        for flow_csv, _ in pbar:
            try:
                df = pd.read_csv(flow_csv, low_memory=False)

                # 略过空的文件
                if df.empty:
                    continue

                all_flow_dfs.append(df)

                if verbose:
                    pbar.set_postfix(file=os.path.basename(flow_csv), rows=len(df))
            except Exception as e:
                logger.warning(f"读取文件失败 {flow_csv}: {e}")
                continue
        
        if all_flow_dfs:
            if verbose:
                logger.info(f"合并flow数据帧... (共 {len(all_flow_dfs)} 个文件)")
            merged_flow = pd.concat(all_flow_dfs, ignore_index=True)
            
            # 使用带进度提示的保存函数
            save_dataframe_with_progress(merged_flow, merged_flow_path, "保存flow数据", verbose)
            
            if verbose:
                logger.info(f"Flow数据合并完成: {merged_flow_path} （{len(merged_flow)} 行）")
        else:
            raise ValueError("没有成功读取任何flow文件")
    
    else:
        if verbose:
            logger.info(f"已存在，跳过合并: {merged_flow_path}")

def merge_flow_csv_files_in_stream(flow_session_csv_pairs, merged_flow_path, verbose=True):
    if os.path.exists(merged_flow_path):
        if verbose:
            logger.info(f"已存在，跳过合并: {merged_flow_path}")
        return

    if verbose:
        logger.info("合并 flow 数据（流式写入）...")
        pbar = tqdm(flow_session_csv_pairs, desc="合并 flow 文件", disable=not verbose)
    else:
        pbar = flow_session_csv_pairs

    first = True
    written_rows = 0
    expected_columns = None

    for flow_csv, _ in pbar:
        try:
            df = pd.read_csv(flow_csv, low_memory=False)

            if df.empty:
                continue

        except Exception as e:
            logger.warning(f"读取 flow 文件失败 {flow_csv}: {e}")
            continue

        if first:
            expected_columns = tuple(df.columns)
        else:
            if tuple(df.columns) != expected_columns:
                raise ValueError(f"列不一致: {flow_csv}")
        
        try:                
            rows = len(df)

            df.to_csv(
                merged_flow_path,
                mode="w" if first else "a",
                header=first,
                index=False
            )

            first = False
            written_rows += rows
            del df

            if verbose:
                pbar.set_postfix(
                    file=os.path.basename(flow_csv),
                    rows=rows,
                    total=written_rows
                )

        except Exception as e:
            logger.warning(f"写入 flow 文件失败 {flow_csv}: {e}")
            continue

    if first:
        raise ValueError("没有成功写入任何 flow 文件")

    if verbose:
        logger.info(f"Flow 数据合并完成: {merged_flow_path} （{written_rows} 行）")
        size_mb = os.path.getsize(merged_flow_path) / 1024 / 1024
        logger.info(f"Flow 文件大小: {size_mb:.2f} MB")

def merge_session_csv_files_in_memory(flow_session_csv_pairs, merged_session_path, verbose):
    raise RuntimeError(
        "Do NOT use in-memory merge for large datasets. "
        "Use merge_session_csv_files_in_stream instead."
    )
    
    if not os.path.exists(merged_session_path):
        # 合并 session_csv，添加进度条
        all_session_dfs = []
        if verbose:
            logger.info(f"合并session数据...")
            pbar = tqdm(flow_session_csv_pairs, desc="合并session文件", disable=not verbose)
        else:
            pbar = flow_session_csv_pairs
            
        for _, session_csv in pbar:
            try:
                df = pd.read_csv(session_csv, low_memory=False)
                all_session_dfs.append(df)
                if verbose:
                    pbar.set_postfix(file=os.path.basename(session_csv), rows=len(df))
            except Exception as e:
                logger.warning(f"读取文件失败 {session_csv}: {e}")
                continue
        
        if all_session_dfs:
            if verbose:
                logger.info(f"合并session数据帧... (共 {len(all_session_dfs)} 个文件)")
            merged_session = pd.concat(all_session_dfs, ignore_index=True)
            
            # 使用带进度提示的保存函数
            save_dataframe_with_progress(merged_session, merged_session_path, "保存session数据", verbose)
            
            if verbose:
                logger.info(f"Session数据合并完成: {merged_session_path} （{len(merged_session)} 行）")
        else:
            raise ValueError("没有成功读取任何session文件")

    else:
        if verbose:
            logger.info(f"已存在，跳过合并: {merged_session_path}")

def merge_session_csv_files_in_stream(flow_session_csv_pairs, merged_session_path, verbose=True):
    if os.path.exists(merged_session_path):
        if verbose:
            logger.info(f"已存在，跳过合并: {merged_session_path}")
        return

    if verbose:
        logger.info("合并 session 数据（流式写入）...")
        pbar = tqdm(flow_session_csv_pairs, desc="合并 session 文件", disable=not verbose)
    else:
        pbar = flow_session_csv_pairs

    first = True
    written_rows = 0
    expected_columns = None

    for _, session_csv in pbar:
        try:
            df = pd.read_csv(session_csv, low_memory=False)

            if df.empty:
                continue
        except Exception as e:
            logger.warning(f"读取 session 文件失败 {session_csv}: {e}")
            continue

        if first:
            expected_columns = tuple(df.columns)
        else:
            if tuple(df.columns) != expected_columns:
                raise ValueError(f"列不一致: {session_csv}")

        try:
            rows = len(df)

            df.to_csv(
                merged_session_path,
                mode="w" if first else "a",
                header=first,
                index=False
            )

            first = False
            written_rows += rows
            del df

            if verbose:
                pbar.set_postfix(
                    file=os.path.basename(session_csv),
                    rows=rows,
                    total=written_rows
                )

        except Exception as e:
            logger.warning(f"写入 session 文件失败 {session_csv}: {e}")
            continue

    if first:
        raise ValueError("没有成功写入任何 session 文件")

    if verbose:
        logger.info(f"Session 数据合并完成: {merged_session_path} （{written_rows} 行）")
        size_mb = os.path.getsize(merged_session_path) / 1024 / 1024
        logger.info(f"Session 文件大小: {size_mb:.2f} MB")


def prepare_dataset_files(dataset_dir, reuse_prev_merged_data=True, verbose=True):
    """
    准备数据集文件：查找配对文件并合并
    """
    merged_flow_path = os.path.join(dataset_dir, "all_flow.csv")
    merged_session_path = os.path.join(dataset_dir, "all_session.csv")
    
    # 检查是否重用已有数据
    if os.path.exists(merged_flow_path) and os.path.exists(merged_session_path):
        if verbose:
            logger.info(f"合并后的数据已存在")
        
        if not reuse_prev_merged_data:
            if verbose:
                logger.info(f"重新执行合并流程")
            os.remove(merged_flow_path)
            os.remove(merged_session_path)
        else:
            if verbose:
                logger.info(f"重用之前合并的数据")
            return merged_flow_path, merged_session_path, True
    
    # 合并CSV文件（如果需要）
    if not os.path.exists(merged_flow_path) or not os.path.exists(merged_session_path):
        pairs = find_flow_session_csv_pairs(dataset_dir, verbose=verbose)
        if not pairs:
            raise ValueError("未找到匹配的 flow/session CSV 文件对！")
        
        merged_flow_path, merged_session_path = merge_csv_files(pairs, dataset_dir, verbose=verbose)
        return merged_flow_path, merged_session_path, False
    
    return merged_flow_path, merged_session_path, True

def random_split_sessions(session_df, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1, random_state=42, verbose=True):
    """
    确保划分比例正确应用，并随机分布split标签
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 打乱数据
    shuffled_df = session_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n_total = len(shuffled_df)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    
    # 确保至少有一个验证样本
    n_val = n_total - n_train - n_test
    if n_val == 0 and n_total > 2:  # 如果验证集为0但数据量足够
        n_train = n_train - 1
        n_val = 1
    
    # 划分数据集
    train_df = shuffled_df.iloc[:n_train].copy()
    test_df = shuffled_df.iloc[n_train:n_train+n_test].copy()
    val_df = shuffled_df.iloc[n_train+n_test:].copy()
    
    # 标准化split标签
    train_df['split'] = 'train'
    test_df['split'] = 'test' 
    val_df['split'] = 'validate'
    
    # 合并数据
    split_session_df = pd.concat([train_df, test_df, val_df], ignore_index=True)
    
    # 关键修改：再次打乱数据，确保split标签随机分布
    split_session_df = split_session_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    if verbose:
        logger.info(f"数据集划分完成:")
        logger.info(f"  - 训练集: {len(train_df)} 个会话 ({len(train_df)/n_total*100:.1f}%)")
        logger.info(f"  - 测试集: {len(test_df)} 个会话 ({len(test_df)/n_total*100:.1f}%)")
        logger.info(f"  - 验证集: {len(val_df)} 个会话 ({len(val_df)/n_total*100:.1f}%)")
        
        # 验证split标签的分布
        split_distribution = split_session_df['split'].value_counts().sort_index()
        logger.info(f"最终split标签分布:")
        for split_type in ['train', 'test', 'validate']:
            count = split_distribution.get(split_type, 0)
            percentage = count / len(split_session_df) * 100
            logger.info(f"  - {split_type}: {count} 个会话 ({percentage:.1f}%)")
        
        # 检查前几行的split标签分布作为示例
        logger.info(f"前10行split标签示例:")
        for i in range(min(10, len(split_session_df))):
            logger.info(f"  行 {i}: {split_session_df.iloc[i]['split']}")
    
    return split_session_df


def stratified_split_sessions(
    session_df,
    label_col="is_malicious",
    train_ratio=0.7,
    test_ratio=0.2,
    val_ratio=0.1,
    random_state=42,
    verbose=True
):
    """
    基于 is_malicious 的分层抽样（Stratified Split）
    切分 train / test / validate，并保持类别比例一致。

    分层采样是一种按类别比例进行数据集划分的方法。其核心思想：
    确保训练集、验证集、测试集中的各类别比例，与原始数据集保持一致。
    尤其适用于 类别极度不平衡 的任务，例如恶意流量检测、欺诈检测、病毒识别等。

    分层采样通过保持训练集、验证集、测试集的类别比例一致，避免随机采样造成的类别缺失或严重偏差。
    对于不平衡数据集，分层采样能显著提升模型的训练稳定性、验证准确性和实验可重复性，因此通常优于随机采样。
    """

    if verbose:
        logger.info("开始基于分层抽样的会话划分 stratified_split_sessions() ...")

    # 1) 检查比例是否正确
    if not np.isclose(train_ratio + test_ratio + val_ratio, 1.0):
        raise ValueError("train_ratio + test_ratio + val_ratio 必须等于 1.0")

    # 2) 获取标签
    labels = session_df[label_col]

    logger.info("==== 原始数据标签分布 ====")
    vc = session_df["is_malicious"].value_counts()
    logger.info(f"   正样本: {vc.get(1,0)}")
    logger.info(f"   负样本: {vc.get(0,0)}")
    logger.info(f"   总数: {len(session_df)}")
    logger.info(f"   恶意比例: {vc.get(1,0) / len(session_df):.6f}")
    
    # 3) 先划分 train 与 temp（val + test）
    temp_ratio = 1.0 - train_ratio

    train_df, temp_df = train_test_split(
        session_df,
        test_size=temp_ratio,
        stratify=labels,
        random_state=random_state
    )

    # 4) 再按照比例切 temp → val + test
    val_ratio_adj = val_ratio / (val_ratio + test_ratio)

    temp_labels = temp_df[label_col]
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adj),
        stratify=temp_labels,
        random_state=random_state
    )

    # 5) 添加 split 字段
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "validate"
    test_df["split"] = "test"

    # 6) 合并
    split_session_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 再次随机打乱，让 split 标签分布更随机
    split_session_df = split_session_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # -----------------------
    # verbose 输出信息
    # -----------------------
    if verbose:
        logger.info("分层抽样完成，数据集划分如下：")
        n_total = len(session_df)

        logger.info(f"  - 训练集: {len(train_df)} 条 ({len(train_df)/n_total*100:.1f}%)")
        logger.info(f"  - 验证集: {len(val_df)} 条 ({len(val_df)/n_total*100:.1f}%)")
        logger.info(f"  - 测试集:  {len(test_df)} 条 ({len(test_df)/n_total*100:.1f}%)")

        # 类别分布
        def log_label_dist(name, df):
            vc = df[label_col].value_counts()
            total = len(df)
            malicious = vc.get(1, 0)
            benign = vc.get(0, 0)
            logger.info(f"    [{name}] 正样本={malicious}, 负样本={benign}, 比例={malicious/total:.4f}")

        logger.info("各数据集的标签分布：")
        log_label_dist("Train", train_df)
        log_label_dist("Validate", val_df)
        log_label_dist("Test", test_df)

        # split 标签分布
        split_distribution = split_session_df["split"].value_counts().sort_index()
        logger.info("最终 split 标签分布：")
        for split_type in ['train', 'validate', 'test']:
            count = split_distribution.get(split_type, 0)
            logger.info(f"  - {split_type}: {count}")

        # 前若干行检查
        logger.info("前 10 行 split 标签示例：")
        for i in range(min(10, len(split_session_df))):
            logger.info(f"  行 {i}: {split_session_df.iloc[i]['split']}")

    return split_session_df

def validate_uid_batch(args):
    """验证一批UID（多进程工作函数）"""
    uid_batch, flow_uid_set = args
    return [uid for uid in uid_batch if uid in flow_uid_set]

def parallel_validate_uids(flow_uids, flow_uid_set, batch_size=5000, num_workers=None):
    """并行验证大量UID"""
    if num_workers is None:
        num_workers = mp.cpu_count()  # 使用所有CPU核心
    
    # 分批处理
    batches = []
    for i in range(0, len(flow_uids), batch_size):
        batch = flow_uids[i:i + batch_size]
        batches.append((batch, flow_uid_set))
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(validate_uid_batch, batches))
    
    # 合并结果
    valid_uids = []
    for result in results:
        valid_uids.extend(result)
    
    return valid_uids

def smart_validate_uids(flow_uids, flow_uid_set, parallel_threshold=1000):
    """
    智能选择验证策略：
    - 小批量：直接串行处理（避免进程创建开销）
    - 大批量：并行处理（充分利用多核）
    """
    if len(flow_uids) <= parallel_threshold:
        # 小批量：串行处理
        return [uid for uid in flow_uids if uid in flow_uid_set]
    else:
        # 大批量：并行处理
        return parallel_validate_uids(flow_uids, flow_uid_set)

def get_train_flow_uids(session_df, flow_df, verbose=True, show_progress=True):
    """
    从session_df的训练集中获取对应的flow uid列表
    
    Args:
        session_df: 会话数据DataFrame
        flow_df: 流量数据DataFrame
        verbose: 是否输出详细信息
        show_progress: 是否显示进度条
    
    Returns:
        set: 训练集对应的flow uid集合
    """
    train_flow_uids = set()
    
    # 筛选训练集的session
    train_sessions = session_df[session_df['split'] == 'train']
    total_sessions = len(train_sessions)
    
    # 预处理：创建UID集合（关键优化）
    if verbose:
        logger.info(f"预处理：创建UID集合...")
    flow_uid_set_start = time.time()
    flow_uid_set = set(flow_df['uid'].values)
    flow_uid_set_time = time.time() - flow_uid_set_start
    if verbose:
        logger.info(f"UID集合创建完成，耗时 {flow_uid_set_time:.2f}秒")
        logger.info(f"UID集合大小: {len(flow_uid_set)}")
        logger.info(f"可用CPU核心数: {mp.cpu_count()}")
    
    # 统计计数器
    stats = {
        'processed': 0,
        'valid_uids': 0,
        'invalid_uids': 0,
        'empty_lists': 0,
        'parse_errors': 0,
        'total_uids_found': 0,
        'parallel_sessions': 0,  # 使用并行处理的会话数
        'serial_sessions': 0,   # 使用串行处理的会话数
        'max_uid_count': 0      # 最大UID数量
    }
    
    # 进度显示逻辑
    start_time = time.time()
    if verbose and show_progress:
        logger.info(f"开始处理 {total_sessions} 个训练会话...")
    
    for idx, (_, session) in enumerate(train_sessions.iterrows()):
        stats['processed'] = idx + 1
        
        # 添加会话级别的调试信息
        if verbose and idx < 30:  # 只显示前30个会话的详细信息
            logger.info(f"\n处理第 {idx+1} 个会话...")
        
        flow_uid_list_str = session.get('flow_uid_list', '')
        
        # 检查是否为空或NaN
        if pd.isna(flow_uid_list_str) or not flow_uid_list_str:
            stats['empty_lists'] += 1
            if verbose and idx < 30:
                logger.info(f"  会话 {idx+1}: flow_uid_list 为空")
            continue
        
        # 显示flow_uid_list的内容（截断显示）
        if verbose and idx < 30:
            preview = str(flow_uid_list_str)[:100] + "..." if len(str(flow_uid_list_str)) > 100 else str(flow_uid_list_str)
            logger.info(f"  会话 {idx+1}: flow_uid_list = '{preview}'")
        
        try:
            # 解析flow_uid_list字符串为列表
            import ast
            flow_uids = ast.literal_eval(flow_uid_list_str)
            
            if verbose and idx < 30:
                logger.info(f"  会话 {idx+1}: 解析成功，得到 {len(flow_uids)} 个UID")
            
            # 更新最大UID数量统计
            stats['max_uid_count'] = max(stats['max_uid_count'], len(flow_uids))
            
            if isinstance(flow_uids, list):
                # 使用智能验证策略（关键优化）
                validation_start = time.time()
                
                if len(flow_uids) > 1000:
                    stats['parallel_sessions'] += 1
                    if verbose and idx < 30:
                        logger.info(f"  会话 {idx+1}: 使用并行验证 ({len(flow_uids)} 个UID)")
                    valid_uids = smart_validate_uids(flow_uids, flow_uid_set)
                else:
                    stats['serial_sessions'] += 1
                    if verbose and idx < 30:
                        logger.info(f"  会话 {idx+1}: 使用串行验证 ({len(flow_uids)} 个UID)")
                    valid_uids = [uid for uid in flow_uids if uid in flow_uid_set]
                
                validation_time = time.time() - validation_start
                invalid_uids_count = len(flow_uids) - len(valid_uids)
                
                if verbose and idx < 30:
                    logger.info(f"  会话 {idx+1}: UID验证完成，耗时 {validation_time:.2f}秒")
                    logger.info(f"  会话 {idx+1}: 有效UID: {len(valid_uids)}, 无效UID: {invalid_uids_count}")
                
                stats['total_uids_found'] += len(flow_uids)
                stats['valid_uids'] += len(valid_uids)
                stats['invalid_uids'] += invalid_uids_count
                
                train_flow_uids.update(valid_uids)
            else:
                stats['parse_errors'] += 1
                if verbose and idx < 30:
                    logger.info(f"  会话 {idx+1}: 解析结果不是列表类型")
                
        except Exception as e:
            # 如果解析失败，尝试其他格式处理
            if verbose and idx < 30:
                logger.info(f"  会话 {idx+1}: 解析失败: {e}")
            
            if isinstance(flow_uid_list_str, str):
                # 尝试按逗号分割
                uids = [uid.strip().strip("'\"") for uid in flow_uid_list_str.split(',') if uid.strip()]
                
                if verbose and idx < 30:
                    logger.info(f"  会话 {idx+1}: 尝试逗号分割，得到 {len(uids)} 个UID")
                
                # 使用智能验证策略
                validation_start = time.time()
                
                if len(uids) > 1000:
                    stats['parallel_sessions'] += 1
                    if verbose and idx < 30:
                        logger.info(f"  [OPTIMIZE] 会话 {idx+1}: 使用并行验证 ({len(uids)} 个UID)")
                    valid_uids = smart_validate_uids(uids, flow_uid_set)
                else:
                    stats['serial_sessions'] += 1
                    if verbose and idx < 30:
                        logger.info(f"  [OPTIMIZE] 会话 {idx+1}: 使用串行验证 ({len(uids)} 个UID)")
                    valid_uids = [uid for uid in uids if uid in flow_uid_set]
                
                validation_time = time.time() - validation_start
                invalid_uids_count = len(uids) - len(valid_uids)
                
                if verbose and idx < 30:
                    logger.info(f"  会话 {idx+1}: UID验证完成，耗时 {validation_time:.2f}秒")
                    logger.info(f"  会话 {idx+1}: 有效UID: {len(valid_uids)}, 无效UID: {invalid_uids_count}")
                
                stats['total_uids_found'] += len(uids)
                stats['valid_uids'] += len(valid_uids)
                stats['invalid_uids'] += invalid_uids_count
                
                train_flow_uids.update(valid_uids)
            else:
                stats['parse_errors'] += 1
                if verbose and idx < 30:
                    logger.info(f"  会话 {idx+1}: flow_uid_list不是字符串类型")
        
        # 更新进度显示（每10个会话或最后一个会话时更新）
        if show_progress and (idx % 10 == 0 or idx == total_sessions - 1):
            elapsed = time.time() - start_time
            progress_percent = (idx + 1) / total_sessions * 100
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total_sessions - idx - 1) / speed if speed > 0 else 0
            
            # 显示进度信息
            progress_bar = f"进度: {progress_percent:.1f}% | " \
                         f"已处理: {idx+1}/{total_sessions} | " \
                         f"速度: {speed:.1f} 会话/秒 | " \
                         f"预计剩余: {eta:.1f}秒"
            
            if verbose:
                print(f"\r{progress_bar}", end='', flush=True)
    
    # 处理完成后的统计信息
    if verbose:
        total_time = time.time() - start_time
        
        if show_progress:
            logger.info(f"") # 换行
        
        logger.info(f" 训练集UID提取完成:")
        logger.info(f"  - 处理会话数: {stats['processed']}")
        logger.info(f"  - 并行处理会话: {stats['parallel_sessions']}")
        logger.info(f"  - 串行处理会话: {stats['serial_sessions']}")
        logger.info(f"  - 最大UID数量: {stats['max_uid_count']}")
        logger.info(f"  - 有效UID数量: {stats['valid_uids']}")
        logger.info(f"  - 无效UID数量: {stats['invalid_uids']}")
        logger.info(f"  - 空列表会话: {stats['empty_lists']}")
        logger.info(f"  - 解析错误: {stats['parse_errors']}")
        logger.info(f"  - 总UID数量: {stats['total_uids_found']}")
        logger.info(f"  - 最终UID集合大小: {len(train_flow_uids)}")
        logger.info(f"  - 处理耗时: {total_time:.2f}秒")
        logger.info(f"  - 预处理耗时: {flow_uid_set_time:.2f}秒")
        
        # 显示数据质量指标
        if stats['total_uids_found'] > 0:
            valid_ratio = stats['valid_uids'] / stats['total_uids_found'] * 100
            logger.info(f"  - UID有效性: {valid_ratio:.1f}%")
    
    return train_flow_uids

def read_large_csv_with_progress(filepath, description="读取数据", verbose=True):
    """带进度条的大型CSV文件读取函数"""
    if verbose:
        logger.info(f"{description}从 {filepath}...")
        file_size = os.path.getsize(filepath) / (1024 * 1024 * 1024)  # GB
        logger.info(f"文件大小: {file_size:.2f}GB")
    
    # 先读取前几行获取列信息
    sample_df = pd.read_csv(filepath, nrows=5)
    columns = sample_df.columns.tolist()
    
    # 分块读取
    chunks = []
    chunk_size = 100000  # 每次读取10万行

    if verbose:
        logger.info(f"检测到 {len(columns)} 列，开始每{chunk_size}行分块读取...")
    
    # 获取总行数（不读取全部内容）
    with open(filepath, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # 减去标题行
    
    if verbose:
        # 使用position=0确保进度条在同一行更新
        pbar = tqdm(total=total_rows, desc=description, position=0, leave=True)
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
        if verbose:
            pbar.update(len(chunk))
    
    if verbose:
        pbar.close()
    
    # 合并所有块
    df = pd.concat(chunks, ignore_index=True)
    
    if verbose:
        logger.info(f"{description}完成! 数据形状: {df.shape}")
    
    return df

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="域名嵌入处理器")
    parser.add_argument('--dataset-dir', type=str, 
                       help='数据集目录路径（如未指定则使用config.cfg中的配置）')
    parser.add_argument('--reuse-merged', action='store_true', default=True,
                       help='是否重用已合并的数据文件')
    parser.add_argument('--no-reuse-merged', action='store_false', dest='reuse_merged',
                       help='强制重新合并数据文件')
    parser.add_argument(
        "--split-method",
        type=str,
        choices=["stratified", "random"],
        default="stratified",
        help="数据划分方式: stratified(默认) 或 random"
    )    
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例（默认：0.7）')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='测试集比例（默认：0.2）')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='验证集比例（默认：0.1）')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='输出详细日志信息')
    
    args = parser.parse_args()
    
    # 验证比例参数
    if abs(args.train_ratio + args.test_ratio + args.val_ratio - 1.0) > 1e-10:
        logger.error("train_ratio + test_ratio + val_ratio 必须等于 1.0")
        return 1
    
    # 确定数据集目录
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        try:
            dataset_dir = ConfigManager.read_plot_data_path_config()
        except Exception as e:
            logger.error(f"无法读取配置文件: {e}")
            logger.error("请使用 --dataset-dir 参数指定数据集目录")
            return 1
    
    if not os.path.exists(dataset_dir):
        logger.error(f"数据集目录不存在: {dataset_dir}")
        return 1
    
    # 记录开始时间
    start_time = time.time()
    
    # 准备数据集文件
    try:
        merged_flow_path, merged_session_path, reused = prepare_dataset_files(
            dataset_dir, 
            reuse_prev_merged_data=args.reuse_merged, 
            verbose=args.verbose
        )
    except Exception as e:
        logger.error(f"准备数据集文件失败: {e}")
        return 1
    
    # 读取合并后的数据
    if args.verbose:
        logger.info(f" 读取合并后的数据...")
    try:
        if args.verbose:
            logger.info(f" 读取flow数据...")
        
        # 这里调用read_large_csv_with_progress在处理大数据集，比如CIC-IDS-2018的时候，会爆内存。
        # 因为这个函数内部实现是首先在内存里面存储所有chunks，然后再合并所有块，构建返回的flow_df。
        # 这导致内存需求翻倍了，既需要保存chunks，也要保存完整dataframe
        # flow_df = read_large_csv_with_progress(merged_flow_path, "读取flow数据", args.verbose)
        flow_df = pd.read_csv(merged_flow_path, low_memory=False)

        if args.verbose:
            logger.info(f" 读取session数据...")
            
        session_df = read_large_csv_with_progress(merged_session_path, "读取session数据", args.verbose)
        
        if args.verbose:
            logger.info(f" 数据读取成功:")
            logger.info(f"  - Flow数据形状: {flow_df.shape}")
            logger.info(f"  - Session数据形状: {session_df.shape}")
            logger.info(f"  - Flow数据第一列: {flow_df.columns[0]}")
            logger.info(f"  - Session数据第一列: {session_df.columns[0]}")
            logger.info(f"  - Flow数据列名: {list(flow_df.columns[:5])}...")  # 只显示前5列
            logger.info(f"  - Session数据列名: {list(session_df.columns[:5])}...")
            
    except Exception as e:
        logger.info(f"[ERROR] 读取数据失败: {e}")
        return 1
    
    # 划分会话数据集
    if args.verbose:
        logger.info(f" 划分会话数据集为'train', 'test', 'validate'...")

    try:
        if args.verbose:
            logger.info(f" 使用采样策略: {args.split_method}")

        if args.split_method == "stratified":
            split_session_df = stratified_split_sessions(
                session_df,
                label_col="is_malicious",
                train_ratio=args.train_ratio,
                test_ratio=args.test_ratio,
                val_ratio=args.val_ratio,
                random_state=42,
                verbose=args.verbose
            )
        else:
            split_session_df = random_split_sessions(
                session_df, 
                train_ratio=args.train_ratio,
                test_ratio=args.test_ratio, 
                val_ratio=args.val_ratio,
                random_state=42,
                verbose=args.verbose
            )        

        # 添加详细的split分布检查
        if args.verbose:
            split_counts = split_session_df['split'].value_counts()
            logger.info(f"[DEBUG] Split列详细分布:")
            for split_type, count in split_counts.items():
                percentage = count / len(split_session_df) * 100
                logger.info(f"  - {split_type}: {count} 个会话 ({percentage:.1f}%)")
            
            # 检查是否有未知的split值
            unknown_splits = set(split_session_df['split'].unique()) - {'train', 'test', 'validate'}
            if unknown_splits:
                logger.info(f"[WARNING] 发现未知的split值: {unknown_splits}")
        
        # 保存划分后的会话数据
        split_session_path = os.path.join(dataset_dir, "all_split_session.csv")
        if args.verbose:
            logger.info(f" 保存划分后的会话数据...")
        save_dataframe_with_progress(split_session_df, split_session_path, "保存划分后的会话数据", args.verbose)
        
        if args.verbose:
            logger.info(f" 会话划分完成:")
            logger.info(f"  - 训练集: {len(split_session_df[split_session_df['split'] == 'train'])} 个会话")
            logger.info(f"  - 测试集: {len(split_session_df[split_session_df['split'] == 'test'])} 个会话")
            logger.info(f"  - 验证集: {len(split_session_df[split_session_df['split'] == 'validate'])} 个会话")
            logger.info(f"  - 输出文件: {split_session_path}")
            logger.info(f"  - 新列顺序: {list(split_session_df.columns)}")
            
    except Exception as e:
        logger.info(f"[ERROR] 划分会话数据失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 读取配置
    try:
        if args.verbose:
            logger.info(f" 读取配置...")        
        session_label_id_map = ConfigManager.read_session_label_id_map()
        config = ConfigManager.get_config_parser()  # 获取完整的配置解析器
    except Exception as e:
        logger.info(f"[ERROR] 读取配置失败: {e}")
        return 1
    
    embedded_flow_path = os.path.join(dataset_dir, "all_embedded_flow.csv")

    if not os.path.exists(embedded_flow_path):
        # 创建域名嵌入处理器并执行（只使用训练集的流数据构建共现矩阵）
        processor = DomainEmbeddingProcessor(verbose=args.verbose)
        
        try:
            # 获取训练集的flow uid
            if args.verbose:
                logger.info(f" 获取训练集对应的flow uid...")

            train_flow_uids = get_train_flow_uids(split_session_df, flow_df, 
                                                verbose=args.verbose, 
                                                show_progress=args.verbose)
            
            if args.verbose:
                logger.info(f" 训练集包含 {len(train_flow_uids)} 个流")
                logger.info(f" 开始域名嵌入处理...")
            
            # 处理域名嵌入（使用流水线并行化）
            embedded_flow_df = processor.process(flow_df, split_session_df, session_label_id_map, train_flow_uids)
            
            # 保存嵌入后的数据
            if args.verbose:
                logger.info(f" 保存嵌入后的数据...")
            save_dataframe_with_progress(embedded_flow_df, embedded_flow_path, "保存嵌入后的flow数据", args.verbose)
            
            if args.verbose:
                total_time = time.time() - start_time
                logger.info(f"[SUCCESS] 域名嵌入完成! 总耗时: {total_time:.2f}秒")
                logger.info(f"  - 输入文件: {merged_flow_path}")
                logger.info(f"  - 输出文件: {embedded_flow_path}")
                
                # 根据是否启用层级拆分显示不同的域名数量统计
                if processor.domain_hierarchy_enabled and processor.domain_hierarchy_freq:
                    total_domains = 0
                    for level in range(processor.domain_split_levels):
                        level_count = len(processor.domain_hierarchy_freq.get(f'level_{level}', {}))
                        total_domains += level_count
                        logger.info(f"  - 层级{level}域名数量: {level_count}")
                    logger.info(f"  - 总嵌入域名数量: {total_domains}")
                elif processor.domain_app_freq:
                    logger.info(f"  - 嵌入域名数量: {len(processor.domain_app_freq)}")
                else:
                    logger.info(f"  - 嵌入域名数量: 0")
                    
                logger.info(f"  - 应用类别数量: {processor.num_apps}")
                logger.info(f"  - 使用的训练集流数量: {sum(processor.app_total_flows)}")
            
            return 0
            
        except Exception as e:
            logger.error(f"[ERROR] 处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
    else:
        logger.info(f"已检测到嵌入后的flow文件已存在，跳过域名嵌入阶段: {embedded_flow_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())