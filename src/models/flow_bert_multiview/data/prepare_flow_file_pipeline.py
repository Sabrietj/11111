import re
import torch
import ast
import os
import pandas as pd
import sys
import logging
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import time
import uuid
import hashlib

BENIGN_PREFIXES = ("benign", "normal", "legitimate")  # 可根据实际数据集调整

utils_path = os.path.join(os.path.dirname(__file__),  '..', '..', '..', 'utils')
sys.path.insert(0, utils_path) 
# 设置日志
from logging_config import setup_preset_logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.INFO)

def short_tag(tag: str, max_len: int = 48):
    if len(tag) <= max_len:
        return tag
    h = hashlib.md5(tag.encode("utf-8")).hexdigest()[:8]
    return tag[:max_len] + "_" + h

def is_rank0():
    # ✅ 1) 优先用环境变量（DDP 初始化前就可用）
    r = os.environ.get("RANK", None)
    if r is not None:
        return int(r) == 0

    # ✅ 2) dist 已初始化后再用 torch.distributed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    # ✅ 3) 其他情况（单进程）
    return True

def wait_for_file_stable(path: str, done_path: str = None, timeout=3600*24, interval=1.0, stable_checks=3):
    '''
    等待该文件24小时
    '''
    t0 = time.time()
    last_size = -1
    stable_cnt = 0

    while True:
        if done_path is not None and not os.path.exists(done_path):
            ok = False
        else:
            ok = os.path.exists(path)

        if ok:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = -1

            if size == last_size and size > 0:
                stable_cnt += 1
            else:
                stable_cnt = 0
            last_size = size

            if stable_cnt >= stable_checks:
                return True

        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"[wait_for_file_stable timeout] "
                f"path={path}, exists={os.path.exists(path)}, "
                f"done_path={done_path}, done_exists={os.path.exists(done_path) if done_path else None}"
            )

        time.sleep(interval)

def prepare_sampled_data_files(cfg):
    """
    返回:
      cfg: 更新后的配置对象，包含采样/过滤/合并后的数据文件路径
    """
    # ============================================================
    # 1️⃣ 根据配置进行随机采样（控制规模）
    # ============================================================
    if cfg.data.sampling.enabled and cfg.data.sampling.sample_ratio < 1.0: 
        if cfg.data.split_mode == "flow":
            flow_data_path = cfg.data.flow_data_path
            sampled_flow_data_path = flow_data_path.replace(
                ".csv",
                f".sampled_{cfg.data.sampling.sample_ratio}_{cfg.data.random_state}.csv"
            )

            if not os.path.exists(sampled_flow_data_path):
                downsample_benign_only = cfg.datasets.downsample_benign_only if hasattr(cfg.datasets, "downsample_benign_only") else False
                label_column = cfg.data.multiclass_label_column if hasattr(cfg.data, "multiclass_label_column") else "label"
                sample_flow_csv(
                    input_flow_csv=flow_data_path,
                    output_flow_csv=sampled_flow_data_path,
                    ratio=cfg.data.sampling.sample_ratio,
                    seed=cfg.data.random_state,
                    downsample_benign_only=downsample_benign_only,
                    label_column=label_column,
                )

            cfg.data.flow_data_path = sampled_flow_data_path
            logger.info(f"使用随机下采样后的网络流数据文件: {sampled_flow_data_path}")

        elif cfg.data.split_mode == "session":
            # 随机采样session文件
            session_data_path = cfg.data.session_split.session_split_path
            sampled_session_data_path = session_data_path.replace(
                ".csv", f".sampled_{cfg.data.sampling.sample_ratio}_{cfg.data.random_state}.csv"
            )

            if not os.path.exists(sampled_session_data_path):
                sample_session_csv(
                    input_session_csv=session_data_path,
                    output_session_csv=sampled_session_data_path,
                    ratio=cfg.data.sampling.sample_ratio,
                    seed=cfg.data.random_state,
                )

            cfg.data.session_split.session_split_path = sampled_session_data_path

            # 根据session文件的采样结果，确定性采样flow的结果
            sampled_flow_data_path = cfg.data.flow_data_path.replace(
                ".csv", f".from_session_{cfg.data.sampling.sample_ratio}_{cfg.data.random_state}.csv"
            )

            if not os.path.exists(sampled_flow_data_path):                
                # 流级过滤（chunk 版，避免一次性读爆内存）
                filter_flow_csv_by_session_flow_uid_list(
                    input_flow_csv=cfg.data.flow_data_path,
                    output_flow_csv=sampled_flow_data_path,
                    sampled_session_csv=sampled_session_data_path,
                    flow_uid_list_column=cfg.data.session_split.flow_uid_list_column,
                )

            cfg.data.flow_data_path = sampled_flow_data_path

        else:
            raise ValueError(f"不支持的 split_mode={cfg.data.split_mode}，无法进行采样")

    # ============================================================
    # 2️⃣ 过滤端口和服务（控制任务语义）
    # ============================================================
    exclude_ports = OmegaConf.select(cfg, "datasets.exclude_ports")
    exclude_services = OmegaConf.select(cfg, "datasets.exclude_services")

    if exclude_ports or exclude_services:
        tag_parts = []
        if exclude_ports:
            tag_parts.append("port_" + "_".join(map(str, sorted(exclude_ports))))
        if exclude_services:
            tag_parts.append("svc_" + "_".join(sorted(exclude_services)))

        tag = "__".join(tag_parts)

        filtered_path = cfg.data.flow_data_path.replace(
            ".csv", f".filtered_{tag}.csv"
        )

        if not os.path.exists(filtered_path):
            filter_flow_csv_by_port_and_service(
                input_flow_csv=cfg.data.flow_data_path,
                output_flow_csv=filtered_path,
                exclude_ports=exclude_ports,
                exclude_services=exclude_services,
            )

        cfg.data.flow_data_path = filtered_path

    # ============================================================
    # 3️⃣ 过滤 excluded_classes（控制任务语义）
    # （如 CIC-IDS-2017数据集的 Infiltration / Heartbleed 攻击等）
    # ============================================================
    excluded_classes = OmegaConf.select(cfg, "datasets.excluded_classes")
    label_column = OmegaConf.select(cfg, "data.multiclass_label_column")

    if excluded_classes and label_column is not None:
        excluded_tag_raw = "_".join(sorted(excluded_classes))
        excluded_tag_raw = re.sub(r"[^a-zA-Z0-9_]", "_", excluded_tag_raw)
        excluded_tag_hash = short_tag(excluded_tag_raw)  
        filtered_flow_data_path = cfg.data.flow_data_path.replace(
            ".csv", f".filtered_{excluded_tag_hash}.csv"
        )

        if not os.path.exists(filtered_flow_data_path):
            logger.info(f"过滤以下 excluded_classes: hash={excluded_tag_hash}, classes={excluded_classes}")
            filter_flow_csv_by_excluded_classes(
                input_flow_csv=cfg.data.flow_data_path,
                output_flow_csv=filtered_flow_data_path,
                label_column=label_column,
                excluded_classes=list(excluded_classes),
            )

        cfg.data.flow_data_path = filtered_flow_data_path
        logger.info(
            f"使用过滤 excluded_classes 后的 flow 数据文件: "
            f"{filtered_flow_data_path}"
        )

    # ============================================================
    # 4️⃣ 合并 merged_classes（label canonicalization）
    # ============================================================
    merged_classes = OmegaConf.select(cfg, "datasets.merged_classes")

    if merged_classes and label_column is not None:
        merged_tag = "_".join(sorted(merged_classes.keys()))
        merged_tag   = re.sub(r"[^a-zA-Z0-9_]", "_", merged_tag)

        merged_flow_data_path = cfg.data.flow_data_path.replace(
            ".csv", f".merged_{merged_tag}.csv"
        )

        if not os.path.exists(merged_flow_data_path):
            logger.info(f"合并 merged_classes: {dict(merged_classes)}")
            merge_flow_csv_classes(
                input_flow_csv=cfg.data.flow_data_path,
                output_flow_csv=merged_flow_data_path,
                label_column=label_column,
                merged_classes=OmegaConf.to_container(merged_classes),
            )

        cfg.data.flow_data_path = merged_flow_data_path
        logger.info(
            f"使用 merged_classes 后的 flow 数据文件: {merged_flow_data_path}"
        )

    return cfg


def sample_flow_csv(
    input_flow_csv: str,
    output_flow_csv: str,
    ratio: float,
    seed: int,
    downsample_benign_only=False,
    label_column="label",
    chunksize: int = 100_000,
):
    tmp_path = output_flow_csv + ".tmp"
    done_path = output_flow_csv + ".done"

    if is_rank0():
        rng = np.random.default_rng(seed)
        first = True
        total_seen = 0
        total_kept = 0

        reader = pd.read_csv(input_flow_csv, chunksize=chunksize, low_memory=False)
        reader = tqdm(reader, desc="Sampling flows", unit="chunk")

        for chunk in reader:
            total_seen += len(chunk)

            if downsample_benign_only and label_column is not None:
                if label_column not in chunk.columns:
                    raise KeyError(f"{label_column} not in CSV")
                
                labels = chunk[label_column].astype(str).str.strip().str.lower()
                benign_mask = labels.str.startswith(BENIGN_PREFIXES)
                random_mask = rng.random(len(chunk)) < ratio

                # benign 才采样
                mask = (~benign_mask) | (benign_mask & random_mask)

            else:
                mask = rng.random(len(chunk)) < ratio

            sampled = chunk[mask]
            kept = len(sampled)
            total_kept += kept

            if kept == 0:
                continue

            sampled.to_csv(
                tmp_path,
                mode="w" if first else "a",
                header=first,
                index=False,
            )

            first = False

        if first:
            raise RuntimeError(
                f"sample_flow_csv: ratio={ratio} 导致采样结果为空"
            )

        logger.info(
            f"[sample_flow_csv] kept {total_kept} / {total_seen} flows "
            f"({total_kept / max(total_seen, 1) * 100:.2f}%)"
            f"(ratio={ratio}, benign_only={downsample_benign_only})"            
        )

        # 👉 原子替换 + done
        os.replace(tmp_path, output_flow_csv)
        open(done_path, "w").close()

    else:    
        # 👉 所有 rank 等
        wait_for_file_stable(output_flow_csv, done_path=done_path)        


def sample_session_csv(
    input_session_csv: str,
    output_session_csv: str,
    ratio: float,
    seed: int,
):
    tmp_path = output_session_csv + ".tmp"
    done_path = output_session_csv + ".done"

    if is_rank0():
        session_df = pd.read_csv(input_session_csv, low_memory=False)
        session_df = session_df.sample(
            frac=ratio, #cfg.data.sampling.sample_ratio,
            random_state=seed, #cfg.data.random_state,
        )
        session_df.to_csv(tmp_path, index=False)

        # 👉 原子替换 + done
        os.replace(tmp_path, output_session_csv)
        open(done_path, "w").close()
    
    else:
        # 👉 所有 rank 等
        wait_for_file_stable(output_session_csv, done_path=done_path)


def filter_flow_csv_by_session_flow_uid_list(
    input_flow_csv: str,
    output_flow_csv: str,
    sampled_session_csv: str,
    flow_uid_list_column: str,
    chunksize: int = 100_000,
):
    tmp_path = output_flow_csv + ".tmp"
    done_path = output_flow_csv + ".done"

    if is_rank0():        
        session_df = pd.read_csv(sampled_session_csv, low_memory=False)
        sampled_flow_uids = set()
        for v in session_df[flow_uid_list_column]:
            if isinstance(v, str):
                v = ast.literal_eval(v)
            sampled_flow_uids.update(v)

        if len(sampled_flow_uids) == 0:
            raise ValueError(
                f"cfg.data.sampling.sample_ratio 配置过低，导致 flow_uids 为空，无法训练"
            )
                
        first = True
        total_seen = 0
        total_kept = 0

        reader = pd.read_csv(input_flow_csv, chunksize=chunksize, low_memory=False)
        reader = tqdm(reader, desc="Filtering flows by uid", unit="chunk")

        for chunk in reader:
            total_seen += len(chunk)
            filtered = chunk[chunk["uid"].isin(sampled_flow_uids)]
            kept = len(filtered)
            total_kept += kept

            if kept == 0:
                continue

            filtered.to_csv(
                tmp_path,
                mode="w" if first else "a",
                header=first,
                index=False,
            )
            
            first = False

        if first:
            raise RuntimeError(
                "filter_flow_csv_by_uid: 过滤后结果为空，请检查 keep_uids 是否正确"
            )
        
        logger.info(
            f"[filter_flow_csv_by_uid] kept {total_kept} / {total_seen} flows "
            f"({total_kept / max(total_seen, 1) * 100:.2f}%)"
        )

        # 👉 原子替换 + done
        os.replace(tmp_path, output_flow_csv)
        open(done_path, "w").close()

    else:
        # 👉 所有 rank 等
        wait_for_file_stable(output_flow_csv, done_path=done_path)


def filter_flow_csv_by_port_and_service(
    input_flow_csv: str,
    output_flow_csv: str,
    exclude_ports: list = None,
    exclude_services: list = None,
    chunksize: int = 100_000,
):
    
    tmp_path = output_flow_csv + ".tmp"
    done_path = output_flow_csv + ".done"

    if is_rank0():
        exclude_ports = set(exclude_ports or [])
        exclude_services = set(exclude_services or [])

        first = True
        total_kept = 0
        total_seen = 0

        reader = pd.read_csv(input_flow_csv, chunksize=chunksize, low_memory=False)

        # 仅 rank0 显示进度条（避免 DDP / 多进程刷屏）
        reader = tqdm(
            reader,
            desc="Filtering flows by port/service",
            unit="chunk",
        )

        for chunk in reader:
            total_seen += len(chunk)

            mask = pd.Series(True, index=chunk.index)

            if exclude_ports:
                mask &= ~chunk["conn.id.resp_p"].isin(exclude_ports)

            if exclude_services:
                mask &= ~chunk["conn.service"].isin(exclude_services)

            filtered = chunk[mask]
            kept = len(filtered)
            total_kept += kept

            if kept == 0:
                continue

            filtered.to_csv(
                tmp_path,
                mode="w" if first else "a",
                header=first,
                index=False,
            )
            first = False

        if first:
            logger.warning("filter_flow_csv_by_port_and_service: 过滤后结果为空")
        else:
            logger.info(
                f"[filter_flow_csv_by_port_and_service] "
                f"kept {total_kept} / {total_seen} flows "
                f"({total_kept / max(total_seen, 1) * 100:.2f}%)"
            )

        os.replace(tmp_path, output_flow_csv)
        open(done_path, "w").close()

    else:
        wait_for_file_stable(output_flow_csv, done_path=done_path)


def filter_flow_csv_by_excluded_classes(
    input_flow_csv: str,
    output_flow_csv: str,
    label_column: str,
    excluded_classes: list,
    chunksize: int = 100_000,
):
    tmp_path = output_flow_csv + ".tmp"
    done_path = output_flow_csv + ".done"

    if is_rank0():
        first = True
        total_seen = 0
        total_kept = 0

        reader = pd.read_csv(input_flow_csv, chunksize=chunksize, low_memory=False)
        if is_rank0():
            reader = tqdm(reader, desc="Filtering flows by excluded classes", unit="chunk")

        for chunk in reader:
            total_seen += len(chunk)
            filtered = chunk[~chunk[label_column].isin(excluded_classes)]
            kept = len(filtered)
            total_kept += kept

            if kept == 0:
                continue

            filtered.to_csv(
                tmp_path,
                mode="w" if first else "a",
                header=first,
                index=False,
            )
            first = False

        if first:
            logger.warning(
                "filter_flow_csv_by_excluded_classes: 未命中任何 excluded_classes，对结果无影响"
            )
        else:
            logger.info(
                f"[filter_flow_csv_by_excluded_classes] "
                f"kept {total_kept} / {total_seen} flows "
                f"({total_kept / max(total_seen, 1) * 100:.2f}%)"
            )

        os.replace(tmp_path, output_flow_csv)
        open(done_path, "w").close()
    
    else:
        wait_for_file_stable(output_flow_csv, done_path=done_path)


def merge_flow_csv_classes(
    input_flow_csv: str,
    output_flow_csv: str,
    label_column: str,
    merged_classes: dict,
    chunksize: int = 100_000,
):
    """
    将 merged_classes 中的多个原始类别，统一替换为新的类别名
    """
    tmp_path = output_flow_csv + ".tmp"
    done_path = output_flow_csv + ".done"

    if is_rank0():

        # 构造反向映射：old_label -> new_label
        merge_map = {}
        for new_label, old_labels in merged_classes.items():
            for old in old_labels:
                merge_map[old] = new_label

        first = True
        total_seen = 0

        reader = pd.read_csv(input_flow_csv, chunksize=chunksize, low_memory=False)
        if is_rank0():
            reader = tqdm(reader, desc="Merging flow classes", unit="chunk")

        for chunk in reader:
            if label_column not in chunk.columns:
                raise KeyError(f"label_column={label_column} 不在 CSV 列中")

            total_seen += len(chunk)

            chunk[label_column] = chunk[label_column].replace(merge_map)

            chunk.to_csv(
                tmp_path,
                mode="w" if first else "a",
                header=first,
                index=False,
            )
            first = False

        if first:
            logger.warning("merge_flow_csv_classes: 本次数据中未命中任何 merged_classes，对结果无影响")
        else:
            logger.info(f"[merge_flow_csv_classes] processed {total_seen} flows")

        os.replace(tmp_path, output_flow_csv)
        open(done_path, "w").close()
    
    else:
        wait_for_file_stable(output_flow_csv, done_path=done_path)
