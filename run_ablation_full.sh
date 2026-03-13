#!/bin/bash

# =========================================================
# Full Version 完整版：TransGraphNet 消融实验自动化脚本
# 功能：
# 1. 依次运行 Table 5 (Ablation results) 所需的 8 个实验
# 2. 自动提取并整理 console_output.txt 到汇总目录
# =========================================================

# 1. 设置路径
LOG_PARENT_DIR="./logs/ablation_study_features"
RESULT_DIR="$LOG_PARENT_DIR/ablation_console_outputs"
VERSION_NAME="run_v1" # 固定版本号，方便脚本定位文件

# 创建目录
mkdir -p "$RESULT_DIR"

echo "====================================================="
echo "[Ablation study]   消融实验完整流程启动"
echo "[Log directory]    日志总目录:     $LOG_PARENT_DIR"
echo "[Result directory] 结果汇总目录:   $RESULT_DIR"
echo "====================================================="

# 错误处理：遇到错误立即停止
set -e

# 定义通用函数来运行实验和复制结果
run_experiment() {
    local EXP_NAME=$1
    local DESC=$2
    local ARGS=$3

    echo ""
    echo "-----------------------------------------------------"
    echo "▶️ 正在运行任务: $DESC"
    echo "   实验名称: $EXP_NAME"
    echo "-----------------------------------------------------"

    # 运行 Python 训练脚本
    # 注意：logging.version 固定为 run_v1 以便后续复制
    python -u src/models/session_gnn_flow_bert_multiview_ssl/train.py \
        logging.save_dir="$LOG_PARENT_DIR" \
        logging.name="$EXP_NAME" \
        logging.version="$VERSION_NAME" \
        $ARGS

    # 复制结果文件
    SOURCE_FILE="$LOG_PARENT_DIR/$EXP_NAME/$VERSION_NAME/console_output.txt"
    DEST_FILE="$RESULT_DIR/${EXP_NAME}.txt"

    if [ -f "$SOURCE_FILE" ]; then
        cp "$SOURCE_FILE" "$DEST_FILE"
        echo "✅ 结果已保存: $DEST_FILE"
    else
        echo "❌ 警告: 未找到输出文件 $SOURCE_FILE"
        # 不退出，继续尝试下一个实验
    fi
}

# =========================================================
# ������ 实验队列 (共 8 个任务)
# =========================================================

# --- Task 1: Full Model ---
run_experiment "01_full_model" \
    "TransGraphNet (Full Model)" \
    "ablation.enabled=false"

# --- Task 2: Remove Flow Graph (Disable GNN) ---
run_experiment "02_no_graph" \
    "Removing inter-flow relations graph" \
    "ablation.enabled=true ablation.disable_gnn=true"

# --- Task 3: Remove IAT Sequence ---
run_experiment "03_no_iat" \
    "Removing inter-packet IAT features" \
    "ablation.enabled=true ablation.masked_views=[packet_iat_seq]"

# --- Task 4: Remove Packet Length Sequence ---
run_experiment "04_no_pktlen" \
    "Removing packet length sequences" \
    "ablation.enabled=true ablation.masked_views=[packet_len_seq]"

# --- Task 5: Remove Domain Features ---
run_experiment "05_no_domain" \
    "Removing domain features" \
    "ablation.enabled=true ablation.masked_views=[domain_probs]"

# --- Task 6: Remove X509 Features ---
run_experiment "06_no_x509" \
    "Removing X509 features" \
    "ablation.enabled=true ablation.masked_views=[x509_numeric_features,x509_categorical_features,x509_textual_features]"

# --- Task 7: Remove SSL Features ---
run_experiment "07_no_ssl" \
    "Removing SSL features" \
    "ablation.enabled=true ablation.masked_views=[ssl_numeric_features,ssl_categorical_features,ssl_textual_features]"

# --- Task 8: Only Flow Statistics (Mask EVERYTHING else) ---
# 注意：这里我们屏蔽了除 flow_numeric/categorical 以外的所有视图
run_experiment "08_only_stats" \
    "Only flow statistical features" \
    "ablation.enabled=true ablation.masked_views=[packet_len_seq,packet_iat_seq,flow_textual_features,ssl_numeric_features,ssl_categorical_features,ssl_textual_features,x509_numeric_features,x509_categorical_features,x509_textual_features,dns_numeric_features,dns_categorical_features,dns_textual_features,domain_probs]"

echo ""
echo "====================================================="
echo "[Experiment Queue] 所有 8 个消融实验已执行完毕！"
echo "[Result Summary]   请前往以下目录查看汇总结果 (txt文件):"
echo "   $RESULT_DIR"
echo "====================================================="