预训练模型已上传百度网盘。
```
通过网盘分享的文件：models_hub
链接: https://pan.baidu.com/s/1j9Vq-E6AkunWZTLG26ck3Q?pwd=7ee5 
提取码: 7ee5
```

# 🔵 **1. bert-base-uncased**

### 📌 简介

BERT 系列中最经典、最广泛使用的基础模型（Base 版）。
采用 12 层 Transformer、768 隐藏维度、12 个注意力头，是 NLP 任务的标准基线。

### 📐 模型结构

* **Layers**: 12
* **Hidden size**: 768
* **Attention heads**: 12
* **Parameters**: ~110M
* **Tokenizer**: WordPiece（不区分大小写）

### 🎯 适用场景

* 语义理解（文本分类、QA、句子推理）
* 句子嵌入
* 分词、关系抽取、命名实体识别等任务
* 中等长度文本（<512 tokens）

### 🔽 下载地址（HuggingFace 官方）

[https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased)

或者直接使用：

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

---

# 🟣 **2. bert_uncased_L-4_H-256_A-4（BERT-Small）**

### 📌 简介

这是 Google 官方发布的 **小型轻量版本 BERT**（通常称为 *Small BERT*）。
它有更少的层数、更小的隐藏维度，但保持完整的 BERT 结构。

属于 Small 系列中常用的变体之一。

### 📐 模型结构

* **Layers**: 4
* **Hidden size**: 256
* **Attention heads**: 4
* **Parameters**: ~14M
* **大小约为 bert-base 的 1/8**

### 🎯 适用场景

* 延迟敏感系统
* 轻量级文本嵌入
* 分类任务的高性价比模型
* 工业部署、移动端、边缘设备
* 你的多视图模型中作为文本 encoder 非常适合

### 🔽 下载地址（Small BERT 官方仓库）

Google 维护的一系列 lightweight BERT 模型：
[https://huggingface.co/google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4)

直接加载：

```python
model = AutoModel.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
```

---

# 🟢 **3. bert_uncased_L-2_H-128_A-2（BERT-Mini）**

### 📌 简介

这是 **最轻量**的 Small BERT 系列（Mini BERT），仅有 **2 层** Transformer，
是速度最快、参数最少的 BERT 变体。

它在你的多视图流量模型中表现非常适合，因为文本字段（域名、SNI 等）本身很短。

### 📐 模型结构

* **Layers**: 2
* **Hidden size**: 128
* **Attention heads**: 2
* **Parameters**: ~4.4M（比 bert-base 小 25 倍）
* **极端轻量级，推理速度极快**

### 🎯 适用场景

* 网络流量、域名、短文本编码
* 多模态模型中的“辅助文本模态”
* 移动端部署、实时系统
* 大 batch size 训练
* 模型蒸馏或知识迁移任务的学生模型

### 🔽 下载地址（Small BERT 官方仓库）

[https://huggingface.co/google/bert_uncased_L-2_H-128_A-2](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)

加载方式：

```python
model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
```

---

# 🧠 **三个模型的快速比较（表格）**

| 模型                    | 层数 | 隐藏维度 | 参数量   | 速度     | 精度   | 典型用途        |
| --------------------- | -- | ---- | ----- | ------ | ---- | ----------- |
| **bert-base-uncased** | 12 | 768  | 110M  | 慢      | 高    | 复杂 NLP      |
| **L-4_H-256_A-4**     | 4  | 256  | ~14M  | 快      | 中等偏高 | 轻量 NLP      |
| **L-2_H-128_A-2**     | 2  | 128  | ~4.4M | **最快** | 中等   | 嵌入、短文本、实时系统 |
