# README

[TOC]

多视图学习（Multi-View Learning）是机器学习领域的一个重要分支，核心思想是利用数据在不同 “视角” 下的表征信息，通过融合或协同学习来提升模型性能，尤其适用于单视角信息不完整、有噪声或存在歧义的场景。“视图（View）” 的定义：同一数据对象在不同维度、来源或处理方式下形成的特征集合，称为该数据的一个 “视图”。例如，一张图片的 “像素 RGB 值” 是一个视图，而通过卷积提取的 “高层语义特征” 是另一个视图；一个人的 “文本评论数据” 是一个视图，“行为点击数据” 是另一个视图。在网络流量分析领域，也可以结合不同视图的特征。比如，可以将网络数据包分为数据包头、数据包载荷，并结合数据包的流特征，TLS握手字段特征，DNS域名特征等等，从不同视角提供输入。例如，基于 MLP-Mixer 的多视图多标签神经网络，就将包头、包体以及流特征作为不同的视图输入，以学习不同场景下的特征表示。基于[MPAF的多阶段深度学习](./docs/DNS/MPAF_Encrypted_Traffic_Classification_With_Multi-Phase_Attribute_Fingerprint.pdf)，就将DNS域名查询、TLS握手特征和数据包传输序列作为不同的视图。本项目的具体功能如下：
-  多视图网络流量分类框架：面向多种安全数据集（[CIC-IDS-2017](./dataset/CIC-IDS-2017/README.md)、[CIC-IDS-2018](./dataset/CIC-IDS-2018/README.md)、[CIC-IoMT-2024](./dataset/CIC-IoMT-2024/README.md)、[ISCX-VPN-nonVPN-2016](./dataset/ISCX-VPN-nonVPN-2016/README.md)），融合流统计、数据包序列、TLS握手、DNS域名、X509等多视角特征。注意：前三个是面向网络入侵检测的黑流量分类数据集；第四个是面向网络应用分类的白流量分类数据集。其他的白流量应用分类数据集还有UTMobilenet-2021、MIRAGE-2019、MIRAGE-2023。但是这些数据集里面只有[ISCX-VPN-nonVPN-2016](./dataset/ISCX-VPN-nonVPN-2016/README.md)里面有pcap文件（覆盖Skype、Facebook、Email等良性应用在明文流量和通过VPN隧道后两种状态下的流量），其它的只有处理后的csv和json，提取不了异构特征。
- 数据全流程工具链：提供 Zeek/FlowMeter 日志打标、特征提取（flow/session CSV）、类别特征嵌入、数据泄露分析，到会话流关系图（DGL）构建的完整脚本。
图与可视化：将会话构造成多视图节点特征的 DGL 图（支持节点/边分布、标签分布和 burst 图可视化）。
- 模型训练：涵盖 flow 级混合模型（BERT+GAT，多种自监督变体 AE/MLM/Seq2Stat）及在研的 session 级 TransGraphNet；可下载预训练 BERT、小模型并配置 batch 大小。
- 评测与对比：提供精度、召回、F1、ROC/PR-AUC等指标对比，给出不同场景下的模型选型建议，并附有 Appscanner/FS-Net/FG-Net/GraphDApp 等 baseline 参考。


本README涵盖环境、数据准备、训练、导出最佳模型、评测与可视化等流程。请按需修改与你的真实路径/数据一致。

## 1. 项目简介

该项目支持多个网络安全/流量数据集上的多视图表示学习与分类，包括数据集：`CIC-IDS-2017`、`CIC-AndMal2017`、`USTC-TFC2016`、`UNSW-IoT`、`CIC-IoT-2023`。

## 2. 目录结构（建议）

各位同学多做些开发工作。现在有AI Coder，读代码，写代码效率应该还是可以提升不少。用
```shell
tree -d -L 2 -I "__pycache__|.git|logs|figs|shap_results|outputs|src_qyf|models_old|hyper_optimus"
```
可以显示本项目的子目录结构。其功能解释如下：
```python
.
├── dataset                         # 原始流量Pcap文件、Zeek log解析代码和日志、conn.log打标签代码
│   ├── CIC-IDS-2017                # 加拿大网络安全研究院CIC的网络入侵检测数据集2017
│   ├── CIC-IDS-2018                # 加拿大网络安全研究院CIC的网络入侵检测数据集2018
│   ├── CIC-IoMT-2024               # 加拿大网络安全研究院CIC的医疗物联网入侵检测数据集2024
│   ├── CIC-IoT-2023                # 加拿大网络安全研究院CIC的物联网网络入侵检测数据集2023
│   └── ISCX-VPN-nonVPN-2016        # CIC的前身ISCX研究室的VPN和NonVPN网络应用分类数据集
├───doc                             # 参考论文
│   ├───DNS                         # DNS 流量解析和域名嵌入
│   ├───FlowRelationGraph           # 五元组流的关系图构建和表征
│   ├── HostRelationGraph           # 主机关系图的构建和表征
│   ├───HTTP                        # 应用层HTTP消息分析
│   ├───Open-World                  # 流量分析领域的开放世界机器学习
│   ├───TCP-IP-FiveTupleFlow        # 五元组流的加密流量分析
│   └───TLS                         # TLS握手字段特征的深度表征
├── models_hub                      # 下载的预训练大模型
│   ├── bert-base-uncased           # 英文小写 BERT 基础版：12 层、768 维、12 头注意力模型
│   ├── bert_uncased_L-2_H-128_A-2  # 英文小写 BERT 小型：2 层、128 维、2 头注意力模型
│   └── bert_uncased_L-4_H-256_A-4  # 英文小写 BERT 中型：4 层、256 维、4 头注意力模型
├───processed_data                  # 预处理后的csv数据文件和图数据文件
└── src
    ├── build_session_graph         # 网络会话的流关系图构建
    ├── draw_session_graph          # 网络会话的流关系图绘制
    ├── embed_feature               # 网络流特征嵌入
    ├── extract_feature             # 网络流特征提取
    ├── models                      # 面向网络流量分类的深度学习模型
    │   ├── flow_autoencoder        # 基于自编码器的五元组流分类
    │   ├── flow_bert_multiview     # 基于bert和mlp的多视图特征融合的五元组流分类
    │   ├── flow_bert_multiview_ssl # 基于bert和自编码器的多视图特征融合的五元组流分类
    │   ├── flow_bert_multiview_ssl_mlm          # 基于bert和自编码器，以及mlm数据包序列自监督学习，的多视图特征融合的五元组流分类
    │   ├── flow_bert_multiview_ssl_seq2stat     # 基于bert和自编码器，以及seq2pat数据包序列自监督学习，的多视图特征融合的五元组流分类
    │   └── session_gnn_flow_bert_multiview_ssl  # 基于gnn、bert和自编码器的多视图特征会话分类
    └── utils                       # 通用工具类
```

## 3. 环境依赖

* Python 3.12+
* PyTorch（支持GPU可选）
* scikit-learn, joblib, matplotlib, numpy
* DGL图神经网络
* umap（用于可视化），SciPy（GMM）
* 其他依赖查看[requirements.txt](./requirements.txt)

### 3.1 创建虚拟环境

当物理主机未安装Conda环境的时候，在 Linux 或 macOS 系统 下，可以用下面的命令创建虚拟环境。
```bash
# 在代码目录下执行
python3 -m venv myvenv
# 启用
source myvenv/bin/activate
```

当物理主机已安装Conda环境的时候，可以用下面的命令创建虚拟环境。其中，`3.x`主流版本目前是`3.12`。
```bash
conda create --name code-multiview-network-traffic-classification-model python=3.12
conda activate code-multiview-network-traffic-classification-model
```

在学校的大数据服务中心，只能在虚拟主机上跑程序，需要先导入对应的环境。（使用说明：https://sc.seu.edu.cn/docs/hpc/contents.html）。
```bash
# 跳转到cuda环境
ssh fat01
# 加载相关依赖
module load anaconda3
# 其余步骤与上面相同

# 如果需要cmake环境来源代码安装DGL，那么可以module av，看下cmake的版本，然后module load 一下
module load cmake-3.24.3
```

### 3.2 安装Pytorch和各种依赖

首先，安装Pytorch。安装CUDA版本的Pytorch，可以用如下的命令（在易安联公司的Nvidia A800服务器上已验证）。安装CPU版本的Pytorch，可以直接`pip install -r requirements.txt`。
```shell
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# pip install --force-reinstall torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```
验证安装
```shell
pip list
python -c "import torch; print(torch.version.cuda)"
```

其次，安装各种依赖包。
```shell
pip install -r requirements.txt
```

然后，要记着下载预训练的transformer模型到`models_hub`目录；具体参考：[models_hub/README.md](models_hub/README.md)。

最后，项目还需要DGL图神经网络包。其安装最为复杂，参考[这个页面安装DGL](./install-dgl.md)。注：在二楼办公室的Nvidia 2080ti的服务器上，已经有了名字是`minteen`的环境，都装好了需要的包括dgl在内的软件包。
```shell
pip install "https://data.dgl.ai/wheels/torch-2.4/cu124/dgl-2.4.0%2Bcu124-cp312-cp312-manylinux1_x86_64.whl"
```
也可以手动下载wheel文件，然后用pip命令安装。注意：这个链接[dgl wheels repo](https://data.dgl.ai/wheels/repo.html)有所有的wheels下载。在东南大学的大数据中心要用`module av`先看看modules列表，然后`module load cuda-12.4`加载cuda模块。并通过`nvcc --version`或者`nvidia-smi`命令确认当前环境的cuda版本号。
```
pip install dgl-2.4.0+cu124-cp312-cp312-manylinux1_x86_64.whl
```
注意：cpxxx指的是python的版本号。如果python环境是3.12，那么就是cp312；如果python环境是3.11，那么就是cp311。


## 4. 日志预处理

* ⚠️ 重要提示：Windows 系统 UTF-8 设置。确保 Windows 系统需要设置本地字符集编码为 UTF-8。**设置方法如下：**
  1. 打开 ​​设置​​ → ​​时间和语言​​ → ​​语言​​和区域
  2. 在右侧找到 ​​管理语言设置​​
  3. 在 `语言​​` 选项卡中，勾选 ​​"使用 Unicode UTF-8 提供全球语言支持"​​
  4. 点击 `​​确定​​` 并重启计算机

* **配置数据集路径**：打开 [src/utils/config.cfg](src/utils/config.cfg)，修改以下参数：
  * 此刻正在激活的数据集：
    ```conf
    # ==========================================
    # 数据集激活配置 - 只需修改这里
    # ==========================================
    # 在要使用的数据集名称前加上 ACTIVE_ 前缀
    ACTIVE_DATASET = CIC-IDS-2017
    # ACTIVE_DATASET = CIC-AndMal2017
    # ACTIVE_DATASET = USTC-TFC2016
    # ACTIVE_DATASET = CIC-IoT-2023
    # ACTIVE_DATASET = CTU-13
    ```
  * `path_to_dataset`：设置为你的数据集所在目录；
  * `plot_data_path`：设置为处理后数据的保存目录；
  * `session_tuple_mode`：根据需求选择会话元组模式。要设置这个配置项是因为不同数据集的session怎么定义合适，预先并不知道。有这个配置项能够灵活一些。有的数据集适合session定义成一元组，为了客户端画像（比如CIC-AndMal-2017数据集，只有目录标记的比较粗略的客户端标签）；有的适合二元组；有的适合五元组（比如CIC-IDS-2017数据集，有五元组流的类别，就适合把session定义成五元组）。
    ```conf
    # 控制 session_tuple 的组成方式：
    # 对应含义：
    #   "srcIP"                  => 一元组 (源IP,)
    #   "dstIP"                  => 一元组 (目的IP,)
    #   "srcIP_dstIP"            => 二元组 (源IP, 目标IP)
    #   "srcIP_dstIP_proto"      => 三元组 (源IP, 目标IP, 协议)
    #   "srcIP_dstIP_dstPort"      => 三元组 (源IP, 目标IP, 目的端口)
    #   "srcIP_dstIP_dstPort_proto" => 四元组 (源IP, 目标端口, 协议) [默认]
    #   "srcIP_srcPort_dstIP_dstPort_proto" => 五元组 (源IP, 源端口，目标端口, 协议) [默认]    
    session_tuple_mode = srcIP
    ```
  * 另外，每个数据集label多分类到多么细致，也需要预先配置。比如下面的设置，只`CIC-AndMal2017`分类到[安卓软件的category级别，不需要分到更细致的family级别](dataset/CIC-AndMal2017/README.md)。
    ```conf
    # CIC-AndMal-2017
    session_label_id_map = benign:0, adware:1, ransomware:2, scareware:3, smsmalware:4
    ```
  * 更多选项参考 [src/utils/config.cfg](src/utils/config.cfg) 文件。

* **生成标签文件**：运行以下命令：
  ```bash
  python ./src/extract_feature/__label__.py
  ```
  运行后，每个 `conn.log` 都会生成对应的 `conn_label.log` 文件。注意：对CIC-IDS-2017数据集的打标签功能还有Bug没修复。可以暂时用数据集目录下面的史大成同学的打标签程序。
  ```bash
  cd dataset/CIC-IDS-2017
  python3 label_conn_log.py
  cd -
  ```
  在本项目中，**五元组流（five-tuple flow）** 与 **连接（connection）** 这两个术语可以互换使用。flowmeter日志可以看成是conn日志的流统计特征和数据包序列特征的深度扩展。这里可以找到[flowmeter log详细字段说明](./doc/flowmeter.log字段说明.md)。如果希望对`flowmeter.log`同样的方式打标签，那么可以进一步用如下的命令。
  ```bash
  python src/extract_feature/__label__.py --conn_filename flowmeter.log
  ```
  
  **注意**：CIC-AndMal-2017数据集，已知哪些主机 IP 地址为恶意软件的受感染主机或正常主机，并以此分出了不同的文件夹。因此，我们用文件夹的名字做五元组流的Label。
  
* **提取特征**：运行以下命令：

  ```bash
  python src/extract_feature/__main__.py --clean-plot-dir --force
  ```

  执行后，`plot_data_path` 指定的目录中将生成网络会话（session）样本，用于后续模型训练。**以 CIC-AndMal2017 数据集为例**：原始数据集包含五类 Android 应用流量样本：**Adware（广告软件）**、**Benign（良性样本）**、**Ransomware（勒索软件）**、**Scareware（恐吓软件）** 和 **SMSMalware（短信木马）**。每一类样本都存放在对应的文件夹中。文件夹下的子文件夹代表不同的恶意软件家族。例如：

  | 类别            | 家族名             | 样本数 |   | 类别             | 家族名     | 样本数 |
  | ------------- | --------------- | --- | - | -------------- | ------- | --- |
  | **Adware**    | Dowgin          | 10  |   | **Ransomware** | Charger | 10  |
  |               | Ewind           | 10  |   |                | Jisut   | 10  |
  |               | …               | …   |   |                | …       | …   |
  | **Scareware** | AndroidDefender | 17  |   | **SMSMalware** | BeanBot | 9   |
  |               | AndroidSpy.277  | 6   |   |                | Biige   | 11  |
  |               | …               | …   |   |                | …       | …   |

  特征提取程序会对这些原始样本进行处理，并在 `progressed_data/CIC-AndMal2017` 目录下生成五个文件夹，保持与原始数据集一致的顶层分类名（**Adware**、**Benign**、**Ransomware**、**Scareware**、**SMSMalware**）。每个分类文件夹中会生成两种类型的 CSV 文件：**流记录（flow records）** 和 **会话记录（session records）**。例如，在 `SMSMalware` 文件夹下，`Biige` 家族对应的输出文件为 `Biige-flow.csv` 和 `Biige-session.csv`。 每个会话由多个流组成，流通过 Zeek 的 UID 进行标识（即 `Biige-flow.csv` 的键值）。流记录文件中的特征来源于多个 Zeek 日志类型的组合：`conn.log`、`ssl.log`、`x509.log` 和 `dns.log`（其中目的 IP 地址通过反向 DNS 查询解析得到）。

* **DNS查询统计分析**：采集`{plot_data_path}/all_flow.csv`文件，深度分析DNS查询数据，提供多维度统计。
   * DNS查询维度：统计每个DNS查询涉及的Class和Family数量
   * Class级别：按流量分类统计DNS查询模式
   * Family级别：按流量家族统计DNS查询行为
   
    
  
  **Label字段统计分析**：分析label字段的分布情况，支持class_family格式。自动基于输入文件名生成输出文件（如：`all_flow_label_anal.csv`，同时生成`all_flow_head50.csv` (只有几十KB)，方便使用采样文件快速查看。
  ```shell
  python src/extract_feature/flow_stats.py
  ```
  
* **特征嵌入**：
  该模块在模型训练前对 `all_flow.csv` 中的类别型特征（categorical columns）进行嵌入处理。具体步骤如下：
  1. 在 `src/utils/config.cfg` 文件中，找到指定 `{plot_data_path}` 目录，例如 `progressed_data/CIC-AndMal2017`。  
  2. 在该目录下，找到位于相同子目录中的 `xxx_flow.csv` 与 `xxx_session.csv` 文件，并将它们分别合并生成 `all_flow.csv` 与 `all_session.csv`。如果已经存在这两个文件，那么这个步骤会略过。
  3. 将 all_session.csv中的会话划分为 train、test、validate 三个部分
  4. 生成 all_split_session.csv，在第二列插入 split 标签列
  5. 只有属于训练集的会话及其对应的流才用于构建域名-应用共现矩阵
  6. 对 `all_flow.csv` 中的 `dns.query` 与 `ssl.domain_name` 字段执行特征嵌入操作。
  7. 输出嵌入后的文件 `all_embedded_flow.csv`，其末尾新增十列：`ssl.server_name0_freq`、`ssl.server_name1_freq`、`ssl.server_name2_freq`、`ssl.server_name3_freq`、`ssl.server_name4_freq` 和 `dns.query0_freq`、`dns.query1_freq`、`dns.query2_freq`、`dns.query3_freq`、`dns.query4_freq`，分别表示对应字段的嵌入结果。
  ```bash
  python src/embed_feature/__main__.py 
  ```
  然后，评估所有的特征列和`is_malicious`目标列的相关性：该脚本的主要用途包括评估所有特征列与恶意流量标签 (is_malicious) 的相关性与泄露风险，同时分析不同数据划分方式是否会导致“泄露式高性能”
  
  ① flow 级别划分分析：使用随机分层划分训练/验证/测试集
  ```shell
  python src/embed_feature/analyze_all_flow_dataset_leakage.py --split_mode flow
  ```
  ② session 级别划分分析
  ```
  python src/embed_feature/analyze_all_flow_dataset_leakage.py --split_mode session
  ```


## 5. 针对会话，构建流关系图

脚本：`src/build_session_graph/__main__.py`

* 从`all_embedded_flow.csv`和`all_split_session.csv`，构建流关系图（节点特征 + 边），并导出为 DGL 图数据集，包括`all_session_graph.bin`二进制文件和`all_session_graph_info.pkl`图标签文件。
* 调用示例： `__main__.py` 是 构建 Session-Level DGL 图（Session Relation Graph） 的入口程序。它负责从多源 Zeek/FlowMeter 特征文件中读取流数据、解析会话结构、构建节点特征，并最终生成可供 GNN 模型训练的图文件。其中调用的 FlowNodeBuilder 会为每个 flow 构造多视图特征：

   * Flow/SSL/DNS/X509 四大类日志 → numeric / categorical / textual 三类视图
   * Flow试图 = conn+flowmeter 日志，其数值特征 → 数值向量
   * packet_len_seq / packet_iat_seq → padded 数据包级的数值特征的时间序列
   * domain_probs → 域名-APP的共现频率 embedding
   * categorical 特征 → top-K vocabulary + OOV=0 → id 编码后的 long tensor
   * textual 特征（保持 raw text 被 BERT tokenizer 转成了定长 token embedding 序列）
   
   五元组流flow的多视图特征配置文件具体参考[src/utils/zeek_columns.py](src/utils/zeek_columns.py)。哪些视图的五元组流flow特征会最后输出到dgl binary graph文件里面，可以通过配置文件[src/utils/config.cfg](src/utils/config.cfg)里面的如下条目控制。
   ```conf
   enabled_flow_node_views = {"flow_numeric_features": true, "flow_categorical_features": false, "flow_textual_features": false, "packet_len_seq": true, "packet_iat_seq": true, "domain_probs": true, "ssl_numeric_features": false, "ssl_categorical_features": false, "ssl_textual_features": true, "x509_numeric_features": false, "x509_categorical_features": false, "x509_textual_features": true, "dns_numeric_features": true, "dns_categorical_features": false, "dns_textual_features": true}
   ```
   对明文流量部分的文本特征，采用什么transformer模型的tokenizer转成token sequence，可以修改如下的配置项。
   ```conf
   # 或 "bert_uncased_L-2_H-128_A-2"，“bert_uncased_L-4_H-256_A-4”，或者"bert-base-uncased"
   # → bert_uncased_L-2_H-128_A-2 是一个 极小的微型 BERT（MiniBERT）
   # → bert_uncased_L-4_H-256_A-4 比 MiniBERT 强很多，但远比 BERT-base 轻。貌似在Nvidia A800的服务器上报错pytorch版本低。
   # → bert-base-uncased 是原版标准 BERT。
   text_encoder_name = bert_uncased_L-2_H-128_A-2
   # max_text_length用来限定XXXX_textual_features的合并文本的最大长度，超长的部分会阶段。
   # 仅 DNS textual	16–32	query/answer 非常短
   # DNS + SSL textual	32–64	SSL subject/issuer 有时较长
   # DNS + SSL + X509 textual（你的情况）	64（最均衡）	适中，能保留足够信息，又不会拖慢 BERT
   # 你想完整保留 certificate textual	128	更长但速度下降明显
   max_text_length = 64
   ```

```bash
python src/build_session_graph/__main__.py --sampling-ratio 1.0 --split-mode random --split-ratio 0.8,0.1,0.1
```
示例输出：
```bash
[2025-12-11 12:13:16,045 __main__.py,24] [INFO] 配置了内核线程数 = 20
[2025-12-11 12:13:16,051 __main__.py,27] [INFO] 配置了 session label string-to-id mapping: {'benign': 0, 'adware': 1, 'ransomware': 2, 'scareware': 3, 'smsmalware': 4}
[2025-12-11 12:13:16,052 __main__.py,30] [INFO] 开始处理 Flow 数据文件 ...
......
[2025-12-11 13:54:49,093 session_graph_builder.py,926] [INFO] ===最终数据集划分===
[2025-12-11 13:54:49,094 session_graph_builder.py,928] [INFO] 验证集: 12060 图 (8.8%)
[2025-12-11 13:54:49,094 session_graph_builder.py,929] [INFO] 测试集: 29139 图 (21.3%)
[2025-12-11 13:54:49,097 session_graph_builder.py,959] [INFO] Saving graphs to ./processed_data/CIC-AndMal2017-session-srcIP_dstIP-test\all_session_graph.bin ...
[2025-12-11 14:06:30,196 session_graph_builder.py,989] [INFO] Graph construction completed successfully!
[2025-12-11 14:06:30,205 __main__.py,70] [INFO] [SUCCESS] 成功构建 unified session graph! 输出文件: ./processed_data/CIC-AndMal2017-session-srcIP_dstIP-test\all_session_graph.bin
```

## 6. 流关系图可视化和统计分布

### 6.1 可视化DGL格式的会话图数据

支持批量处理.bin文件中的图数据，生成优化的burst流图可视化，并提供详细的统计信息。重要：程序会自动从配置文件读取数据集路径，并在此路径下查找 `all_session_graph.bin`文件。程序通过 `ConfigManager.read_plot_data_path_config()`读取数据路径，通常配置为：
```
processed_data/
└── {dataset_name}/
    ├── all_session_graph.bin      # DGL图数据（程序读取的主要文件）
    └── all_session_graph_info.pkl # 标签信息文件（自动生成）
```
主要特性：
- ✅ 批量处理DGL .bin文件中的图数据
- ✅ 自动识别和可视化burst聚类
- ✅ 支持节点和边的颜色编码
- ✅ 生成详细的统计报告和分布图表
- ✅ 支持按图ID范围选择性处理
- ✅ 自动处理中文字体显示

```bash
python src/draw_session_graph/draw_session_graph.py [选项]
```
主要参数
| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--out_dir` | 无 | `figs` | 输出目录路径 |
| `--min_idx` | 无 | `0` | 起始图索引（包含） |
| `--max_idx` | 无 | `20` | 结束图索引（包含） |
| `--burst_num_see` | 无 | `100` | 可视化的burst数量上限 |
| `--graph_file_name` | 无 | `all_session_graph` | 会话图数据文件名前缀（不含扩展名），用于指定要加载的 `.bin` / `_info.pkl` 图文件，例如 `all_session_graph__port_53_67_68_123__svc_dns_ntp` |
| `--only_draw_non_benign` | 无 | 关闭 | 仅绘制非 benign 会话图，用于快速查看恶意样本 |
| `--skip_labels` | 无 | 空 | 逗号分隔的标签列表，用于在**可视化阶段**跳过指定标签的会话图（不影响数据与统计），如 `portscan,bot` |


典型用法：
```bash
# 处理前20个图（索引0-19）
python src/draw_session_graph/draw_session_graph.py --only_draw_non_benign --skip_labels=portscan,bot --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp

# 指定输出目录和处理范围
python src/draw_session_graph/draw_session_graph.py --only_draw_non_benign --skip_labels=portscan,bot --min_idx 50 --max_idx 100 --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp
```

### 6.2 可视化会话图的节点和边数量分布

支持多种绘图风格和数据类型组合。重要：程序会自动从配置文件读取数据集路径，并在此路径下查找 `all_session_graph.bin`文件。
```bash
python src/draw_session_graph/draw_session_graph_size_distr.py [选项]
```
参数说明：
| 参数 | 缩写 | 可选值 | 默认值 | 说明 |
|------|------|--------|--------|------|
| `--type` | `-t` | `nodes`, `edges`, `both`, `comparison` | `both` | 要分析的数据类型 |
| `--plot-style` | `-p` | `loglog_dot`, `normal_bar`, `both` | `loglog_dot` | 绘图样式 |
| `--no-cache` | 无 | 无 | False | 强制重新计算，忽略缓存 |
| `--maxshow` | 无 | 整数 | 100 | 柱状图最大显示值 |
| `--report` | 无 | 无 | False | 打印详细统计报告 |
| `--graph_file_name` | 无 | 无 | `all_session_graph` | 会话图数据文件名前缀（不含扩展名），用于指定要加载的 `.bin` / `_info.pkl` 图文件，例如 `all_session_graph__port_53_67_68_123__svc_dns_ntp` |


多种绘图样式组合

```bash
# 同时生成节点和边的两种样式图（推荐！）
python src/draw_session_graph/draw_session_graph_size_distr.py --no-cache --type both --plot-style both --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp

# 只生成节点的两种样式图
python src/draw_session_graph/draw_session_graph_size_distr.py --no-cache --type nodes --plot-style both --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp

# 只生成边的两种样式图
python src/draw_session_graph/draw_session_graph_size_distr.py --no-cache --type edges --plot-style both --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp
```

### 6.3 可视化图数据集中标签分布情况

该工具能够从DGL图数据文件中提取标签信息，生成详细的统计报告和可视化图表，帮助用户了解数据集的类别分布情况。
1. 数据加载与处理：自动加载DGL图数据文件及其关联的标签信息
2. 标签分布统计：计算训练集、验证集和测试集的标签分布
3. 多种可视化方式：支持分离图、组合图和百分比图等多种可视化形式
4. 缓存机制：支持将统计结果缓存到JSON文件，提高后续分析效率
```shell
python src/draw_session_graph/draw_label_distr.py [选型]
```
命令行参数：

| 参数 | 缩写 | 可选值 | 默认值 | 描述 |
|------|------|--------|--------|------|
| `--type` | `-t` | `separate`, `combined`, `percentage`, `all` | `separate` | 图表类型：分离图/组合图/百分比图/全部 |
| `--split` | `-s` | `train`, `valid`, `test`, `all` | `all` | 要分析的数据集分割 |
| `--no-cache` | 无 | 无 | False | 忽略缓存，重新计算分布 |
| `--report` | 无 | 无 | False | 打印详细的标签分布报告 |
| `--graph_file_name` | 无 | 无 | `all_session_graph` | 会话图数据文件名前缀（不含扩展名），用于指定要加载的 `.bin` / `_info.pkl` 图文件，例如 `all_session_graph__port_53_67_68_123__svc_dns_ntp` |

使用示例：

1. **生成组合分布图（三个数据集对比）**
   ```bash
   python src/draw_session_graph/draw_label_distr.py --no-cache --type combined --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp
   ```

2. **生成所有数据集分割的分离分布图**
   ```bash
   python src/draw_session_graph/draw_label_distr.py --no-cache --type separate --split all --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp
   ```

3. **仅分析训练集的百分比分布**
   ```bash
   python src/draw_session_graph/draw_label_distr.py --no-cache --type percentage --split train --graph_file_name=all_session_graph__port_53_67_68_123__svc_arp_dhcp_dns_llmnr_mdns_nbns_ntp
   ```

## 7. 模型训练与测试

BERT+GAT 的混合神经网络对显存要求较高，特别是当BERT配置了较大的序列长度。代码测试可以用项目组的Nvidia 3090服务器。全面的模型性能评估则需要使用易安联公司的Nvidia A800 GPU服务器。访问步骤参考：[易安联公司的NVIDIA A800 GPU服务器的访问指南](./Enlink-GPU-guide.md)。


### 7.1 五元组流级别的多视图分类模型的训练

下面是基于CIC-IDS-2017数据集的实验，因为该数据集有五元组流级别的网络行为标签。

首先，编辑[src/utils/config.cfg](src/utils/config.cfg)文件。修改SESSION相关配置如下：

```conf
[SESSION]
path_to_dataset = ./dataset/CIC-IDS-2017
plot_data_path = ./processed_data/CIC-IDS-2017

# CIC-IDS-2017
# session_tuple_mode = srcIP_dstIP # 在 CIC-IDS-2017数据集上，如果用二元组做会话ID，容易造成预处理数据集的类别分布不平衡
session_tuple_mode = srcIP_srcPort_dstIP_dstPort_proto

# CIC-IDS-2017
# CIC-IDS-2017数据集的标签映射
# https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html
# 数据集里面有三种web attack，包括Web Attack - Brute Force, Web Attack - XSS, Web Attack - Sql Injection。统统标记为web attack。
session_label_id_map = BENIGN:0, DoS Hulk:1, DoS Slowhttptest:2, DoS slowloris:3, DoS GoldenEye:4, PortScan:5, DDoS:6, FTP-Patator:7, SSH-Patator:8, Bot:9, Web Attack:10, Infiltration: 11, Heartbleed:12
```

然后，查验四个流级别模型的配置文件。
* [src\models\flow_bert_multiview\config\flow_bert_multiview_config.yaml](src\models\flow_bert_multiview\config\flow_bert_multiview_config.yaml)
* [src\models\flow_bert_multiview_ssl\config\flow_bert_multiview_ssl_config.yaml](src\models\flow_bert_multiview_ssl\config\flow_bert_multiview_ssl_config.yaml) 
* [src\models\flow_bert_multiview_ssl_mlm\config\flow_bert_multiview_ssl_mlm_config.yaml](src\models\flow_bert_multiview_ssl_mlm\config\flow_bert_multiview_ssl_mlm_config.yaml)
* [src\models\flow_bert_multiview_ssl_seq2stat\config\flow_bert_multiview_ssl_seq2stat_config.yaml](src\models\flow_bert_multiview_ssl_seq2stat\config\flow_bert_multiview_ssl_seq2stat_config.yaml)

确保里面的`data.flow_data_path`和`data.session_split.session_split_path`的配置如下。并且确认这两个csv文件存在。没有的话，根据[processed_data/README.md](processed_data/README.md)指引，下载已经预处理好的csv文件到`processed_data`目录下面。
```yaml
# 数据配置
data:
  # 流数据文件路径
  flow_data_path: "./processed_data/CIC-IDS-2017/all_embedded_flow.csv"
  ......
  session_split:
    # 会话划分文件路径
    session_split_path: "./processed_data/CIC-IDS-2017/all_split_session.csv"
```
同时，`data.batch_size`变量要根据GPU显卡情况配置。如果从32增加到256，甚至1024就可以加速训练速度。比如，假设bert_uncased_L-2_H-128_A-2模型，那么在Nvidia A800显卡上，可以配置batch_size到1024；但是在2080ti和3090显卡上，只能设置到32。如果假设bert-base-uncased模型，那么相应的batch_size要除以2，甚至除以4。

第三，根据[models_hub/README.md](models_hub/README.md)指引，下载预训练BERT模型到`models_hub`目录。

最后，逐一运行四个模型的训练代码。在Nvidia A800 GPU单卡GPU上面，每个模型要训练大约6个小时。
```shell
python src/models/flow_bert_multiview/train.py
```
训练完成以后，能看到如下的命令行提示。
```
[2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1282] - precision recall f1-score support 正常 0.9933 0.9428 0.9674 93301 恶意 0.9433 0.9934 0.9677 89412 accuracy 0.9676 182713 macro avg 0.9683 0.9681 0.9676 182713 weighted avg 0.9689 0.9676 0.9676 182713 [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1289] - 🎯 混淆矩阵: [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1290] - [[87967 5334] [ 592 88820]] [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1298] - 📈 高级指标: [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1299] - ROC-AUC: 0.9926 [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1300] - Average Precision: 0.9906 [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1305] - 📊 样本统计: [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1306] - 总样本数: 182713 [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1307] - 正样本数: 89412.0 [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1308] - 负样本数: 93301.0 [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1309] - 正样本比例: 48.94% [2025-11-26 03:55:57] [INFO][flow_bert_multiview.py:1311] - 
============================================================ 
Testing DataLoader 0: 100%|██████████| 179/179 [09:22<00:00, 0.32it/s] 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ 
┃ Test metric               ┃    DataLoader 0           ┃ 
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ 
│ test_accuracy             │    0.9675666093826294     │ 
│ test_classification_loss  │    0.15476346015930176    │ 
│ test_f1                   │    0.9676861763000488     │ 
│ test_precision            │    0.9433475136756897     │ 
│ test_recall               │    0.9933767318725586     │ 
│ test_total_loss           │    0.15476356446743011    │ 
└───────────────────────────┴───────────────────────────┘
```
继续运行这三个有自监督学习模型的训练代码。
```shell
python src/models/flow_bert_multiview_ssl/train.py
python src/models/flow_bert_multiview_ssl_mlm/train.py
python src/models/flow_bert_multiview_ssl_seq2stat/train.py
```

四个流级别模型的完整横向对比（无 SSL → AE → MLM → Seq2Stat）。系统级性能评价：

| 模型代号   | 模型名称                                       | 自监督任务           | 说明                  |
| ------ | ------------------------------------------ | --------------- | ------------------- |
| **M0** | flow_bert_multiview（无 SSL）                 | ❌ 无             | 仅 supervised        |
| **M1** | flow_bert_multiview_ssl（Tabular AE）        | ✔ AE            | 稳定、容易训练             |
| **M2** | flow_bert_multiview_ssl_mlm（MLM）           | ✔ MLM + AE      | 加强序列 token 语义       |
| **M3** | flow_bert_multiview_ssl_seq2stat（Seq2Stat） | ✔ Seq2Stat + AE | 强 flow-level 语义、难训练 |

四模型关键指标对比表（最清晰的一张表）

| 指标            | **M0 无 SSL** | **M1 AE**      | **M2 AE+MLM** | **M3 Seq2Stat** |
| ------------- | ------------ | -------------- | ------------- | --------------- |
| **Accuracy**  | 0.9676       | **0.9712**     | 0.9698        | 0.9691          |
| **Precision** | 0.9433       | **0.9758**     | 0.9750        | 0.9422          |
| **Recall**    | 0.9934       | 0.9651         | 0.9630        | **0.9980（最高）**  |
| **F1 Score**  | 0.9677       | **0.9704（最高）** | 0.9689        | 0.9693          |
| **ROC-AUC**   | 0.9926       | **0.9971（最高）** | 0.9968        | 0.9914          |
| **PR-AUC**    | 0.9906       | **0.9966（最高）** | 0.9957        | 0.9887          |
| **误报 FP**     | 5334         | **2142（最少）**   | 2211          | 5472            |
| **漏报 FN**     | 592          | 3117           | 3312          | **175（最少）**     |

模型风格定位（一句话概括四个模型的性格）

| 模型               | 风格              | 核心特征                                        |
| ---------------- | --------------- | ------------------------------------------- |
| **M0 无 SSL**     | 默认混合模型 baseline | recall 很高、精度较低、整体稳定但不突出                     |
| **M1 AE（最佳 F1）** | 工业级稳健模型         | precision 最强、AUC 最高、误报最低、泛化最好               |
| **M2 MLM**       | “序列理解更强，但无提升”   | token-level 有增强，但 flow-level 分类并未受益（略弱于 M1） |
| **M3 Seq2Stat**  | 安全场景最优模型        | recall 接近完美（0.998），几乎不漏报恶意流量                |

最终选择建议（真正落地）

| 应用场景                      | 最佳模型                    |
| ------------------------- | ----------------------- |
| **安全检测（DDoS、APT、恶意流量检测）** | **M3 Seq2Stat（安全优先）**   |
| **大规模线上部署（误报少、稳）**        | **M1 AE（工业级最稳模型）**      |
| **学术方向（可发表、性能最好）**        | **M1 AE 或 M3 Seq2Stat** |
| **探索序列语义（NLP风格）**         | M2（但不推荐用于部署）            |

### 7.2 Session级的 TransGraphNet 图神经网络模型（肖卿俊开发中）

初步可用。但是还在开发过程中。还缺少一些功能，比如flow categorical特征的嵌入。具体查看to-do list：

```shell
python .\src\models\session_gnn_flow_bert_multiview_ssl\train.py 
```


## 8. 基于Wandb的模型性能评估和超参调节（易安联秦益飞开发，待查验）

```shell
python src/hyper_optimus/main.py
```

## 9. Baseline对比模型的运行


- **Appscanner** 从每个流中提取 54 个统计特征（如数据包数量、平均包大小、到达间隔时间等），并使用随机森林分类器来区分不同的移动应用程序。这个模型只用流统计特征，可以用[src/models/flow_autoencoder](src/models/flow_autoencoder)代表其性能；也可以用[src/models/flow_bert_multiview/](src/models/flow_bert_multiview/)，disable掉其他特征view，只保留流统计特征，代表其性能。

- **FS-Net** 是面向加密流量分类的端到端模型，采用多层双向门控循环单元（GRU）构建编码器 - 解码器结构，结合重构机制从原始流量序列（如数据包长度序列）中自动学习代表性特征，再通过全连接层压缩特征后，由 softmax 分类器实现应用级加密流量分类。可以用[src/models/flow_bert_multiview/](src/models/flow_bert_multiview/)，disable掉SSL、X509、DNS特征views，只保留流统计特征和数据包序列特征，代表其性能。

- **FG-Net** 提出了一种基于图神经网络（GNN）的框架，用于同时编码数据包级和流级信息。它构建了一个**流关系图（Flow Relationship Graph, FRG）**，其中每个节点表示一个网络流，节点属性包括数据包大小序列和到达间隔时间序列；边表示两种流之间的关系：**并发关系（concurrency）**（同一突发中的流）和**触发关系（trigger）**（连续突发之间的流）。流节点编码器从数据包级特征中学习潜在表示，并通过图注意力层进行传播与聚合，最终获得全局会话表示用于应用分类。

- **GraphDApp** 是面向区块链去中心化应用（DApp）加密流量识别的图神经网络（GNN）模型，旨在解决DApp因共享通信接口与加密设置导致流量判别性低、传统方法依赖人工特征的痛点。其核心是将DApp流量流构建为**流量交互图（TIG）**：以带符号长度（上行包为负、下行包为正）的数据包为节点，通过intra-burst边（连接同一方向连续数据包）与inter-burst边（连接相邻方向数据包的首尾节点）刻画交互关系；再利用多层感知机（MLP）与全连接层组成的GNN分类器，从TIG中自动提取特征完成图分类。该模型在含1300个DApp（16.9万+流）的真实数据集上表现优异，闭/开世界场景分类精度均优于主流方法，且可扩展至传统移动应用加密流量分类。

---

总结：**Appscanner** 仅依赖流级统计特征对单个五元组流进行分类，**FS-Net** 仅关注数据包级序列表示，二者均只能捕获流量行为的单一视角；而 **GraphDApp** 虽创新地将 **DApp** 加密流量流构建为流量交互图（TIG）（以带符号长度的数据包为节点，但仍局限于以图结构表征流量交互，未融合 TLS 握手、**DNS** 域名等关键场景特征，特征视角仍有拓展空间。
相比之下，我们提出的 **TransGraphNet** 不仅延续了多视角特征融合的思路，还进一步突破了上述模型的局限：一方面，通过联合表征每个五元组流的统计特征、定向数据包长度及到达间隔时间序列、TLS 握手特征和服务器域名信息，构建了比 **GraphDApp** 的 **TIG**、**FG-Net** 的流关系图更丰富的特征空间，实现了数据包级、流级与会话级表示的深度连接；另一方面，不同于 **FG-Net** 的流关系图融合逻辑与 **GraphDApp** 的 **GNN** 图分类机制，**TransGraphNet** 引入基于 **Transformer** 的数据包时间序列建模，结合 TLS 握手与 DNS 特征带来的分类精度提升，最终在应用指纹识别的区分能力与性能表现上，均优于 **FG-Net** 与 **GraphDApp**。

## 10. TODO List

以下功能尚未完成，后续需要补充：

- [ ] 在`src/models/flow_xxxx`和`src/models/session_xxxx`模型中，添加对http.log、ftp.log、mqtt.log的字段支持。可以开始的时候就，简单用bert来表征这些应用层日志的文本特征。
- [ ] 更准确的关联http.log文件：该文件里面一条记录 = 一个 HTTP 事务（request–response）。如果同一条 TCP 连接上：HTTP/1.1 keep-alive、HTTP pipelining、多个 GET / POST，那么都会生成多条 http.log。每条 http 记录都有：uid = conn.uid。这是 1:N 的关系。目前的`src/extract_feature/__main__.py`只能关联到其中一个http事务。因为它强行把http.log转成了uid -> http log记录的一对一映射的字典，而且保留的是http.log文件中最后出现的那条 http 事务（同 uid）。

- [ ] 除了数据包ip packet序列，新增 messsage (single packet + multi-packet bulk)的序列？可以参考论文[Piet, Anderson, McGrew - 2019 - An In-Depth Study of Open-Source Command and Control Frameworks.pdf](./docs/TCP-IP-FiveTupleFlow/Piet,%20Anderson,%20McGrew%20-%202019%20-%20An%20In-Depth%20Study%20of%20Open-Source%20Command%20and%20Control%20Frameworks.pdf)。注意利用tcp seqno处理数据包乱序和重传的问题。然后，也可以再做sequential pattern mining进一步减少序列长度？
  ![](./figs/ip-packet-message-exchange-sequence.png)
- [ ] Zeek Flowmeter插件的改进，关于如何正确实现Bulk的信息提取。具体参考[zeek-flowmeter插件的问题描述.md](zeek-flowmeter插件的问题描述.md)。

- [ ] 添加对更多dataset的支持，比如 [CIC-IDS-2018](./dataset/CIC-IDS-2018/README.md)、[CIC-IoMT-2024](./dataset/CIC-IoMT-2024/README.md)、[ISCX-VPN-nonVPN-2016](./dataset/ISCX-VPN-nonVPN-2016/README.md)，需要数据集的百度网盘链接，以及相应的打标签处理代码。另外，史大成反馈说：您的[CIC-IDS-2017](./dataset/CIC-IDS-2017/README.md)打标代码应该是有bug的。我记得您之前说改了哪里，全能打上标了。要检查[label_log.py](src/extract_feature/label_log.py)代码。但实际上CSV里就是有些根本不存在，按理讲不可能都打上标的。您的[CIC-IDS-2017](./dataset/CIC-IDS-2017/README.md)打标后的量和我的量不一样，比我的多不少。问题回答：[CIC-IDS-2017](./dataset/CIC-IDS-2017/README.md)可以先用史大成同学的打标程序[dataset/CIC-IDS-2017/label_conn_log.py](dataset/CIC-IDS-2017/label_conn_log.py)。

- [ ] 在分析流关系图的[flow_node_encoder.py](src/models/session_gnn_flow_bert_multiview_ssl/models/flow_node_encoder.py)的代码中，categorical特征要融合到节点里面。
- [ ] 在构建流关系图的[session_graph_build.py](src/build_session_graph/session_graph_builder.py)里面，如果流关系图包含的flow节点有混合标签，那么称其为“mixed” graph。这种标签混杂的图被丢弃了，要找到方法处理这种mixed图。
- [ ] 同样会话下的五元组流关系图构建代码，已经开发完成，放在[src/build_session_graph](src/build_session_graph/)的目录下面；但是还需要构建主机关系图，放到[src/build_host_graph](src/build_host_graph)的目录下面。

未来可以考虑进一步研究的内容：
- [ ] 基于NvFlare开发基于联邦学习范式的网络入侵检测。
- [ ] TLS握手字段特征向量，需要更深入的表征学习，比如用日志分析方法处理X509证书的文本字段。
- [ ] DNS握手字段特征向量，需要更深入的表征学习，比如考虑域名的层次化结构，或者采用复杂的异构图表示学习。
- [ ] [session_gnn_flow_bert_multiview_ssl](src/models/session_gnn_flow_bert_multiview_ssl/README.md) 基于对比学习实现图自监督学习，以及噪音标签学习。可以参考的仓库：[awesome-graph-self-supervised-learning](https://gitee.com/seu-csqjxiao/awesome-graph-self-supervised-learning)。
- [ ] 网络流flow的tabular feature向量可以用变分自编码器来做生成扰动。可以参考的仓库：[benchmark_VAE](https://gitee.com/seu-csqjxiao/benchmark_VAE)。
- [ ] 分析Zeek提取出来的`http.log`，增强对恶意软件的Web类型攻击的识别能力。
- [ ] 基于深度度量学习的流量相似度评估。

已经完成的功能：
- [ ] 针对Web攻击检测的需求，还需要额外关联http.log和mqtt.log日志。
- [ ] （已完成）分析流关系图的[flow_node_encoder.py](src/models/session_gnn_flow_bert_multiview_ssl/models/flow_node_encoder.py)代码最好不是`packet_len_key`逻辑与`packet_iat_key`的关系，而是逻辑或的关系。被disable的那个key，输入数据允许不存在，可以自动补零代替。
  ```python
    seq_enabled_by_cfg = (
        self.enabled_views.get(self.packet_len_key, False)
        and self.enabled_views.get(self.packet_iat_key, False)
    )
  ```
- [ ] （貌似没有必要，留着挺好，提醒有这些域）目前生成的`xxx-flow.csv`文件里面，还残留有`conn.uid`、`flowmeter.uid`、`dns.uid`和`ssl.uid`字段。需要修改`src/extract_feature/evaluate_data.py`文件，删掉它们。

抛弃的功能：

- [ ] 旧版本的流关系图的Graph Transformer模型训练脚本：`TransGraphNet_light.py`
  - 功能：载入 DGL 数据集，基于 BERT+GAT 的混合神经网络训练并评估，输出 Accuracy / Precision / Recall / F1。
  - 请手动下载`pytorch_model.bin`到`src/models/bert-mini`目录再运行模型训练代码TransGraphNet.py，下载地址如下：https://huggingface.co/prajjwal1/bert-mini/tree/main
  - 调用示例：默认从`{plot_data_path}/all_session_graph.bin`加载图数据。
  ```bash
  python src/models/TransGraphNet_light_multiview_pretrain.py --stage end2end
  ```
