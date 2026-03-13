# 易安联公司的NVIDIA A800 GPU服务器的访问指南

[TOC]

## 一、易安联GPU硬件、模型算力需求及项目相关说明

易安联配置的GPU卡为**NVIDIA A800 80GB PCIe**，属于英伟达面向数据中心与人工智能场景的专业级计算卡，核心参数与定位如下：
- **架构与芯片**：基于Ampere架构的GA100芯片，专为企业级高性能计算、大规模深度学习训练设计，是高端AI算力硬件的核心选择。
- **关键硬件参数**：A800 是中国区对标 A100 的降配版（受出口限制）。配备80GB HBM2e高速显存，采用PCIe 4.0接口，80GB HBM2e，受限带宽但仍是非常强大的训练卡。性能约为 A100 80GB 的 85% 左右。
- **显存验证**：从硬件监控截图中“73460MiB / 81920MiB”的显存数据可直接验证——因1GB=1024MiB，81920MiB÷1024=80GB，与官方规格完全一致。
- **市场价格**：该型号全新GPU卡的市场价格大致在**6.0–7.2 万人民币**区间，属于企业级高价值算力硬件。


### 1. 模型训练的GPU算力需求
结合项目前期工作与模型选择，不同类型模型对A800显卡的数量需求差异明确：
#### 前期小模型（BERT/BART/T5）：单卡即可满足
项目前期以流量数据预处理为主，Transformer架构仅需BERT、BART、T5等小模型（非大模型），此类模型参数量较小、显存占用低，**单张NVIDIA A800 80GB PCIe卡完全可支撑训练与推理**，无需额外增加显卡。

####  Llama大模型：需根据参数量与微调方式确定卡数
若后续涉及Llama系列大模型微调，显卡需求取决于模型参数量与微调策略（高效微调技术可大幅减少卡数），具体如下：
| 模型型号       | 参数量 | 全参数微调所需A800数量 | 高效微调（LoRA/QLoRA）所需A800数量 |
|----------------|--------|------------------------|------------------------------------|
| Llama 3-8B     | 80亿   | 4张                    | 3张                                |
| Llama 3-70B    | 700亿  | 8张                    | 2-3张                              |

核心逻辑：模型参数量越大、采用全参数微调时，对显存与算力的需求越高，所需显卡数量越多；而LoRA（低秩适应）、QLoRA（量化低秩适应）等高效微调技术，可通过“冻结部分参数、仅训练少量适配层”减少显存占用，从而降低显卡需求。

### 2. GPU服务器运行环境
用于运行基础大模型的服务器为**物理服务器**，未做虚拟化处理（资源直接分配，无虚拟化损耗），软件环境配置特点为：通过**Python虚拟环境**管理依赖包——可隔离不同模型的开发环境（如不同版本的PyTorch、TensorFlow），避免依赖冲突，保障模型训练与调试的稳定性。


### 3. 项目经费与开发补助
* **项目经费信息**：易安联在校内对应的项目经费支持如下：
   - 经费号：8509016031
   - 项目名称：零信任架构下用户管理与评价模型研究
* **开发补助**：参与项目核心工作（包括模型代码构建、训练调参等）的同学，后续将根据贡献发放开发补助，具体发放细则将结合项目进度与个人参与情况确定。

## 二、VPN访问易安联内网的方法


> EnBox 是易安联零信任战略的重要落地产品，通过 “隔离 + 动态控制” 的技术路径，为企业提供终端安全、数据防泄漏和网络访问控制的一体化解决方案。尽管用户提到的 “EnUESBOX” 可能存在拼写误差，但结合易安联产品线和行业实践，EnBox是其核心终端安全工具的准确名称。

### 步骤1：下载EnUESBOX，里面提供Windows桌面端的版本和MacOS系统的版本。

```
通过网盘分享的文件：EnUESBOX
链接: https://pan.baidu.com/s/11-egumFQoGzENxAm2BBNGw?pwd=fkfx 
提取码: fkfx
```
注意：MacOS上面，开下面的两个网络权限。请前往系统设置【通用】-【登录项与扩展】-【扩展】-【按类别】-【网络扩展】进行如下操作。在图形化界面上，务必要点击一下那些灰化的未授权按钮，才能在MacOS系统设置，找到那些请求权限的项目。
1. EnUESBaseTunnelExtension，即隧道连接：授权后，才能使用VPN相关功能
2. SimpieFirewallExtension，即网络管控：授权后，才能管控本机网络权限

![](./figs/macos-network-rights-for-ENSDP.png)

MacOS上面，开如下两个磁盘访问权限。请前往系统设置【通用】-【登录项与扩展】-【端点安全性扩展】以及【隐私与安全】-【完全磁盘访问权限】进行如下操作。如果不给权限，那么整个应用窗口都会被模态提示对话框给灰化。
![](./figs/macos-disk-rights-for-ENSDP.png)

### 步骤2：登录易安联公司对外网的VPN服务

用秦益飞博士的用户名和密码，登陆VPN。

过期了
```
服务器名称：ensbrain1.sdp.enlink.top
服务器域名：ensbrain1.sdp.enlink.top
端口号：4481
用户名：qinyfseu
密码：Enlink@121
```

点击【用户》企业配置》点击此处加入企业》】，可以添加目前可用的新的企业服务器。注意：接入节点选择广州或者北京，别选择上海。上海节点还有问题。
```
服务器名称：cloudwing-100135.enlinkcloud.net
服务器域名：cloudwing-100135.enlinkcloud.net
端口号：443
用户名：qinyf
密码：Enlink@123
```

看这个服务器一直连不上：`cloudwing-100135.enlinkcloud.net`。回复：有可能服务器挂了
```
服务器名称：ztna.ensase.top
服务器域名：ztna.ensase.top
端口号：65515
用户名：qinyf
密码: Enlink@123
```

### 步骤3：SSH连接易安联公司内网的GPU服务器

只要上面VPN登录成功。就可以ping通易安联公司内网的GPU服务器内网IP：192.168.100.57。

```shell
% ping 192.168.100.57
PING 192.168.100.57 (192.168.100.57): 56 data bytes
64 bytes from 192.168.100.57: icmp_seq=0 ttl=62 time=39.974 ms
64 bytes from 192.168.100.57: icmp_seq=1 ttl=62 time=22.730 ms
64 bytes from 192.168.100.57: icmp_seq=2 ttl=62 time=19.034 ms
64 bytes from 192.168.100.57: icmp_seq=3 ttl=62 time=18.515 ms
64 bytes from 192.168.100.57: icmp_seq=4 ttl=62 time=25.046 ms
64 bytes from 192.168.100.57: icmp_seq=5 ttl=62 time=20.699 ms
```

可以远程SSH连接易安联内网的GPU服务器。这是易安联公司运行基础大模型的GPU服务器，上面跑ubuntu系统。

```
内网IP地址：192.168.100.57
用户名：ubuntu
密码：Kuanke@tzh
```

```shell
$ ssh -p 22 ubuntu@192.168.100.57

The authenticity of host '192.168.100.57 (192.168.100.57)' can't be established.
ED25519 key fingerprint is SHA256:vKrQi6kzBbnkxrKzvSmyI+nSXbsF/vjSl+S0YWrF794.
This key is not known by any other names.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '192.168.100.57' (ED25519) to the list of known hosts.
ubuntu@192.168.100.57's password: 
Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 5.15.0-25-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro

Expanded Security Maintenance for Applications is not enabled.

364 updates can be applied immediately.
261 of these updates are standard security updates.
To see these additional updates run: apt list --upgradable

71 additional security updates can be applied with ESM Apps.
Learn more about enabling ESM Apps service at https://ubuntu.com/esm


The list of available updates is more than a week old.
To check for new updates run: sudo apt update
Last login: Fri Oct 10 22:40:02 2025 from 192.168.120.53
```

## 三、用 FileZilla 向 GPU 服务器上传文件（图形化便捷方案）

只要能通过 SSH 登录 GPU 服务器，**FileZilla** 是最优的图形化文件传输工具——操作直观、无需记命令。其底层是将 `scp` 命令封装为图形化功能，依托 SSH 协议实现加密传输，安全性与 `scp` 一致，且支持断点续传（传输中断后可恢复，适合大批量数据）。在**兼容性**方面，Windows、macOS、Linux 系统均支持，只需服务器开启 SSH 服务（能 SSH 登录即满足条件）。

### 步骤1：下载并安装 FileZilla
从 [FileZilla 官方网站](https://filezilla-project.org/download.php?type=client) 下载“FileZilla Client”（免费开源），按系统引导完成安装（默认选项即可）。

### 步骤2：配置服务器连接（仅首次需设置）
1. 打开 FileZilla，点击顶部菜单栏 **“文件”→“站点管理器”**（或直接按 `Ctrl+S`）。
2. 在弹出的窗口中，点击左下角 **“新站点”**，按以下信息配置：
   - **协议**：选择 `SFTP - SSH File Transfer Protocol`（基于 SSH，与 `scp` 同底层）。
   - **主机**：输入 GPU 服务器的 IP 地址（如 `192.168.100.57`，与 SSH 登录地址一致）。
   - **端口**：默认填 `22`（若服务器 SSH 端口非默认，需输入实际端口，如 `2222`）。
   - **登录类型**：选择 `正常`。
   - **用户**：输入 SSH 登录的用户名（如 `ubuntu`）。
   - **密码**：输入 SSH 登录的密码（或使用密钥登录，需提前配置密钥文件）。
3. 配置完成后，点击 **“连接”**，首次连接会提示“信任服务器指纹”，点击“确定”即可建立连接。


### 步骤3：批量上传文件到指定路径 `/data/qinyf`
连接成功后，FileZilla 界面分为左右两栏：
- **左栏**：本地电脑的文件目录，找到你需要上传的批量文件/文件夹（如 `本地文档/LLM数据`）。
- **右栏**：GPU 服务器的文件目录，通过路径导航栏定位到 **`/data/qinyf`**（这是你们组的统一数据路径，务必确认路径正确）。

#### 执行上传：

- 单个文件：直接在左栏选中文件，拖拽到右栏的 `/data/qinyf` 目录中。
- 批量文件/文件夹：按住 `Ctrl` 键多选文件，或直接选中整个文件夹，拖拽到右栏 `/data/qinyf` 目录即可（会自动递归上传文件夹内所有内容）。

#### 查看进度：
底部状态栏会显示上传速度、已传大小、剩余时间，大文件/批量文件传输时可最小化窗口，无需值守（支持断点续传，若中途断网，重新连接后右键点击“继续”即可恢复）。

### 两个关键注意事项（避免影响他人）

1. **上传时间：优先选择晚上，避免占用白天易安联出口带宽**  
   白天是公司网络使用高峰，大批量数据上传会占用易安联出口带宽，可能影响其他同事的网络使用；晚上（如 20:00 后）网络负载低，上传速度更稳定，也不会干扰他人工作。

2. **避免影响服务器上其他 LLM 的训练任务**  
   GPU 服务器上有其他团队的 LLM 训练任务，训练过程会占用大量 CPU、内存和网络资源。建议：
   - 上传前可通过 SSH 登录服务器，用 `top` 或 `nvidia-smi` 命令查看当前服务器负载（若显示 GPU/CPU 使用率过高，可暂缓上传）。
   - 晚上上传时，尽量避开其他团队的训练高峰期（可提前和相关同事确认训练时间），减少对训练任务的资源抢占。

### 常见问题解决

* 上传失败提示 “权限不足”：检查`/data/qinyf` 目录的写入权限，若没有权限，需联系服务器管理员执行 `sudo chmod 755 /data/qinyf`（或赋予你用户名的写入权限）。
* 上传速度慢：确认当前网络是否为晚上低峰期，若仍慢，可在 FileZilla 顶部菜单栏 “编辑”→“设置”→“传输” 中，将 “并发传输数” 调整为 2-3（避免单线程传输过慢，也不建议过多占用服务器资源）