通过百度网盘分享的文件：code-multiview-network-traffic-classification-model

```
链接: https://pan.baidu.com/s/1inexCSX3qCPra6ZIJKWnow?pwd=6nyq 
提取码: 6nyq
```

可以进入每个数据集的子目录，然后用下面的命令统计标签分布。在Linux和MacOS系统下面，则执行
```shell
python - <<'EOF'
import pandas as pd

df = pd.read_csv("all_embedded_flow.csv", usecols=["label"], low_memory=True)
print(df["label"].value_counts())
EOF
```
在Windows命令行下面，则执行
```shell
python -c "import pandas as pd; df=pd.read_csv('all_embedded_flow.csv', usecols=['label'], low_memory=True); print(df['label'].value_counts())"
```
比如，

* 对于CIC-IDS-2017数据集，类别分布如下：
    ```
    label
    BENIGN                        466497
    DoS Hulk                      162883
    PortScan                      158982
    DDoS                           86466
    DoS Slowhttptest               16045
    DoS GoldenEye                   7607
    FTP-Patator                     3986
    DoS slowloris                   3877
    SSH-Patator                     2979
    Bot                             2208
    Web Attack – Brute Force        1364
    Web Attack – XSS                 629
    Infiltration                      21
    Web Attack – Sql Injection        12
    Heartbleed                         1
    ```
    因此这提示我们在攻击类别的多分类任务里面，可以按照下面的表格，来不同攻击类别的OvR认为配置类别权重。
    ```
    # 统一分档 class_weights（推荐）
    BENIGN:              [1.0, 1.0]
    DoS Hulk:            [5.0, 1.0]
    PortScan:            [5.0, 1.0]
    DDoS:                [10.0, 1.0]
    DoS Slowhttptest:    [50.0, 1.0]
    DoS GoldenEye:       [100.0, 1.0]
    DoS slowloris:       [200.0, 1.0]
    FTP-Patator:         [200.0, 1.0]
    SSH-Patator:         [300.0, 1.0]
    Bot:                 [400.0, 1.0]
    Web Attack – *:      不做 OvR
    Infiltration:        不做 OvR
    Heartbleed:          不做 OvR
    ```
* 对于CIC-IDS-2018数据集，原始pcap就有1.02TB。经过zeek处理并做了日志关联之后，all_flow.csv文件有77GB。已经数据规模下降10几倍了。其类别分布如下：
    ```
    label
    benign                      28718757
    ftp-bruteforce                182067
    bot                           144961
    infilteration                 108005
    ssh-bruteforce                 80449
    ddos attack-hoic               19552
    dos attacks-hulk                4652
    dos attacks-slowhttptest          71
    ```
    这个数据集和CIC-IoMT-2024正好相反，其中的正常流量占比远大于攻击流量，更贴近实际的应用场景。另外，对应的完整标签列表如下，说明大量的流量样本没有打上标签：
    ```conf
    session_label_id_map = Benign:0, DoS attacks-Hulk:1, DoS attacks-SlowHTTPTest:2, DoS attacks-Slowloris:3, DoS attacks-GoldenEye:4, DDoS attacks-LOIC-HTTP:5, DDOS attack-LOIC-UDP:6, DDOS attack-HOIC:7, Brute Force-Web:8, Brute Force -XSS:9, FTP-BruteForce:10, SSH-Bruteforce:11, SQL Injection:12, Bot:13, Infilteration:14
    ```
* 对于CIC-IoMT-2024数据集，类别分布如下：
    ```
    label
    malicious_MQTT-DDoS-Connect_Flood           2730710
    malicious_TCP_IP-DDoS-TCP                   2322590
    malicious_TCP_IP-DoS-TCP                     656960
    malicious_Recon-Port_Scan                    211788
    malicious_TCP_IP-DDoS-UDP                    165723
    malicious_MQTT-DoS-Connect_Flood             145529
    malicious_TCP_IP-DoS-UDP                      58170
    malicious_MQTT-DDoS-Publish_Flood             51075
    malicious_Recon-OS_Scan                       30974
    benign_active                                 25010
    benign_idle                                   14585
    benign_unknown                                14125
    malicious_Recon-VulScan                        4265
    malicious_TCP_IP-DDoS-ICMP                     1812
    malicious_ARP_Spoofing                         1359
    malicious_TCP_IP-DoS-ICMP                      1138
    benign_interaction_Ecobee_Camera                824
    malicious_MQTT-Malformed_Data                   789
    benign_power                                    389
    malicious_MQTT-DoS-Publish_Flood                383
    benign_interaction_Owltron_Camera               354
    benign_interaction_M1T_Camera                   165
    benign_interaction_Singcall                      72
    malicious_Recon-Ping_Sweep                       44
    benign_interaction_Multifunctional_Pager         33
    benign_interaction_Blink_Camera                  32
    benign_interaction_SenseU                         5
    ```
    CIC-IoMT-2024 不是一个“平面多分类问题”，而是一个天然的分层安全推理问题。也可以合并类别，比如Benign可以合成一个大类别，解决is_malicious分类任务。这些 不是攻击类型，而是设备状态、交互场景、业务行为。
    ```
    benign_active
    benign_idle
    benign_unknown
    benign_interaction_XXX
    ```
    然后，采用分层 OvR（Hierarchical OvR）的策略，识别攻击类别。
    第一层：攻击“族群”级别（Attack Family）。
    ```
    Attack Family ∈ {
        TCP_IP,
        MQTT,
        Recon,
        ARP,
        ICMP
    }
    ```
    第二层：Family 内 OvR / 小多分类。
    ```
    TCP_IP:
    - DDoS-TCP
    - DoS-TCP
    - DDoS-UDP
    - DoS-UDP
    - DDoS-ICMP
    - DoS-ICMP
    ```    