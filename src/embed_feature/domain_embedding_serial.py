import numpy as np
import pandas as pd
from collections import defaultdict
import tqdm
import time
import sys
import os
import pickle

# 导入完整的日志功能
# 添加utils目录到Python路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)

# ========== 简化的工具函数 ==========

def get_system_memory_info():
    """获取系统内存信息"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total_mb': memory.total // (1024 * 1024),
            'available_mb': memory.available // (1024 * 1024),
            'used_percent': memory.percent,
            'free_mb': memory.free // (1024 * 1024)
        }
    except:
        return {'total_mb': 8192, 'available_mb': 4096, 'used_percent': 50, 'free_mb': 4096}

def save_embedding_data(embedding_data, file_path):
    """保存嵌入数据到文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(embedding_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        logger.error(f"保存嵌入数据失败: {e}")
        return False

def load_embedding_data(file_path):
    """从文件加载嵌入数据"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"加载嵌入数据失败: {e}")
        return None

def calculate_chunk_size(df, chunk_size=5000):
    """计算分块大小"""
    return max(1, (len(df) + chunk_size - 1) // chunk_size)

class DomainEmbeddingProcessor:
    """
    域名嵌入处理器，专注于域名嵌入功能 - 单线程串行版本
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # 核心属性初始化
        self.domain_app_freq = None
        self.num_apps = None
        self.app_classes = None
        self.app_index = None
        
        # 层级相关属性
        self.domain_hierarchy_enabled = False
        self.domain_split_levels = 5
        self.domain_hierarchy_stats = None
        self.domain_hierarchy_freq = None
        self.level_weights = None
        
        # 新增必需的属性
        self.domain_stats = None
        self.app_total_flows = None
    
    def _create_zero_list(self):
        """创建零列表（用于pickle序列化）"""
        return [0] * self.num_apps

    def _split_domain_by_levels(self, domain):
        """
        高性能域名层级拆分
        使用缓存和预分配内存优化性能
        """
        if not domain or domain == 'nan':
            return [None] * self.domain_split_levels
        
        # 预分配结果列表
        levels = [None] * self.domain_split_levels
        
        # 快速域名解析
        parts = domain.split('.')
        if len(parts) < 2:
            levels[0] = domain
            return levels
        
        # 从右向左构建层级（性能优化：避免重复字符串连接）
        current_domain = ""
        for i in range(min(len(parts), self.domain_split_levels)):
            if i == 0:
                current_domain = parts[-1]
            else:
                current_domain = parts[-(i+1)] + '.' + current_domain
            levels[i] = current_domain
        
        return levels
        
    def build_domain_app_cooccurrence(self, flow_df, session_df, session_label_id_map, train_flow_uids):
        """构建域名-应用共现矩阵 - 单线程版本"""
        # 原有初始化逻辑保持不变
        self.app_classes = list(session_label_id_map.keys())
        self.num_apps = len(self.app_classes)
        self.app_index = {cls: i for i, cls in enumerate(self.app_classes)}
        
        # if 'background' not in self.app_index or 'Background' not in self.app_index:
        #     # 方案1：将background添加到app_classes中（推荐）
        #     self.app_classes.append('background')
        #     self.num_apps = len(self.app_classes)
        #     self.app_index['background'] = len(self.app_classes) - 1  # 最后一个索引
            
        #     logger.info(f" [INFO] 添加background标签到应用类别，索引: {self.app_index['background']}")
        #     logger.info(f" [INFO] 更新后的应用类别数量: {self.num_apps}")
            
        # 初始化统计数据结构
        if self.domain_hierarchy_enabled:
            # 层级统计结构
            self.domain_hierarchy_stats = {}
            for i in range(self.domain_split_levels):
                self.domain_hierarchy_stats[f'level_{i}'] = defaultdict(lambda: [0] * self.num_apps)
        else:
            # 原有统计结构
            self.domain_stats = defaultdict(lambda: [0] * self.num_apps)
        
        self.app_total_flows = [0] * self.num_apps

        def match_configured_label(flow_label):
            """匹配配置的应用标签"""
            flow_label_str = str(flow_label).strip().lower()
            
            # 预处理配置标签：去除空格并转为小写
            configured_labels_lower = {label.strip().lower(): label for label in session_label_id_map.keys()}
            
            # 方案1：精确匹配（推荐）
            if flow_label_str in configured_labels_lower:
                return configured_labels_lower[flow_label_str]
            
            # 方案2：部分匹配（如果精确匹配失败）
            for configured_lower, original_label in configured_labels_lower.items():
                if configured_lower in flow_label_str:
                    return original_label
            
            # 如果都没有匹配到，返回 'background'
            return 'background'        

        if self.verbose:
            logger.info(f" 训练集包含 {len(train_flow_uids)} 个流")
            if self.domain_hierarchy_enabled:
                logger.info(f" 启用域名层级拆分，层级数: {self.domain_split_levels}")

        # 性能优化：预计算训练集UID集合
        train_flow_uids_set = set(train_flow_uids)
        
        # 统计域名与应用的共现频数 - 单线程处理
        # for _, flow in tqdm.tqdm(flow_df.iterrows(), total=len(flow_df), 
        #                        desc="构建共现矩阵", disable=not self.verbose):
        for flow in tqdm.tqdm(
                flow_df.itertuples(index=False),
                total=len(flow_df),
                desc="构建共现矩阵",
                disable=not self.verbose
        ):            
            # 只处理训练集的流
            flow_uid = flow.uid
            if flow_uid not in train_flow_uids_set:
                continue
            
            # 获取当前flow的应用类别
            # flow_label = flow.get("label", "background")
            flow_label = getattr(flow, 'label', 'background')
            matched_cls = match_configured_label(flow_label)
            
            app_idx = self.app_index[matched_cls]
            self.app_total_flows[app_idx] += 1

            # 提取DNS域名和SSL SNI域名
            domains = self._extract_domains_from_flow(flow)

            # 去重并统计
            for domain in set(domains):
                if not domain:
                    continue
                    
                if self.domain_hierarchy_enabled:
                    # 层级拆分统计
                    domain_levels = self._split_domain_by_levels(domain)
                    for level_idx, level_domain in enumerate(domain_levels):
                        if level_domain:
                            level_stats = self.domain_hierarchy_stats[f'level_{level_idx}']
                            level_stats[level_domain][app_idx] += 1
                else:
                    # 原有统计逻辑
                    self.domain_stats[domain][app_idx] += 1

        # 计算归一化的共现频率矩阵
        if self.domain_hierarchy_enabled:
            self._compute_hierarchy_frequency()
        else:
            self._compute_original_frequency()

        if self.verbose:
            self._print_statistics()
            self.run_comprehensive_validation()

        return self._get_final_frequency_matrix()

    def _extract_domains_from_flow(self, flow_record):
        """从flow记录中提取域名"""
        
        def _get(attr, default=''):
            return getattr(flow_record, attr, default)
    
        domains = []
        
        # 提取DNS查询域名
        # dns_query = flow_record.get('dns.query', '')
        dns_query = _get('dns_query', '')
        if dns_query and pd.notna(dns_query):
            try:
                if isinstance(dns_query, str) and dns_query.startswith('['):
                    # 尝试解析列表格式
                    import ast
                    dns_list = ast.literal_eval(dns_query)
                    if isinstance(dns_list, (list, tuple)):
                        domains.extend([str(d).strip() for d in dns_list if d])
                    else:
                        domains.append(str(dns_query).strip())
                else:
                    domains.append(str(dns_query).strip())
            except:
                domains.append(str(dns_query).strip())
        
        # 提取SSL域名
        # ssl_fields = ['ssl.server_name', 'ssl.domain_name', 'ssl.sni']
        ssl_fields = ['ssl_server_name', 'ssl_domain_name', 'ssl_sni']
        for ssl_field in ssl_fields:
            # ssl_value = flow_record.get(ssl_field, '')
            ssl_value = _get(ssl_field, '')
            if ssl_value and pd.notna(ssl_value):
                domains.append(str(ssl_value).strip())
                break
        
        return [d for d in domains if d and d != 'nan']

    def _embed_single_domain(self, domain):
        """单个域名嵌入"""
        if not domain or (self.domain_app_freq is None and self.domain_hierarchy_freq is None):
            return np.zeros(self.num_apps, dtype=np.float32)
        
        if self.domain_hierarchy_enabled:
            return self._embed_domain_with_hierarchy(domain)
        else:
            return self.domain_app_freq.get(domain, np.zeros(self.num_apps, dtype=np.float32))

    def _embed_dns_query(self, dns_query):
        """DNS查询嵌入"""
        if not dns_query or pd.isna(dns_query):
            return np.zeros(self.num_apps, dtype=np.float32)
        
        # DNS解析逻辑
        domains = []
        try:
            if isinstance(dns_query, str) and dns_query.startswith('['):
                import ast
                dns_list = ast.literal_eval(dns_query)
                if isinstance(dns_list, (list, tuple)):
                    domains.extend([str(d).strip() for d in dns_list if d])
                else:
                    domains.append(str(dns_query).strip())
            else:
                domains.append(str(dns_query).strip())
        except:
            domains.append(str(dns_query).strip())
        
        if not domains:
            return np.zeros(self.num_apps, dtype=np.float32)
        
        # 多个域名时的处理 - 单线程版本
        vec_list = []
        for domain in domains:
            vec = self._embed_single_domain(domain)
            vec_list.append(vec)
        return np.mean(vec_list, axis=0)

    def embed_domains_in_flow_data(self, flow_df):
        """
        对flow数据中的域名进行嵌入 - 单线程版本
        """
        if (self.domain_app_freq is None and self.domain_hierarchy_freq is None):
            raise ValueError("请先构建域名-应用共现矩阵")
        
        return self.embed_domains_in_flow_data_serial(flow_df)

    def _compute_hierarchy_frequency(self):
        """计算层级频率矩阵"""
        self.domain_hierarchy_freq = {}
        
        for level in range(self.domain_split_levels):
            level_stats = self.domain_hierarchy_stats[f'level_{level}']
            level_freq = {}
            
            # 向量化计算
            for domain, counts in level_stats.items():
                if domain is None:
                    continue
                    
                # 避免除零错误
                denominators = np.maximum(self.app_total_flows, 1e-8)
                freq = np.array(counts, dtype=np.float32) / denominators
                level_freq[domain] = freq
                
            self.domain_hierarchy_freq[f'level_{level}'] = level_freq

    def _compute_original_frequency(self):
        """原有频率计算逻辑"""
        self.domain_app_freq = {}
        for domain, counts in self.domain_stats.items():
            denominators = np.maximum(self.app_total_flows, 1e-8)
            freq = np.array(counts, dtype=np.float32) / denominators
            self.domain_app_freq[domain] = freq

    def _print_statistics(self):
        """输出统计信息"""
        if self.domain_hierarchy_enabled:
            total_domains = 0
            for level in range(self.domain_split_levels):
                level_count = len(self.domain_hierarchy_stats[f'level_{level}'])
                total_domains += level_count
                if self.verbose:
                    logger.info(f"  - 层级{level}域名数量: {level_count}")
            
            logger.info(f"构建完成域名层级-应用共现矩阵")
            logger.info(f"  - 总域名数量: {total_domains}")
        else:
            logger.info(f"构建完成域名-应用共现矩阵")
            logger.info(f"  - 包含域名数量: {len(self.domain_app_freq)}")
        
        logger.info(f"  - 应用类别数量: {self.num_apps}")
        logger.info(f"  - 应用类别顺序: {self.app_classes}")
        logger.info(f"  - 使用的训练集流数量: {sum(self.app_total_flows)}")

    def run_comprehensive_validation(self, test_domains=None):
        """运行全面验证"""
        logger.info("开始全面验证域名嵌入处理过程")
        
        # 域名分解验证
        if test_domains is None:
            test_domains = ["www.google.com", "example.com", "a.b.c.d.e.f.g", "", "nan"]
        
        logger.info("域名分解验证:")
        for domain in test_domains:
            result = self._split_domain_by_levels(domain)
            logger.info(f"  '{domain}' -> {result}")
        
        # 共现统计验证
        logger.info("共现统计验证:")
        if self.domain_hierarchy_enabled:
            total_domains = 0
            for level in range(self.domain_split_levels):
                level_count = len(self.domain_hierarchy_stats[f'level_{level}'])
                total_domains += level_count
                logger.info(f"  - 层级{level}: {level_count} 个域名")
            logger.info(f"  - 总域名数量: {total_domains}")
        else:
            domain_count = len(self.domain_app_freq) if self.domain_app_freq else 0
            logger.info(f"  - 非层级模式域名数量: {domain_count}")
        
        logger.info("全面验证完成")

    def _get_final_frequency_matrix(self):
        """获取最终频率矩阵"""
        if self.domain_hierarchy_enabled:
            return self.domain_hierarchy_freq, self.num_apps
        else:
            return self.domain_app_freq, self.num_apps

    def _embed_domain_with_hierarchy(self, domain):
        """层级域名嵌入"""
        if not domain:
            return np.zeros(self.num_apps, dtype=np.float32)
        
        # 拆分域名层级
        domain_levels = self._split_domain_by_levels(domain)
        
        # 为每个有效层级生成嵌入向量
        level_embeddings = []
        
        for level_idx, level_domain in enumerate(domain_levels):
            if level_domain:
                level_freq_dict = self.domain_hierarchy_freq[f'level_{level_idx}']
                level_vec = level_freq_dict.get(level_domain, np.zeros(self.num_apps, dtype=np.float32))
                level_embeddings.append(level_vec)
        
        # 合并所有层级的嵌入向量
        if level_embeddings:
            return np.mean(level_embeddings, axis=0)
        else:
            return np.zeros(self.num_apps, dtype=np.float32)

    def process(self, flow_df, session_df, session_label_id_map, train_flow_uids):
        """
        域名嵌入处理流程 - 单线程版本
        """
        return self.process_serial(flow_df, session_df, session_label_id_map, train_flow_uids)

    def process_serial(self, flow_df, session_df, session_label_id_map, train_flow_uids):
        """
        单线程串行处理版本
        """
        if self.verbose:
            logger.info(f"开始串行处理")
            logger.info(f"  - 数据总量: {len(flow_df)} 行")
            logger.info(f"  - 训练集流数量: {len(train_flow_uids)}")
        
        # 阶段1：构建共现矩阵
        stage1_start = time.time()
        if self.verbose:
            logger.info(f"阶段1: 构建共现矩阵...")
        
        self.build_domain_app_cooccurrence(flow_df, session_df, session_label_id_map, train_flow_uids)
        
        stage1_time = time.time() - stage1_start
        if self.verbose:
            logger.info(f"阶段1完成，耗时: {stage1_time:.2f}秒")
        
        # 阶段2：串行嵌入处理
        stage2_start = time.time()
        
        if self.verbose:
            logger.info(f"阶段2: 串行嵌入处理...")
        
        # 使用串行嵌入方法
        embedded_flow_df = self.embed_domains_in_flow_data_serial(flow_df)
        
        stage2_time = time.time() - stage2_start
        total_time = time.time() - stage1_start
        
        if self.verbose:
            logger.info(f"阶段2完成，耗时: {stage2_time:.2f}秒")
            logger.info(f"串行处理完成，总耗时: {total_time:.2f}秒")
            logger.info(f"  - 处理速度: {len(flow_df)/total_time:.1f} 行/秒")
        
        return embedded_flow_df

    def embed_domains_in_flow_data_serial(self, flow_df):
        """
        单线程串行域名嵌入（使用 tqdm 单行进度条）
        """
        if (self.domain_app_freq is None and self.domain_hierarchy_freq is None):
            raise ValueError("请先构建域名-应用共现矩阵")

        total = len(flow_df)
        start_time = time.time()

        if self.verbose:
            logger.info("开始串行嵌入处理")
            logger.info(f"  - 数据总量: {total} 行")

        # 创建结果 DataFrame
        embedded_flow_df = flow_df  # ❗ 不 copy:删掉了 flow_df.copy()

        # 预分配嵌入列
        for i in range(5):
            embedded_flow_df[f'ssl.server_name{i}_freq'] = None
            embedded_flow_df[f'dns.query{i}_freq'] = None

        # tqdm 进度条
        pbar = tqdm.tqdm(
            total=total,
            desc="Domain embedding",
            unit="row",
            disable=not self.verbose,
            mininterval=0.5,   # 至少 0.5s 刷新一次，避免过于频繁
            smoothing=0.1      # 速度平滑，数值更稳定
        )

        # for idx, (_, flow) in enumerate(flow_df.iterrows()):
        
        #     # SSL 域名嵌入
        #     ssl_domain = flow.get('ssl.server_name', '')
        #     if pd.isna(ssl_domain) or not ssl_domain:
        #         ssl_domain = flow.get('ssl.sni', '')
                
        #     ssl_embedding = self._embed_single_domain(
        #         str(ssl_domain).strip() if ssl_domain else ''
        #     )

        #     # DNS 查询嵌入
        #     dns_query = flow.get('dns.query', '')
        #     dns_embedding = self._embed_dns_query(dns_query)

        #     # 写回嵌入
        #     self._assign_embedding_to_row(
        #         embedded_flow_df, idx, ssl_embedding, dns_embedding
        #     )

        #     # 更新进度条
        #     pbar.update(1)

        for idx, flow in enumerate(flow_df.itertuples(index=False)):

            # ---------- SSL 域名嵌入 ----------
            ssl_domain = getattr(flow, 'ssl_server_name', '')
            if not ssl_domain or pd.isna(ssl_domain):
                ssl_domain = getattr(flow, 'ssl_sni', '')

            ssl_embedding = self._embed_single_domain(
                str(ssl_domain).strip() if ssl_domain else ''
            )

            # ---------- DNS 查询嵌入 ----------
            dns_query = getattr(flow, 'dns_query', '')
            dns_embedding = self._embed_dns_query(dns_query)

            # ---------- 写回嵌入 ----------
            self._assign_embedding_to_row(
                embedded_flow_df, idx, ssl_embedding, dns_embedding
            )

            # ---------- 更新进度 ----------
            pbar.update(1)
            
        pbar.close()

        total_time = time.time() - start_time
        if self.verbose:
            logger.info(f"串行嵌入完成，总耗时: {total_time:.2f} 秒")
            logger.info(f"  - 平均速度: {total / total_time:.1f} 行/秒")

        return embedded_flow_df

    # def embed_domains_in_flow_data_serial_old(self, flow_df):
    #     """
    #     单线程串行域名嵌入
    #     """
    #     if (self.domain_app_freq is None and self.domain_hierarchy_freq is None):
    #         raise ValueError("请先构建域名-应用共现矩阵")
        
    #     start_time = time.time()
        
    #     if self.verbose:
    #         logger.info(f"开始串行嵌入处理")
    #         logger.info(f"  - 数据总量: {len(flow_df)} 行")
        
    #     # 创建结果DataFrame的副本
    #     embedded_flow_df = flow_df.copy()
        
    #     # 预分配嵌入列
    #     for i in range(5):
    #         embedded_flow_df[f'ssl.server_name{i}_freq'] = None
    #         embedded_flow_df[f'dns.query{i}_freq'] = None
        
    #     # 串行处理每一行
    #     for idx, (_, flow) in enumerate(flow_df.iterrows()):
    #         # SSL域名嵌入
    #         ssl_domain = flow.get('ssl.server_name', '')
    #         if pd.isna(ssl_domain) or not ssl_domain:
    #             ssl_domain = flow.get('ssl.sni', '')
            
    #         ssl_embedding = self._embed_single_domain(str(ssl_domain).strip() if ssl_domain else '')
            
    #         # DNS查询嵌入
    #         dns_query = flow.get('dns.query', '')
    #         dns_embedding = self._embed_dns_query(dns_query)
            
    #         # 处理嵌入结果并赋值
    #         self._assign_embedding_to_row(embedded_flow_df, idx, ssl_embedding, dns_embedding)
            
    #         # 显示进度
    #         if self.verbose and (idx + 1) % 1000 == 0:
    #             progress = (idx + 1) / len(flow_df) * 100
    #             elapsed = time.time() - start_time
    #             speed = (idx + 1) / elapsed if elapsed > 0 else 0
    #             logger.info(f"处理进度: {progress:.1f}% ({idx + 1}/{len(flow_df)}), 速度: {speed:.1f} 行/秒")
        
    #     total_time = time.time() - start_time
    #     if self.verbose:
    #         logger.info(f"串行嵌入完成，总耗时: {total_time:.2f}秒")
    #         logger.info(f"  - 处理速度: {len(flow_df)/total_time:.1f} 行/秒")
        
    #     return embedded_flow_df

    def _assign_embedding_to_row(self, embedded_flow_df, row_idx, ssl_embedding, dns_embedding):
        """
        将嵌入结果赋值给DataFrame的指定行
        """
        # 处理SSL嵌入
        if hasattr(ssl_embedding, 'tolist'):
            vec_list = ssl_embedding.tolist()
        else:
            vec_list = [0.0] * self.num_apps
        
        for i in range(5):
            embedded_flow_df.at[row_idx, f'ssl.server_name{i}_freq'] = vec_list
        
        # 处理DNS嵌入
        if hasattr(dns_embedding, 'tolist'):
            vec_list = dns_embedding.tolist()
        else:
            vec_list = [0.0] * self.num_apps
        
        for i in range(5):
            embedded_flow_df.at[row_idx, f'dns.query{i}_freq'] = vec_list