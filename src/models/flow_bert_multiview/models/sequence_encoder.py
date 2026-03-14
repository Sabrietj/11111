import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any

import os, sys
import logging
utils_path = os.path.join(os.path.dirname(__file__),  '..', '..', '..', 'utils')
sys.path.insert(0, utils_path) 
# 设置日志
from logging_config import setup_preset_logging

# 只在主进程设置详细日志，其他进程设置错误级别
if int(os.environ.get("RANK", 0)) == 0:
    logger = setup_preset_logging(log_level=logging.INFO)
else:
    # 非主进程设置更高的日志级别减少输出
    logging.getLogger().setLevel(logging.ERROR)
    # 创建一个简单的logger用于非主进程
    logger = logging.getLogger(__name__)

class SequenceEncoder(nn.Module):
    """序列编码器 - 处理不定长序列特征"""
    
    def __init__(
        self, 
        embedding_dim: int, 
        num_layers: int,
        num_heads: int,  # 多头注意力头数，统一使用 num_heads
        dropout: float = 0.1,
        max_packet_seq_length: int = 1000  # 提供默认值以保持向后兼容，但推荐从外部传入
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 序列特征嵌入层
        self.direction_projection = nn.Linear(1, embedding_dim)
        self.payload_projection = nn.Linear(1, embedding_dim)
        self.iat_projection = nn.Linear(1, embedding_dim)
        self.packet_number_projection = nn.Linear(1, embedding_dim)
        self.avg_payload_projection = nn.Linear(1, embedding_dim)
        self.duration_projection = nn.Linear(1, embedding_dim)

        # 特征融合投影层：融合方向、载荷、IAT、数据包数、平均载荷大小、持续时间 六个维度
        self.feature_fusion_projection = nn.Linear(6 * embedding_dim, embedding_dim)

        # 位置编码
        self.position_embedding = nn.Embedding(max_packet_seq_length, embedding_dim)
        
        # 添加LayerNorm层提高稳定性
        # self.direction_norm = nn.LayerNorm(embedding_dim)
        # self.payload_norm = nn.LayerNorm(embedding_dim)
        # self.iat_norm = nn.LayerNorm(embedding_dim)
        # self.packet_number_norm = nn.LayerNorm(embedding_dim)
        # self.avg_payload_norm = nn.LayerNorm(embedding_dim)  
        # self.duration_norm = nn.LayerNorm(embedding_dim)
        # self.feature_fusion_norm = nn.LayerNorm(embedding_dim)
        # self.combined_norm = nn.LayerNorm(embedding_dim) 
        
        # Transformer编码器
        # nn.TransformerEncoderLayer 参数说明：
        #   - d_model: 输入维度
        #   - nhead: 多头注意力头数（注意：这里参数名是 nhead，不是 num_heads）
        #   - dim_feedforward: 前馈网络隐藏层大小
        #   - dropout: dropout比例
        #   - batch_first: True 表示输入为 [batch, seq_len, feature_dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True ## ✅✅✅ 必须加上这行！开启 Pre-LayerNorm,这是解决 Transformer 梯度 Inf 问题的标准解法。将结构改为 Pre-LayerNorm 可以极大地稳定梯度。
            ## 主干路径上的梯度不需要经过 Norm 层，梯度流动更加“平滑”，极难出现 Inf 爆炸。
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 注意力池化
        # nn.MultiheadAttention 参数说明：
        #   - embed_dim: 输入特征维度
        #   - num_heads: 多头注意力头数（这里参数名是 num_heads）
        #   - dropout: dropout比例
        #   - batch_first: True 表示输入为 [batch, seq_len, feature_dim]
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
    def forward(self, sequence_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            sequence_data: 包含序列特征的字典（修改后的结构）
                - directions: [batch_size, seq_len] 传输方向的序列
                - payload_sizes: [batch_size, seq_len] 载荷大小序列
                - iat_times: [batch_size, seq_len] 时间间隔序列
                - packet_numbers: [batch_size, seq_len] 数据包数量序列
                - avg_payload_sizes: [batch_size, seq_len] 平均载荷大小序列           
                - durations: [batch_size, seq_len] 持续时间序列
                - sequence_mask: [batch_size, seq_len] 序列掩码
                
        Returns:
            sequence_embeddings: [batch_size, embedding_dim] 序列嵌入表示
        """
        directions = sequence_data['directions']
        payload_sizes = sequence_data['payload_sizes']        
        iat_times = sequence_data['iat_times']
        packet_numbers = sequence_data['packet_numbers']
        avg_payload_sizes = sequence_data['avg_payload_sizes']
        durations = sequence_data['durations']
        sequence_mask = sequence_data['sequence_mask']

        batch_size, seq_len = directions.shape
        
        assert seq_len <= self.position_embedding.num_embeddings, (
            f"Sequence length {seq_len} exceeds max_seq_length "
            f"{self.position_embedding.num_embeddings}"
        )

        # 1️⃣ 内容特征 embedding
        directions_emb = self.direction_projection(directions.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # directions_emb = self.direction_norm(directions_emb)

        payload_emb = self.payload_projection(payload_sizes.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # payload_emb = self.payload_norm(payload_emb)        

        iat_emb = self.iat_projection(iat_times.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # iat_emb = self.iat_norm(iat_emb)

        packet_number_emb = self.packet_number_projection(packet_numbers.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # packet_number_emb = self.packet_number_norm(packet_number_emb)

        avg_payload_emb = self.avg_payload_projection(avg_payload_sizes.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # avg_payload_emb = self.avg_payload_norm(avg_payload_emb)

        duration_emb = self.duration_projection(durations.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # duration_emb = self.duration_norm(duration_emb)

        # 2️⃣ 内容融合
        combined_emb = torch.cat(
            [directions_emb, payload_emb, iat_emb, packet_number_emb, avg_payload_emb, duration_emb],
            dim=-1
        )
        combined_emb = self.feature_fusion_projection(combined_emb)        

        # 3️⃣ 加位置编码
        positions = torch.arange(seq_len, device=directions.device).unsqueeze(0).expand(batch_size, -1)

        # 特殊处理空序列
        valid_token_count = sequence_mask.sum(dim=1)  # [B]
        empty_mask = (valid_token_count == 0)  # [B]

        # 对空序列样本，强制填充一个安全 token
        if empty_mask.any():
            # logger.warning(
            #     f"⚠ Found empty packet sequence batch item: "
            #     f"{empty_mask.nonzero(as_tuple=False).flatten().tolist()}"
            # )
            sequence_mask = sequence_mask.clone()
            sequence_mask[empty_mask, 0] = 1  # 强制至少有一个 token 可见

            # 重新计算 valid_token_count 和 empty_mask
            valid_token_count = sequence_mask.sum(dim=1).float()
            empty_mask = (valid_token_count == 0)
        
        pos_emb = self.position_embedding(positions)  # [batch_size, seq_len, emb_dim]
        
        # 组合token语义编码和位置编码，通过逐元素相加（element-wise sum）
        combined_emb = combined_emb + pos_emb
        
        # -------- 把 mask 用在 Transformer 和 AttentionPooling --------        
        # combined_emb = self.combined_norm(combined_emb)
        # Transformer编码 - 使用正确的掩码格式
        # src_key_padding_mask: True表示需要被mask的位置

        # -------- 关键修复开始：处理 Mask --------

        # 定义一个局部变量 padding_mask 来存储计算后的掩码
        
        if sequence_mask is not None:
            # sequence_mask: 1为有效，0为padding
            # PyTorch要求: False为有效，True为padding/masked
            padding_mask = ~sequence_mask.bool()
            
            # ✅ [修复] 防止全 True 掩码导致的 Softmax NaN
            # 检查是否有样本的所有 token 都被 mask (即全 True)
            all_masked = padding_mask.all(dim=1)  # [batch_size]
            if all_masked.any():
                # 如果发现全被 mask 的样本，强制 unmask 第一个 token (设为 False)
                # 这样 Transformer 至少能关注到 padding 向量，避免 Softmax(-inf) = NaN
                padding_mask[all_masked, 0] = False
        else:
            padding_mask = None
        
        # -------- 关键修复结束 --------

        sequence_token_states = self.transformer(
            combined_emb,
            src_key_padding_mask=padding_mask
        )

        # ===== 之前不安全的没法对空序列做处理的池化操作 =====
        # # 注意力池化得到序列表示，输出 pooled vector
        # # query：序列的全局表示（均值或 learnable CLS）        
        # query = torch.mean(sequence_token_states, dim=1, keepdim=True)  # [batch_size, 1, emb_dim]
        # # 关键点：query 只有 1 个 token，key/value 有 L 个 token        
        # attn_output, attn_weights = self.attention_pooling(
        #     query, sequence_token_states, sequence_token_states, 
        #     key_padding_mask=padding_mask
        # )
        # pooled_sequence_embedding = attn_output.squeeze(1)  # [B,H] = [batch_size, emb_dim]

        # ===== 可以安全处理空序列的池化操作 =====
        # 避免 mean over all masked tokens 导致 NaN 的安全池化方法：
        mask_float = sequence_mask.float().unsqueeze(-1)
        safe_mean = (sequence_token_states * mask_float).sum(dim=1) / valid_token_count.float().clamp(min=1).unsqueeze(-1)
        query = safe_mean.unsqueeze(1)  # [B,1,H]
        attn_output, _ = self.attention_pooling(
            query,
            sequence_token_states,
            sequence_token_states,
            key_padding_mask=padding_mask
        )
        pooled_sequence_embedding = attn_output.squeeze(1)


        if empty_mask.any():
            pooled_sequence_embedding[empty_mask] = 0.0

        if torch.isnan(pooled_sequence_embedding).any():
            raise RuntimeError("Sequence embedding NaN detected!")

        return {
            "sequence_embedding": pooled_sequence_embedding,
        }
