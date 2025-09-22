import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TransformerLayer(nn.Module):
    """
    标准的Transformer层，采用pre-norm结构 (LayerNorm -> Attention/FFN -> Residual).
    """
    def __init__(self, embedding_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 使用 batch_first=True 来匹配 (B, T, D) 的输入格式
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embedding_dim),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        """
        前向传播
        :param query: (B, T_q, D)
        :param key: (B, T_k, D)
        :param value: (B, T_v, D)
        :param attn_mask: 类似于key_padding_mask, 形状为(B, 1, T_k),
                          其中1表示有效位置，0表示需要mask的位置.
        """
        # 1. Pre-Norm + Multi-Head Attention + Residual
        # PyTorch的MultiheadAttention期望的key_padding_mask中, True表示需要mask的位置
        # 我们的attn_mask中, 0表示需要mask的位置, 所以需要转换
        key_padding_mask = (attn_mask.squeeze(1) == 0) if attn_mask is not None else None

        # Pre-norm
        q_norm = self.norm1(query)
        k_norm = self.norm1(key)
        v_norm = self.norm1(value)
        
        attn_output, _ = self.attention(q_norm, k_norm, v_norm, key_padding_mask=key_padding_mask)
        x = query + self.dropout1(attn_output)

        # 2. Pre-Norm + FFN + Residual
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        output = x + self.dropout2(ffn_output)
        
        return output

class ScaledDotProductAttention(nn.Module):
    """一个简单的缩放点积注意力实现"""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.embedding_dim ** 0.5
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        output = torch.bmm(attn, v)
        return output

class CategoryInterestAttention(nn.Module):
    """
    两阶段注意力模块:
    1. 类目内注意力: 对每个category分组，以该category最后一个item为Q，该category所有item为KV，得到category兴趣向量。
    2. 目标注意力: 以target_item为Q，以所有category兴趣向量为KV，得到最终的用户序列兴趣向量。
    """
    def __init__(self, embedding_dim: int, max_categories: int, num_heads: int = 4, num_layers: int = 2, dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        初始化
        :param embedding_dim: item嵌入向量的维度
        :param max_categories: 每个用户行为序列中，预设的最大的不重复category数量，用于padding
        :param num_heads: 多头注意力头数
        :param num_layers: 阶段一Transformer的层数
        :param dim_feedforward: FFN中间层维度
        :param dropout: dropout rate
        """
        super(CategoryInterestAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_categories = max_categories
        
        # 阶段一: 两层Transformer叠加
        self.category_attention_stack = nn.ModuleList([
            TransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 阶段二: 简单的缩放点积注意力 (无FFN, 无矩阵变换)
        self.target_attention_net = ScaledDotProductAttention(embedding_dim)

    def forward(self, 
                target_item_emb: torch.Tensor, 
                sequence_item_emb: torch.Tensor, 
                sequence_cat_ids: torch.Tensor, 
                sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param target_item_emb: 目标item的嵌入 (B, D)
        :param sequence_item_emb: 序列item的嵌入 (B, T, D)
        :param sequence_cat_ids: 序列category的ID (B, T)
        :param sequence_mask: 序列的mask (B, T), 1表示有效，0表示padding
        :return: 最终的兴趣向量 (B, D)
        """
        batch_size, seq_len, _ = sequence_item_emb.shape

        # --- 阶段一：类目内注意力 ---

        # 1. 获取每个用户的不重复category和对应的mask
        unique_cats, cat_mask = self.get_unique_categories_and_mask(sequence_cat_ids, sequence_mask)

        # 2. 准备Q, K, V
        cat_match_mask = unique_cats.unsqueeze(2) == sequence_cat_ids.unsqueeze(1)
        cat_match_mask = cat_match_mask & sequence_mask.unsqueeze(1)
        q_indices = self.get_last_item_indices(cat_match_mask, seq_len)
        q_indices = q_indices.clamp(0, seq_len - 1)
        queries = torch.gather(
            sequence_item_emb, 
            dim=1, 
            index=q_indices.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        )
        
        # 3. 执行类目内注意力计算
        queries_flat = queries.view(-1, 1, self.embedding_dim)
        keys_flat = sequence_item_emb.unsqueeze(1).expand(-1, self.max_categories, -1, -1)
        keys_flat = keys_flat.reshape(-1, seq_len, self.embedding_dim)
        attn_mask_flat = cat_match_mask.view(-1, 1, seq_len)
        
        # 阶段一计算: 多层Transformer叠加
        # K和V在各层之间共享，只对Q进行更新
        x = queries_flat
        for layer in self.category_attention_stack:
            x = layer(x, keys_flat, keys_flat, attn_mask_flat)
        category_interest_vectors_flat = x
        
        # 恢复形状
        category_interest_vectors = category_interest_vectors_flat.view(
            batch_size, self.max_categories, self.embedding_dim
        )

        # --- 阶段二：目标注意力 ---
        
        # Q: target_item_emb (B, 1, D)
        # K, V: category_interest_vectors (B, max_cat, D)
        # mask: cat_mask (B, max_cat)
        # 使用简单的Attention, 无FFN和矩阵变换
        final_interest_vector = self.target_attention_net(
            target_item_emb.unsqueeze(1),
            category_interest_vectors,
            category_interest_vectors,
            cat_mask.unsqueeze(1) # (B, 1, max_cat)
        )

        return final_interest_vector.squeeze(1)

    def get_unique_categories_and_mask(self, sequence_cat_ids: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取每个batch中不重复的category ID，并进行padding
        注意: 这个函数为了代码清晰和正确性，在batch维度上使用了循环。
              对于性能要求极致的场景，可以考虑使用更复杂的无循环实现（如利用torch_unique），
              但通常这里的循环不会是主要瓶leaping。
        """
        batch_size = sequence_cat_ids.shape[0]
        device = sequence_cat_ids.device

        all_unique_cats = []
        all_cat_masks = []

        for i in range(batch_size):
            # 获取当前用户有效的category
            valid_cats = torch.masked_select(sequence_cat_ids[i], sequence_mask[i].bool())
            # 找到不重复的category，并保持出现顺序
            unique_cats_i = torch.unique_consecutive(torch.sort(valid_cats)[0])
            
            num_unique = unique_cats_i.shape[0]
            
            # padding到max_categories
            padding_needed = self.max_categories - num_unique
            if padding_needed > 0:
                padded_cats = F.pad(unique_cats_i, (0, padding_needed), 'constant', 0)
            else: # 如果超出，则截断
                padded_cats = unique_cats_i[:self.max_categories]

            # 创建mask
            cat_mask = torch.zeros(self.max_categories, device=device)
            cat_mask[:min(num_unique, self.max_categories)] = 1
            
            all_unique_cats.append(padded_cats)
            all_cat_masks.append(cat_mask)

        return torch.stack(all_unique_cats, dim=0), torch.stack(all_cat_masks, dim=0)

    def get_last_item_indices(self, cat_match_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        获取每个category对应的最后一个item的索引。
        :param cat_match_mask: (B, max_cat, T)
        :param seq_len: 序列长度 T
        :return: (B, max_cat)
        """
        device = cat_match_mask.device
        
        # 创建一个位置序列 (0, 1, 2, ..., T-1)
        positions = torch.arange(seq_len, device=device).float()
        
        # 将不匹配的位置设为-1，然后取argmax即可得到最后一个匹配的位置
        # 加1是为了避免位置0本身被mask掉
        masked_positions = (positions + 1) * cat_match_mask.float()
        
        # argmax会返回第一个最大值的索引，由于我们把不匹配的设为0，匹配的设为pos+1
        # 这等价于找到了最后一个匹配的位置
        last_indices = torch.argmax(masked_positions, dim=2)
        
        return last_indices