import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random

# 配置参数
class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 512
        self.embedding_dim = 64
        self.max_seq_length = 201  # 最大序列长度
        self.num_heads = 2
        self.num_blocks = 2  # transformer block数量
        self.dropout_rate = 0.1
        self.lr = 0.001
        self.num_epochs = 10
        self.item_vocab_size = 5000000  # 根据你的数据调整
        self.pad_token = 0
        self.eval_per_step = 100

config = Config()

# 自定义数据集（现在将target_item作为序列最后一个元素）
class SASRecDataset(Dataset):
    def __init__(self, df, config):
        self.data = df
        self.config = config
        self.pad_token = config.pad_token
        self.max_len = config.max_seq_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 将target_item作为序列最后一个元素
        full_seq = row['item_hist'].tolist() + [row['target_item']]
        full_seq = full_seq[-self.max_len:]  # 截断
        
        # 输入序列（去掉最后一个元素）
        input_seq = full_seq[:-1]
        # 目标序列（去掉第一个元素，每个位置预测下一个）
        target_seq = full_seq[1:]
        
        seq_len = len(input_seq)
        
        # 填充短序列
        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            input_seq = [self.pad_token] * pad_len + list(input_seq)
            target_seq = [self.pad_token] * pad_len + list(target_seq)
        
        # 转换为tensor
        input_seq = torch.LongTensor(input_seq)
        target_seq = torch.LongTensor(target_seq)
        
        # 创建mask（忽略padding位置的loss）
        mask = (input_seq != self.pad_token).float()
        
        return input_seq, target_seq, mask

class PreNormTransformerBlock(nn.Module):
    def __init__(self, config):
        super(PreNormTransformerBlock, self).__init__()
        embed_dim = config.embedding_dim
        num_heads = config.num_heads
        ff_dim = config.ffn_dim if hasattr(config, 'ffn_dim') else embed_dim * 4
        dropout = config.dropout_rate

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # PreNorm: x + f(LN(x))
        # Attention block
        residual1 = x
        x = self.norm1(x)
        attn_out, _ = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        x = residual1 + self.dropout(attn_out)

        # FFN block
        residual2 = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual2 + self.dropout(ffn_out)

        return x


class SASRecModel(nn.Module):
    def __init__(self, config):
        super(SASRecModel, self).__init__()
        self.config = config
        self.item_emb = nn.Embedding(
            config.item_vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token
        )
        self.pos_emb = nn.Embedding(config.max_seq_length, config.embedding_dim)
        # 使用自定义的 PreNorm Block
        self.transformer_blocks = nn.ModuleList([
            PreNormTransformerBlock(config) for _ in range(config.num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_seq):
        seq_mask = (input_seq == self.config.pad_token)  # (batch_size, seq_len)
        positions = torch.arange(input_seq.size(1), dtype=torch.long, device=input_seq.device)
        positions = positions.unsqueeze(0).expand_as(input_seq)

        item_embs = self.item_emb(input_seq)
        pos_embs = self.pos_emb(positions)
        seq_embs = item_embs + pos_embs
        seq_embs = self.dropout(seq_embs)

        # 注意力 mask（上三角，防止未来信息泄露）
        attn_mask = self._get_attention_mask(input_seq.size(1)).to(input_seq.device)

        out = seq_embs
        for block in self.transformer_blocks:
            out = block(out, src_mask=attn_mask, src_key_padding_mask=seq_mask)

        out = self.layer_norm(out)
        return out  # (batch_size, seq_len, embedding_dim)

    def _get_attention_mask(self, seq_len):
        # 生成上三角 mask，屏蔽未来 token
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask  # (seq_len, seq_len), 注意：会被广播到 (B, N, N) 如果使用 attn_mask

# BPR Loss实现
class SequenceLoss(nn.Module):
    def __init__(self, num_negatives=100, item_vocab_size=5000000, embedding_dim=64):
        super(SequenceLoss, self).__init__()
        self.num_negatives = num_negatives
        self.item_vocab_size = item_vocab_size
        self.embedding_dim = embedding_dim

    def forward(self, seq_embs, target_seq, mask, item_emb_layer):
        # seq_embs: (batch_size, seq_len, embedding_dim)
        # target_seq: (batch_size, seq_len)
        # mask: (batch_size, seq_len)
        batch_size, seq_len = target_seq.size()
        device = seq_embs.device

        # 正样本embedding
        pos_emb = item_emb_layer(target_seq)  # (batch_size, seq_len, embedding_dim)

        # 负样本采样
        neg_items = torch.randint(
            1, self.item_vocab_size, 
            (batch_size, seq_len, self.num_negatives), 
            device=device
        )
        neg_emb = item_emb_layer(neg_items)  # (batch_size, seq_len, num_negatives, embedding_dim)

        # 扩展seq_embs用于与负样本做内积
        seq_embs_exp = seq_embs.unsqueeze(2)  # (batch_size, seq_len, 1, embedding_dim)
        pos_score = (seq_embs * pos_emb).sum(-1)  # (batch_size, seq_len)
        neg_score = (seq_embs_exp * neg_emb).sum(-1)  # (batch_size, seq_len, num_negatives)

        # BPR loss: log(sigmoid(pos_score - neg_score))
        bpr_loss = -torch.log(torch.sigmoid(pos_score.unsqueeze(-1) - neg_score) + 1e-8)  # (batch_size, seq_len, num_negatives)
        # mask扩展
        mask = mask.unsqueeze(-1).expand_as(bpr_loss)
        masked_loss = bpr_loss * mask
        return masked_loss.sum() / mask.sum()

# 加载数据
def load_data(dir_path):
    train_df = pd.read_parquet(f'{dir_path}/train.parquet')
    val_df = pd.read_parquet(f'{dir_path}/valid.parquet')
    test_df = pd.read_parquet(f'{dir_path}/test.parquet')
    
    # 确保item_hist是列表形式
    for df in [train_df, val_df, test_df]:
        if not isinstance(df['item_hist'].iloc[0], list):
            df['item_hist'] = df['item_hist'].to_list()
    
    return train_df, val_df, test_df


def eval_model(model, val_loader, criterion, train_loss, cnt):        
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_seq, target_seq, mask = [x.to(config.device) for x in batch]
            seq_embs = model(input_seq)
            loss = criterion(seq_embs, target_seq, mask, model.item_emb)
            val_loss += loss.item()
    
    avg_train_loss = train_loss / config.eval_per_step
    avg_val_loss = val_loss / len(val_loader)
    print(f'Step {cnt+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

    model.train()
    return avg_val_loss

# 训练函数
def train_model():
    # 加载数据
    dir_path = '/home/liqun03/FuxiCTR/my_datasets/taobao' 
    train_df, val_df, test_df = load_data(dir_path)
    
    # 创建数据集和数据加载器
    train_dataset = SASRecDataset(train_df, config)
    val_dataset = SASRecDataset(val_df, config)
    test_dataset = SASRecDataset(test_df, config)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # 初始化模型
    model = SASRecModel(config).to(config.device)
    criterion = SequenceLoss(
        num_negatives=100, 
        item_vocab_size=config.item_vocab_size, 
        embedding_dim=config.embedding_dim
    )
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float('inf')
    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        cnt = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            cnt += 1
            input_seq, target_seq, mask = [x.to(config.device) for x in batch]
            optimizer.zero_grad()
            seq_embs = model(input_seq)
            loss = criterion(seq_embs, target_seq, mask, model.item_emb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(loss.item())
            if cnt % config.eval_per_step == 0:
                avg_val_loss = eval_model(model, val_loader, criterion, train_loss, cnt)
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), 'sasrec_seq_best_model.pth')
                train_loss = 0
    
    # 测试
    model.load_state_dict(torch.load('sasrec_seq_best_model.pth'))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_seq, target_seq, mask = [x.to(config.device) for x in batch]
            seq_embs = model(input_seq)
            loss = criterion(seq_embs, target_seq, mask, model.item_emb)
            test_loss += loss.item()
    
    print(f'Test Loss: {test_loss / len(test_loader):.4f}')
    
    # 保存item表征
    save_item_embeddings(model)

def save_item_embeddings(model):
    """保存所有item的embedding"""
    item_embeddings = model.item_emb.weight.detach().cpu().numpy()
    np.save('sasrec_item_embeddings.npy', item_embeddings)
    print(f"Item embeddings saved with shape: {item_embeddings.shape}")

if __name__ == '__main__':
    train_model()