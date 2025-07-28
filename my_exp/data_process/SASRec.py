import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

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
        target_len = self.config.max_seq_length - 1
        
        # 填充短序列
        if seq_len < target_len:
            pad_len = target_len - seq_len
            input_seq = [self.pad_token] * pad_len + list(input_seq)
            target_seq = [self.pad_token] * pad_len + list(target_seq)
        
        # 转换为tensor
        input_seq = torch.LongTensor(input_seq)
        target_seq = torch.LongTensor(target_seq)
        
        # 创建mask（忽略padding位置的loss）
        mask = (input_seq != self.pad_token).float()
        
        return input_seq, target_seq, mask


class SASRecModel(nn.Module):
    def __init__(self, config):
        super(SASRecModel, self).__init__()
        self.item_vocab_size = config.item_vocab_size
        self.embedding_dim = config.embedding_dim
        self.max_seq_length = config.max_seq_length - 1  # -1是因为输入序列不包含target
        self.num_blocks = config.num_blocks
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        self.pad_token = config.pad_token
        
        # 物品嵌入层
        self.item_emb = nn.Embedding(config.item_vocab_size, config.embedding_dim, padding_idx=config.pad_token)
        
        # 位置编码
        self.positional_emb = nn.Embedding(self.max_seq_length, config.embedding_dim)
        
        # LayerNorm & Dropout
        self.emb_dropout = nn.Dropout(config.dropout_rate)
        self.layernorm = nn.LayerNorm(config.embedding_dim)
        
        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.embedding_dim, config.num_heads, config.dropout_rate)
            for _ in range(config.num_blocks)
        ])
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, input_seq):
        # input_seq: [batch_size, seq_len]
        batch_size, seq_len = input_seq.size()
        device = input_seq.device
        
        # 创建位置id
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(input_seq)
        
        # 创建attention mask (避免关注padding和未来位置)
        attention_mask = self.get_attention_mask(input_seq)
        
        # 嵌入层 + 位置编码
        item_embeddings = self.item_emb(input_seq)
        pos_embeddings = self.positional_emb(pos_ids)
        seqs = item_embeddings + pos_embeddings
        
        # Dropout & Layer Norm
        seqs = self.emb_dropout(seqs)
        seqs = self.layernorm(seqs)
        
        # 传递每个Transformer块
        for block in self.transformer_blocks:
            seqs = block(seqs, attention_mask)
            
        # 返回最终的序列表示
        return seqs
    
    def get_attention_mask(self, input_seq):
        """
        创建attention mask矩阵，用于:
        1. 避免关注padding位置
        2. 实现因果注意力 (Causal Attention)：每个位置只能关注自己和之前的位置
        """
        batch_size, seq_len = input_seq.size()
        device = input_seq.device
        
        # Padding mask: 将padding位置设为True (会被masked out)
        pad_mask = (input_seq == self.pad_token)  # [batch_size, seq_len]
        
        # 扩展padding mask: [batch_size, 1, 1, seq_len]
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        
        # 因果注意力掩码(确保每个位置只看到过去的信息)
        # 创建一个上三角矩阵(对角线以上为True)
        subsequent_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device), diagonal=1
        )
        subsequent_mask = (subsequent_mask == 1)  # 转换为布尔型
        
        # 扩展到所有batch: [1, 1, seq_len, seq_len]
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)
        
        # 合并两个掩码: [batch_size, 1, seq_len, seq_len]
        # 只要其中一个为True，就masked out
        attention_mask = pad_mask.expand(-1, -1, seq_len, -1) | subsequent_mask
        
        return attention_mask


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout_rate)
        self.feed_forward = PointWiseFeedForward(hidden_size, dropout_rate)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, attention_mask=None):
        # Self-Attention
        attention_output = self.attention(x, x, x, attention_mask)
        # 残差连接和Layer Normalization
        x = self.layernorm1(x + self.dropout(attention_output))
        # Feed Forward Network
        ffn_output = self.feed_forward(x)
        # 残差连接和Layer Normalization
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        assert self.head_size * num_heads == hidden_size, "hidden_size必须能被num_heads整除"
        
        # Query, Key, Value投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def transpose_for_scores(self, x):
        # x: [batch_size, seq_len, hidden_size]
        batch_size, seq_len = x.size(0), x.size(1)
        # 重塑和转置: [batch_size, seq_len, num_heads, head_size] -> [batch_size, num_heads, seq_len, head_size]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)
        
        # 线性投影
        q = self.q_linear(query)  # [batch_size, seq_len, hidden_size]
        k = self.k_linear(key)
        v = self.v_linear(value)
        
        # 重塑以进行多头注意力
        q = self.transpose_for_scores(q)  # [batch_size, num_heads, seq_len, head_size]
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        
        # 缩放点积注意力
        # q * k^T / sqrt(head_size)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores / (self.head_size ** 0.5)
        
        # 应用注意力掩码(如果有)
        if attention_mask is not None:
            # 将mask中为True的位置设置为一个非常小的值，这样在softmax后接近于0
            attention_scores = attention_scores.masked_fill(attention_mask, -1e9)
        
        # Softmax归一化权重
        attention_probs = torch.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attention_probs = self.dropout(attention_probs)
        
        # 加权求和
        context_layer = torch.matmul(attention_probs, v)  # [batch_size, num_heads, seq_len, head_size]
        
        # 转置回原始形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_size]
        context_layer = context_layer.view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        
        # 输出投影
        output = self.out_proj(context_layer)
        
        return output


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


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
            df['item_hist'] = df['item_hist'].apply(list)
    
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