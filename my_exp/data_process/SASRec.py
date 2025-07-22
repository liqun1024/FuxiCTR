import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# 配置参数
class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1024
        self.embedding_dim = 64
        self.max_seq_length = 201  # 最大序列长度
        self.num_heads = 2
        self.num_blocks = 2  # transformer block数量
        self.dropout_rate = 0.1
        self.lr = 0.001
        self.num_epochs = 10
        self.item_vocab_size = 5000000  # 根据你的数据调整
        self.pad_token = 0

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
        full_seq = row['item_hist'] + [row['target_item']]
        full_seq = full_seq[-self.max_len:]  # 截断
        
        # 输入序列（去掉最后一个元素）
        input_seq = full_seq[:-1]
        # 目标序列（去掉第一个元素，每个位置预测下一个）
        target_seq = full_seq[1:]
        
        seq_len = len(input_seq)
        
        # 填充短序列
        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            input_seq = [self.pad_token] * pad_len + input_seq
            target_seq = [self.pad_token] * pad_len + target_seq
        
        # 转换为tensor
        input_seq = torch.LongTensor(input_seq)
        target_seq = torch.LongTensor(target_seq)
        
        # 创建mask（忽略padding位置的loss）
        mask = (input_seq != self.pad_token).float()
        
        return input_seq, target_seq, mask

# SASRec模型（改为序列预测）
class SASRecModel(nn.Module):
    def __init__(self, config):
        super(SASRecModel, self).__init__()
        self.config = config
        
        # Item embedding
        self.item_emb = nn.Embedding(
            config.item_vocab_size, 
            config.embedding_dim, 
            padding_idx=config.pad_token
        )
        
        # Positional embedding
        self.pos_emb = nn.Embedding(config.max_seq_length, config.embedding_dim)
        
        # Transformer blocks
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dropout=config.dropout_rate,
                batch_first=True
            ) for _ in range(config.num_blocks)
        ])
        
        # LayerNorm and Dropout
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Output layer
        self.output_layer = nn.Linear(config.embedding_dim, config.item_vocab_size)
        
    def forward(self, input_seq):
        # 获取mask (padding部分为True)
        seq_mask = (input_seq == self.config.pad_token)
        
        # 获取位置编码
        positions = torch.arange(input_seq.size(1), dtype=torch.long, device=input_seq.device)
        positions = positions.unsqueeze(0).expand_as(input_seq)
        
        # Embedding
        item_embs = self.item_emb(input_seq)
        pos_embs = self.pos_emb(positions)
        seq_embs = item_embs + pos_embs
        seq_embs = self.dropout(seq_embs)
        
        # Transformer blocks
        attention_mask = self._get_attention_mask(input_seq)
        out = seq_embs
        for layer in self.attention_layers:
            out = layer(out, src_key_padding_mask=seq_mask, mask=attention_mask)
        
        out = self.layer_norm(out)
        
        # 预测每个位置的输出
        logits = self.output_layer(out)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def _get_attention_mask(self, input_seq):
        """生成因果掩码，防止看到未来信息"""
        batch_size, seq_len = input_seq.size()
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(input_seq.device)

# 序列预测损失函数
class SequenceLoss(nn.Module):
    def __init__(self):
        super(SequenceLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
    def forward(self, logits, target_seq, mask):
        # logits: (batch_size, seq_len, vocab_size)
        # target_seq: (batch_size, seq_len)
        # mask: (batch_size, seq_len)
        
        # 计算每个位置的loss
        loss = self.criterion(
            logits.view(-1, logits.size(-1)), 
            target_seq.view(-1)
        ).view_as(target_seq)
        
        # 只计算有效位置的loss
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()

# 加载数据
def load_data():
    train_df = pd.read_parquet('train.parquet')
    val_df = pd.read_parquet('val.parquet')
    test_df = pd.read_parquet('test.parquet')
    
    # 确保item_hist是列表形式
    for df in [train_df, val_df, test_df]:
        if not isinstance(df['item_hist'].iloc[0], list):
            df['item_hist'] = df['item_hist'].apply(eval)
    
    return train_df, val_df, test_df

# 训练函数
def train_model():
    # 加载数据
    train_df, val_df, test_df = load_data()
    
    # 创建数据集和数据加载器
    train_dataset = SASRecDataset(train_df, config)
    val_dataset = SASRecDataset(val_df, config)
    test_dataset = SASRecDataset(test_df, config)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # 初始化模型
    model = SASRecModel(config).to(config.device)
    criterion = SequenceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            input_seq, target_seq, mask = [x.to(config.device) for x in batch]
            
            optimizer.zero_grad()
            logits = model(input_seq)
            loss = criterion(logits, target_seq, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_seq, target_seq, mask = [x.to(config.device) for x in batch]
                logits = model(input_seq)
                loss = criterion(logits, target_seq, mask)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'sasrec_seq_best_model.pth')
    
    # 测试
    model.load_state_dict(torch.load('sasrec_seq_best_model.pth'))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_seq, target_seq, mask = [x.to(config.device) for x in batch]
            logits = model(input_seq)
            loss = criterion(logits, target_seq, mask)
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