# train.py
import torch
from torch.utils.data import DataLoader
from functools import partial
import pandas as pd
import numpy as np
import os

# Import local modules
import config
from model import GenRecMultiHead
from dataset import YourTokenizer, ParquetDataset, collate_fn
from trainer import Trainer
from transformers import T5Config

def create_dummy_data():
    """Creates a dummy parquet file for demonstration."""
    data_dir = os.path.dirname(config.TRAIN_DATA_PATH)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if os.path.exists(config.TRAIN_DATA_PATH):
        print("Dummy data file already exists.")
        return

    print("Creating dummy parquet file...")
    num_samples = 100
    data = []
    for _ in range(num_samples):
        # Generate random sequences of integers
        input_len = np.random.randint(5, config.MAX_INPUT_LENGTH)
        target_len = np.random.randint(5, config.MAX_TARGET_LENGTH)
        
        # Example: input tokens are from all vocab levels
        total_vocab = config.SPECIAL_VOCAB_SIZE + sum(config.TOKEN_LEVEL_VOCAB_SIZES)
        input_seq = np.random.randint(10, total_vocab, size=input_len).tolist()
        
        # Example: target tokens correspond to their levels
        target_seq = []
        level_offsets = [0] * len(config.TOKEN_LEVEL_VOCAB_SIZES)
        cumulative_offset = config.SPECIAL_VOCAB_SIZE
        for i, size in enumerate(config.TOKEN_LEVEL_VOCAB_SIZES):
            level_offsets[i] = cumulative_offset
            cumulative_offset += size
            
        for i in range(target_len):
            level_idx = i % len(config.TOKEN_LEVEL_VOCAB_SIZES)
            token_id = np.random.randint(0, config.TOKEN_LEVEL_VOCAB_SIZES[level_idx])
            target_seq.append(level_offsets[level_idx] + token_id)
            
        data.append({"input_seq": input_seq, "target_seq": target_seq})

    df = pd.DataFrame(data)
    df.to_parquet(config.TRAIN_DATA_PATH)
    print(f"Dummy data saved to {config.TRAIN_DATA_PATH}")


def main():
    # 1. Create dummy data for demonstration
    create_dummy_data()

    # 2. Setup Device
    device = torch.device(config.DEVICE)
    
    # 3. Initialize Tokenizer
    tokenizer = YourTokenizer(pad_token_id=config.PAD_TOKEN_ID)

    # 4. Create Datasets and DataLoaders
    train_dataset = ParquetDataset(file_path=config.TRAIN_DATA_PATH)
    eval_dataset = ParquetDataset(file_path=config.EVAL_DATA_PATH)
    
    # Use partial to pass the tokenizer to collate_fn
    collate_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_with_tokenizer
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_with_tokenizer
    )

    # 5. Initialize Model
    model_config = T5Config(
        vocab_size=1, # Not directly used, handled by multi-level heads
        d_model=config.MODEL_DIM,
        d_kv=config.MODEL_DIM // 2,
        d_ff=config.MODEL_DIM * 2,
        num_layers=config.NUM_LAYERS,
        decoder_start_token_id=config.DECODER_START_TOKEN_ID,
        # Custom attributes
        token_levels=len(config.TOKEN_LEVEL_VOCAB_SIZES) 
    )

    model = GenRecMultiHead(
        config=model_config,
        token_level_vocab_sizes=config.TOKEN_LEVEL_VOCAB_SIZES,
        special_vocab_size=config.SPECIAL_VOCAB_SIZE,
        loss_temperature=config.LOSS_TEMPERATURE
    )

    # 6. Initialize Trainer and Start Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device
    )
    trainer.train()

if __name__ == '__main__':
    main()