# dataset.py
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Any

import config # Import configuration


class ParquetDataset(Dataset):
    """
    Custom Dataset for reading data from a Parquet file.
    """
    def __init__(self, file_path: str):
        # For large files, consider using pyarrow to read row by row
        # instead of loading the whole file into memory with pandas.
        self.data = pd.read_parquet(file_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """
        !!! IMPORTANT !!!
        This is the method you need to implement based on your data format.
        It should return a dictionary containing two lists of integers:
        - 'input_seq': The input sequence for the encoder.
        - 'target_seq': The target sequence for the decoder.
        """
        # --- YOUR LOGIC HERE ---
        # Example implementation:
        row = self.data.iloc[idx]
        # Assuming your parquet file has columns 'input_seq' and 'target_seq'
        # which store lists of integers.
        input_seq = row['input_seq']
        target_seq = row['target_seq']
        
        return {
            "input_seq": input_seq,
            "target_seq": target_seq,
        }

def collate_fn(batch: List[Dict[str, List[int]]], tokenizer: YourTokenizer) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of data points into a single batch tensor.
    """
    input_sequences = [item['input_seq'] for item in batch]
    target_sequences = [item['target_seq'] for item in batch]

    encoder_inputs = tokenizer(input_sequences)
    labels = tokenizer(target_sequences)["input_ids"]
    
    return {
        "input_ids": encoder_inputs["input_ids"],
        "attention_mask": encoder_inputs["attention_mask"],
        "labels": labels,
    }