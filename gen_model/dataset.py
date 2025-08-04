# dataset.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from GenRecTokenizer import GenRecTokenizer


class ParquetDataset(Dataset):
    """
    Custom Dataset for reading data from a Parquet file.
    """
    def __init__(self, file_path: str):
        self.data = pd.read_parquet(file_path)

        list_col = ['item_hist', 'top_20_items', 'top_20_sims']
        for col in list_col:
            if isinstance(self.data[col].iloc[0], np.ndarray):
                self.data[col] = self.data[col].apply(list)

        item_hist_list = self.data['item_hist'].tolist()
        target_item_list = self.data['target_item'].tolist()
        final_input_seq_list = [hist + [-2, target] for hist, target in zip(item_hist_list, target_item_list)]
        self.data['input_seq'] = final_input_seq_list

        self.data['top_20_sims'] = self.data['top_20_sims'].apply(
            lambda sims: [(int(i * 10000) + 10000) / 2 for i in sims] # [-1, 1] -> [0, 10000]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.data.iloc[idx]
        return {
            "input_seq": row["input_seq"],
            "target_seq": row["top_20_items"],
            "target_sim": row["top_20_sims"]
        }

class SimDataLoader(DataLoader):
    def __init__(self, data_path, tokenizer, batch_size=32, shuffle=False, num_workers=1):
        self.batch_size = batch_size
        self.dataset = ParquetDataset(data_path)
        
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=BatchCollator(tokenizer)
        )
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches

class BatchCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_sequences = [item['input_seq'] for item in batch]
        target_sequences = [item['target_seq'] for item in batch]
        target_similarities = [item['target_sim'] for item in batch]

        encoder_inputs = self.tokenizer(input_sequences)
        labels = self.tokenizer(target_sequences, target_similarities)["input_ids"]

        return {
            "input_ids": encoder_inputs["input_ids"],
            "attention_mask": encoder_inputs["attention_mask"],
            "labels": labels,
        }

class TaobaoSimDataLoader(object):
    def __init__(self, stage="both", 
                 train_data=None, valid_data=None, test_data=None,
                 map_file_path=None, token_level_vocab_sizes=None,
                 special_vocab_size=None, batch_size=32):
        print("Loading datasets...")
        train_gen = None
        valid_gen = None
        test_gen = None
        self.stage = stage
        
        tokenizer = GenRecTokenizer(
            map_file_path=map_file_path,
            token_level_vocab_sizes=token_level_vocab_sizes,
            special_vocab_size=special_vocab_size
        )

        if stage in ["both", "train"]:
            train_gen = SimDataLoader(train_data, tokenizer, batch_size=batch_size, shuffle=True)
            print(
                "Train samples: total/{:d}, blocks/{:d}"
                .format(train_gen.num_samples, train_gen.num_blocks)
            )     
            if valid_data:
                valid_gen = SimDataLoader(valid_data, tokenizer, batch_size=batch_size)
                print(
                    "Validation samples: total/{:d}, blocks/{:d}"
                    .format(valid_gen.num_samples, valid_gen.num_blocks)
                )

        if stage in ["both", "test"]:
            if test_data:
                test_gen = SimDataLoader(test_data, tokenizer, batch_size=batch_size)
                print(
                    "Test samples: total/{:d}, blocks/{:d}"
                    .format(test_gen.num_samples, test_gen.num_blocks)
                )

        self.train_gen, self.valid_gen, self.test_gen = train_gen, valid_gen, test_gen

    def make_iterator(self):
        if self.stage == "train":
            print("Loading train and validation data done.")
            return self.train_gen, self.valid_gen
        elif self.stage == "test":
            print("Loading test data done.")
            return self.test_gen
        else:
            print("Loading data done.")
            return self.train_gen, self.valid_gen, self.test_gen