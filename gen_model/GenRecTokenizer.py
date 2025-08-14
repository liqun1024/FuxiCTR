import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Union, Tuple

class GenRecTokenizer:
    """
    A tokenizer for generative recommendation models that converts item IDs 
    into hierarchical token sequences.

    Special tokens are represented by negative integers, e.g., -1 is mapped to 
    special token 0, -2 to special token 1, and so on.
    """
    def __init__(self, 
                 map_file_path: str, 
                 token_level_vocab_sizes: List[int], 
                 special_vocab_size: int = 10,
                 pad_token_id: int = 0,
                 item_id_col: str = 'item_id',
                 token_col_prefix: str = 'token_'):
        """
        Initializes the tokenizer.

        Args:
            map_file_path (str): Path to the parquet file containing item-to-token mappings.
            token_level_vocab_sizes (List[int]): List of vocabulary sizes for each token level.
            special_vocab_size (int): Number of special tokens.
            pad_token_id (int): The ID to use for padding.
            item_id_col (str): The column name for item IDs in the map file.
            token_col_prefix (str): The prefix for token columns in the map file.
        """
        self.token_level_vocab_sizes = token_level_vocab_sizes
        self.special_vocab_size = special_vocab_size
        self.pad_token_id = pad_token_id
        self.item_id_col = item_id_col
        self.token_col_prefix = token_col_prefix
        
        self.item_to_tokens_map = self._load_token_map(map_file_path)
        self.tokens_to_item_map = {tuple(v): k for k, v in self.item_to_tokens_map.items()}

        self.level_offsets = [0] * len(token_level_vocab_sizes)
        cumulative_offset = special_vocab_size
        for i, size in enumerate(token_level_vocab_sizes):
            self.level_offsets[i] = cumulative_offset
            cumulative_offset += size
        
        self._create_flat_item_map()

    def _load_token_map(self, map_file_path: str) -> Dict[int, List[int]]:
        df = pd.read_parquet(map_file_path)
        
        if self.item_id_col not in df.columns:
            raise ValueError(f"Item ID column '{self.item_id_col}' not found in the file.")
        df[self.item_id_col] = df[self.item_id_col].astype(int)
        df.set_index(self.item_id_col, inplace=True)

        token_cols = sorted(
            [col for col in df.columns if col.startswith(self.token_col_prefix)], 
            key=lambda x: int(x.split('_')[1])
        )
        assert len(token_cols) + 1 == len(self.token_level_vocab_sizes), \
            "Mismatch between token columns and token level vocab sizes."

        token_map = df[token_cols].apply(lambda row: row.tolist(), axis=1).to_dict()
        return token_map

    def _create_flat_item_map(self):
        num_hierarchical_tokens = len(self.token_level_vocab_sizes) - 1

        item_token_vocabs = self.token_level_vocab_sizes[:num_hierarchical_tokens]

        vocabs = np.array(item_token_vocabs)
        self.strides_for_flat_map = np.array([np.prod(vocabs[i+1:]) for i in range(len(vocabs)-1)] + [1])

        self.flat_tokens_to_item_map = {}
        for unoffset_tokens_tuple, item_id in self.tokens_to_item_map.items():
            flat_key = np.dot(unoffset_tokens_tuple, self.strides_for_flat_map)
            self.flat_tokens_to_item_map[int(flat_key)] = item_id

    def encode(self, item_list: List[int], sim_list: List[int] = None) -> List[int]:
        input_ids = []
        sim_idx = 0
        for item_id in item_list:
            if -self.special_vocab_size <= item_id <= -1:
                token = -item_id - 1  # Convert special token ID to zero-based index
                input_ids.append(token)
            else:
                tokens = self.item_to_tokens_map.get(item_id)
                if tokens is not None:
                    encoded_tokens = [self.level_offsets[i] + token for i, token in enumerate(tokens)]
                    input_ids.extend(encoded_tokens)
                    if sim_list is not None and sim_idx < len(sim_list):
                        input_ids.append(sim_list[sim_idx] + self.level_offsets[-1])  # Append similarity value
                        sim_idx += 1
                else:
                    raise ValueError(f"Item ID {item_id} not found in the token map.")
        
        assert sim_idx == len(sim_list) if sim_list is not None else True, \
            "Mismatch between number of items and similarity values."

        return input_ids

    def __call__(
            self,
            batch_items: List[List[int]],
            batch_items_sim: List[List[int]] = None,
            padding: str = 'longest',
            truncation: str = 'left',
            padding_side: str = 'right',
            max_length: int = None,
            return_tensors: str = 'pt',
    ) -> Dict[str, Union[List[List[int]], 'torch.Tensor']]:
        if not batch_items or not isinstance(batch_items[0], list):
            batch_items = [batch_items]
        if batch_items_sim is not None and not isinstance(batch_items_sim[0], list):
            batch_items_sim = [batch_items_sim]

        if batch_items_sim is not None:
            tokenized_sequences = [self.encode(item_list, sim_list) for item_list, sim_list in zip(batch_items, batch_items_sim)]
        else:
            tokenized_sequences = [self.encode(item_list) for item_list in batch_items]

        if truncation and max_length is not None:
            for i in range(len(tokenized_sequences)):
                if len(tokenized_sequences[i]) > max_length:
                    if truncation == 'left':
                        tokenized_sequences[i] = tokenized_sequences[i][-max_length:]
                    else:
                        tokenized_sequences[i] = tokenized_sequences[i][:max_length]

        batch_input_ids = []
        if padding:
            if padding == 'longest':
                target_len = max(len(seq) for seq in tokenized_sequences)
            elif padding == 'max_length':
                if max_length is None:
                    raise ValueError("`max_length` must be specified for `padding='max_length'`.")
                target_len = max_length
            else:
                raise ValueError(f"Invalid padding strategy: {padding}. Choose 'longest' or 'max_length'.")

            for seq in tokenized_sequences:
                diff = target_len - len(seq)
                if diff >= 0:
                    if padding_side == 'right':
                        padded_seq = seq + [self.pad_token_id] * diff
                    elif padding_side == 'left':
                        padded_seq = [self.pad_token_id] * diff + seq
                    else:
                        raise ValueError(f"Invalid padding_side: {padding_side}. Choose 'left' or 'right'.")
                    batch_input_ids.append(padded_seq)
                else:
                    raise ValueError(f"Tokenized sequence length {len(seq)} exceeds target length {target_len}.")
        else:
            batch_input_ids = tokenized_sequences

        batch_attention_mask = []
        for seq in batch_input_ids:
            mask = [1 if token_id != self.pad_token_id else 0 for token_id in seq]
            batch_attention_mask.append(mask)

        result = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask
        }

        if return_tensors == 'pt':
            result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
            result["attention_mask"] = torch.tensor(result["attention_mask"], dtype=torch.long)

        return result

    def decode(self, sequences: torch.Tensor) -> Tuple[List[int], int]:
        sequences_np = sequences.cpu().numpy()
        
        batch_size, seq_len = sequences_np.shape
        num_hierarchical_tokens = len(self.token_level_vocab_sizes) - 1

        indices = np.arange(seq_len)
        mask = indices % (num_hierarchical_tokens + 1) != num_hierarchical_tokens

        item_chunks = sequences_np[:, mask].reshape(batch_size, -1, num_hierarchical_tokens)
        
        valid_item_mask = (item_chunks[:, :, 0] != self.pad_token_id)
        
        valid_chunks = item_chunks[valid_item_mask]
        
        level_offsets_np = self.level_offsets[:num_hierarchical_tokens]
        unoffset_tokens = valid_chunks - level_offsets_np
        flat_keys = np.dot(unoffset_tokens, self.strides_for_flat_map)
        decoded_ids = np.array([self.flat_tokens_to_item_map.get(key, -1) for key in flat_keys])

        final_decoded_ids = np.full(valid_item_mask.shape, -2, dtype=int) # -2 代表 padding
        final_decoded_ids[valid_item_mask] = decoded_ids

        valid_item_mask = final_decoded_ids != -1
        missing_counts = np.sum(~valid_item_mask, axis=1).tolist()
        valid_item_mask &= (final_decoded_ids != -2)
        decoded_batch = [
            final_decoded_ids[i][valid_item_mask[i]].tolist() for i in range(batch_size)
        ]

        return decoded_batch, missing_counts


if __name__ == "__main__":
    import os
    mock_data = {
        'item_id': [101, 102, 103, 104],
        'token_0': [0, 1, 0, 2],
        'token_1': [5, 12, 8, 15],
        'token_2': [23, 45, 11, 30]
    }
    mock_df = pd.DataFrame(mock_data)
    mock_file_path = 'mock_item_map.parquet'
    mock_df.to_parquet(mock_file_path)

    tokenizer = GenRecTokenizer(
        map_file_path=mock_file_path,
        token_level_vocab_sizes=[4, 20, 50, 10000],
        special_vocab_size=10,
        pad_token_id=0,
    )
    # Expected offsets: [10, 10+4, 10+4+20] -> [10, 14, 34]
    print("Tokenizer initialized successfully.")
    print(f"Level offsets: {tokenizer.level_offsets}\n")

    input = [[101, 102, 104, -2, 103], [102, -2, 101]]
    output = tokenizer(input, return_tensors='pt')
    print(f"Input: {input}")
    print(f"Output 'input_ids':\n{output['input_ids']}")

    input = [[10, 19, 57, 999, 11, 27, 79, 999, 12, 29, 64, 999, 0, 0, 0, 0],
             [11, 26, 79, 999, 10, 19, 57, 999, 0, 0, 0, 0, 0, 0, 0, 0]]
    input = torch.tensor(input, dtype=torch.long)
    output, missing_counts = tokenizer.decode(input)
    print(f"Decoded output: {output}")
    print(f"Missing counts: {missing_counts}")

    os.remove(mock_file_path)  # Clean up mock file after testing