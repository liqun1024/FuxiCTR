# =========================================================================
# Copyright (C) 2025. FuxiCTR Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import torch
import logging


class ParquetDataset(Dataset):
    def __init__(self, data_path):
        self.column_index = dict()
        self.darray = self.load_data(data_path)
        
    def __getitem__(self, index):
        return self.darray[index, :]
    
    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path):
        df = pd.read_parquet(data_path)
        data_arrays = []
        idx = 0
        for col in df.columns:
            if df[col].dtype == "object":
                array = np.array(df[col].to_list())
                seq_len = array.shape[1]
                self.column_index[col] = [i + idx for i in range(seq_len)]
                idx += seq_len
            else:
                array = df[col].to_numpy()
                self.column_index[col] = idx
                idx += 1
            data_arrays.append(array)
        return np.column_stack(data_arrays)


class SeqDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False, max_len=None,
                 num_workers=1, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(data_path)
        column_index = self.dataset.column_index
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=BatchCollator(feature_map, column_index, max_len)
        )
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches


class BatchCollator(object):
    def __init__(self, feature_map, column_index, max_len):
        self.feature_map = feature_map
        self.column_index = column_index
        self.max_len = max_len

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = set(list(self.feature_map.features.keys()) + self.feature_map.labels)
        batch_dict = dict()
        for col, idx in self.column_index.items():
            if col in all_cols:
                batch_dict[col] = batch_tensor[:, idx]

        item_dict = {}
        item_dict["item_hist"] = batch_dict["item_hist"].numpy()
        batch_dict.pop("item_hist", None)
        if self.max_len:
            item_dict["item_hist"] = item_dict["item_hist"][:, -self.max_len:]
        
        mask = (torch.from_numpy(item_dict["item_hist"]) > 0).float() # zeros for masked positions

        item_dict["item_hist"] = torch.from_numpy(item_dict["item_hist"])

        return batch_dict, item_dict, mask[:, :-1]


class TaobaoDataLoader(object):
    def __init__(self, feature_map, stage="both", 
                 train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, **kwargs):
        logging.info("Loading datasets...")
        train_gen = None
        valid_gen = None
        test_gen = None
        self.stage = stage
        if stage in ["both", "train"]:
            train_gen = SeqDataLoader(feature_map, train_data, split="train", batch_size=batch_size,
                                   shuffle=shuffle, **kwargs)
            logging.info(
                "Train samples: total/{:d}, blocks/{:d}"
                .format(train_gen.num_samples, train_gen.num_blocks)
            )     
            if valid_data:
                valid_gen = SeqDataLoader(feature_map, valid_data, split="valid",
                                       batch_size=batch_size, shuffle=False, **kwargs)
                logging.info(
                    "Validation samples: total/{:d}, blocks/{:d}"
                    .format(valid_gen.num_samples, valid_gen.num_blocks)
                )

        if stage in ["both", "test"]:
            if test_data:
                test_gen = SeqDataLoader(feature_map, test_data, split="test", batch_size=batch_size,
                                      shuffle=False, **kwargs)
                logging.info(
                    "Test samples: total/{:d}, blocks/{:d}"
                    .format(test_gen.num_samples, test_gen.num_blocks)
                )
        self.train_gen, self.valid_gen, self.test_gen = train_gen, valid_gen, test_gen

    def make_iterator(self):
        if self.stage == "train":
            logging.info("Loading train and validation data done.")
            return self.train_gen, self.valid_gen
        elif self.stage == "test":
            logging.info("Loading test data done.")
            return self.test_gen
        else:
            logging.info("Loading data done.")
            return self.train_gen, self.valid_gen, self.test_gen