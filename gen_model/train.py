import torch
from torch.utils.data import DataLoader
from functools import partial
import pandas as pd
import numpy as np
import os

import config
from GenRecMultiHead import GenRecMultiHead
from GenRecTokenizer import GenRecTokenizer
from dataset import TaobaoSimDataLoader
from trainer import Trainer
from transformers import T5Config



def main():
    device = torch.device(config.DEVICE)

    train_loader, valid_loader = TaobaoSimDataLoader(
        stage='train', 
        train_data=config.TRAIN_DATA_PATH, 
        valid_data=config.VAL_DATA_PATH,
        map_file_path=config.TOKENIZER_MAP_FILE,  # Assuming no map file is needed for this example
        token_level_vocab_sizes=config.TOKEN_LEVEL_VOCAB_SIZES,
        special_vocab_size=config.SPECIAL_VOCAB_SIZE,
        batch_size=config.BATCH_SIZE
    ).make_iterator()

    model_config = T5Config(
        vocab_size=1, 
        d_model=config.MODEL_DIM,
        d_kv=config.MODEL_DIM // 2,
        d_ff=config.MODEL_DIM * 2,
        num_layers=config.NUM_LAYERS,
        decoder_start_token_id=config.DECODER_START_TOKEN_ID,
        token_levels=len(config.TOKEN_LEVEL_VOCAB_SIZES) 
    )

    model = GenRecMultiHead(
        config=model_config,
        token_level_vocab_sizes=config.TOKEN_LEVEL_VOCAB_SIZES,
        special_vocab_size=config.SPECIAL_VOCAB_SIZE,
        loss_temperature=config.LOSS_TEMPERATURE
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=valid_loader,
        device=device
    )
    trainer.train()

if __name__ == '__main__':
    main()