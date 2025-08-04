import torch

# -- File Paths --
TRAIN_DATA_PATH = "/home/liqun03/FuxiCTR/my_datasets/taobao_sim_zero_based/train_with_similarity.parquet"
VAL_DATA_PATH = "/home/liqun03/FuxiCTR/my_datasets/taobao_sim_zero_based/valid_with_similarity.parquet" 
TEST_DATA_PATH = "/home/liqun03/FuxiCTR/my_datasets/taobao_sim_zero_based/test_with_similarity.parquet"
OUTPUT_DIR = "/home/liqun03/FuxiCTR/checkpoints/GenRec"

# -- Model Configuration --
TOKEN_LEVEL_VOCAB_SIZES = [1024, 1024, 1024, 10000]
SPECIAL_VOCAB_SIZE = 10
MODEL_DIM = 128
NUM_LAYERS = 2
LOSS_TEMPERATURE = 1.0

# -- Tokenizer Configuration --
PAD_TOKEN_ID = 0
HIST_TARGER_TOKEN_ID = 1
DECODER_START_TOKEN_ID = 2
TOKENIZER_MAP_FILE = "/home/liqun03/FuxiCTR/checkpoints/item_tokens_zero_based.parquet"

# -- Training Configuration --
DEVICE = "cuda"
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
LOG_INTERVAL = 10 # Log training loss every N steps