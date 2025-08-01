# config.py
import torch

# -- File Paths --
TRAIN_DATA_PATH = "data/dummy_data.parquet"
EVAL_DATA_PATH = "data/dummy_data.parquet" # Use the same for simplicity, change if you have a separate eval set
OUTPUT_DIR = "checkpoints"

# -- Model Configuration --
TOKEN_LEVEL_VOCAB_SIZES = [10, 10, 10, 100]
SPECIAL_VOCAB_SIZE = 10
MODEL_DIM = 128
NUM_LAYERS = 2
DECODER_START_TOKEN_ID = 1
LOSS_TEMPERATURE = 1.0

# -- Tokenizer Configuration --
PAD_TOKEN_ID = -100

# -- Training Configuration --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
LOG_INTERVAL = 10 # Log training loss every N steps