# generate.py
import torch

# Import local modules
import config
from model import GenRecMultiHead
from dataset import YourTokenizer # We need the tokenizer for preparing input
from transformers import T5Config

def main():
    # 1. Setup Device
    device = torch.device(config.DEVICE)
    
    # 2. Initialize Model
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
        special_vocab_size=config.SPECIAL_VOCAB_SIZE
    )

    # 3. Load fine-tuned weights
    # Make sure to change the path to your best checkpoint
    checkpoint_path = "checkpoints/model_epoch_10.pt" 
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}. Please run train.py first.")
        return
        
    model.to(device)
    model.eval() # Set model to evaluation mode

    # 4. Prepare Input Data
    # This is an example input. You would replace this with your actual data.
    tokenizer = YourTokenizer(pad_token_id=config.PAD_TOKEN_ID)
    input_sequences = [
        [13, 25, 37, 17, 25, 34], 
        [13, 26, 32, 15, 24, 32, 11, 24, 32]
    ]
    
    # Use the tokenizer to prepare the batch
    inputs = tokenizer(input_sequences, max_length=config.MAX_INPUT_LENGTH)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print("\nInput IDs shape:", input_ids.shape)
    print("Input IDs:", input_ids)

    # 5. Generate Sequence
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.MAX_TARGET_LENGTH
        )
    
    print("\nGenerated IDs shape:", generated_ids.shape)
    print("Generated IDs:\n", generated_ids)

if __name__ == '__main__':
    main()