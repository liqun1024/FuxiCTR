import torch
import numpy as np
from tqdm import tqdm
import config
from GenRecMultiHead import GenRecMultiHead
from dataset import TaobaoSimDataLoader
from transformers import T5Config

def recall_at_k(true_items_list, pred_items_list, k):
    """
    Calculates Macro-averaged Recall@k.
    """
    total_recall = 0.0
    num_users = 0
    for true, pred in zip(true_items_list, pred_items_list):
        true_set = {tuple(item) for item in true}
        if not true_set:
            continue
        
        num_users += 1
        pred_set = {tuple(item) for item in pred[:k]}
        hits = len(true_set & pred_set)
        total_recall += hits / len(true_set)
        
    return total_recall / num_users if num_users > 0 else 0

def ndcg_at_k(true_items_list, pred_items_list, k):
    """
    Calculates NDCG@k.
    """
    total_ndcg = 0.0
    num_users = len(true_items_list)
    for true, pred in zip(true_items_list, pred_items_list):
        true_set = [tuple(item) for item in true]
        available_true_set = list(true_set)
        dcg = 0.0
        for j, p in enumerate(pred[:k]):
            if tuple(p) in available_true_set:
                dcg += 1.0 / np.log2(j + 2)
                available_true_set.remove(tuple(p))
        
        idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(true_set), k)))
        
        if idcg > 0:
            total_ndcg += dcg / idcg
    return total_ndcg / num_users if num_users > 0 else 0

def evaluate_generation():
    device = torch.device(config.DEVICE)

    _, valid_loader = TaobaoSimDataLoader(
        stage='train', 
        train_data=config.TRAIN_DATA_PATH, 
        valid_data=config.VAL_DATA_PATH,
        map_file_path=config.TOKENIZER_MAP_FILE,
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
    ).to(device)

    model_ckpt = torch.load("/home/liqun03/FuxiCTR/checkpoints/GenRec/model_epoch_5.pth", map_location=device)
    model.load_state_dict(model_ckpt["model_state_dict"])

    model.eval()

    all_pred_items = []
    all_true_items = []

    token_levels = len(config.TOKEN_LEVEL_VOCAB_SIZES)
    item_token_length = token_levels
    num_tokens_to_compare = token_levels - 1  # Exclude the last token for comparison

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Evaluating Generation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=20 * item_token_length
            )

            batch_size = labels.size(0)

            pred_items = generated_ids.view(batch_size, 20, item_token_length)
            pred_items_to_compare = pred_items[:, :, :num_tokens_to_compare].cpu().numpy()

            true_items = labels.view(batch_size, -1, item_token_length)
            valid_item_mask = torch.all(true_items != model.pad_token_id, dim=2)
            true_items_to_compare = true_items[:, :, :num_tokens_to_compare].cpu().numpy()

            for i in range(batch_size):
                sample_true_items = true_items_to_compare[i][valid_item_mask[i].cpu().numpy()]
                if len(sample_true_items) > 0:
                    all_true_items.append(sample_true_items)
                    all_pred_items.append(pred_items_to_compare[i])

    # Calculate and print metrics
    recall_20 = recall_at_k(all_true_items, all_pred_items, k=20)
    ndcg_20 = ndcg_at_k(all_true_items, all_pred_items, k=20)

    print(f"Recall@20: {recall_20:.4f}")
    print(f"NDCG@20: {ndcg_20:.4f}")

if __name__ == '__main__':
    evaluate_generation()
