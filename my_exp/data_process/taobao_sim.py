import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


def find_top_k_similar_items(row, embeddings, k=20):
    try:
        target_item_id = row['target_item']
        history_item_ids = row['item_hist']

        target_emb = embeddings[target_item_id]
        hist_embs = embeddings[history_item_ids]

        sims = cosine_similarity(target_emb.reshape(1, -1), hist_embs)[0]

        top_k_indices = np.argsort(sims)[::-1][:k]
        history_item_ids_arr = np.array(history_item_ids)
        top_k_items = history_item_ids_arr[top_k_indices].tolist()
        top_k_sims = sims[top_k_indices].tolist()

        return pd.Series([top_k_items, top_k_sims], index=['top_20_items', 'top_20_sims'])

    except IndexError as e:
        print(f"Error processing row: {e}. target_item: {row['target_item']}. Returning empty lists.")
        return pd.Series([[], []], index=['top_20_items', 'top_20_sims'])



if __name__ == "__main__":
    INPUT_DIR = "/home/liqun03/FuxiCTR/my_datasets/taobao"
    OUTPUT_DIR = "/home/liqun03/FuxiCTR/my_datasets/taobao_sim"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for split in ['train', 'valid', 'test']:
        df = pd.read_parquet(os.path.join(INPUT_DIR, f'{split}.parquet'))
        item_embeddings = np.load(os.path.join(INPUT_DIR, 'item_embeddings.npy'))

        results_df = df.apply(
            find_top_k_similar_items,
            axis=1,
            embeddings=item_embeddings,
            k=20
        )

        final_df = pd.concat([df, results_df], axis=1)

        output_filename = f'{split}_with_similarity.parquet'
        final_df.to_parquet(os.path.join(OUTPUT_DIR, output_filename), index=False)

        print(final_df.head())