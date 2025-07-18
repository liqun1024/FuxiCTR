import pandas as pd

path = 'my_datasets/taobao'
df_train = pd.read_parquet(f'{path}/train.parquet')
df_valid = pd.read_parquet(f'{path}/valid.parquet')
df_test = pd.read_parquet(f'{path}/test.parquet')
df = pd.concat([df_train, df_valid, df_test])

# Step 1: Extract all unique item-category pairs
all_items = []
all_cates = []

# target_item
all_items.extend(df['target_item'].tolist())
all_cates.extend(df['target_item_cate'].tolist())

# item_hist
item_hist_flat = [item for sublist in df['item_hist'] for item in sublist]
cate_hist_flat = [cate for sublist in df['cate_hist'] for cate in sublist]
all_items.extend(item_hist_flat)
all_cates.extend(cate_hist_flat)

# neg_item_hist
neg_item_hist_flat = [item for sublist in df['neg_item_hist'] for item in sublist]
neg_cate_hist_flat = [cate for sublist in df['neg_cate_hist'] for cate in sublist]
all_items.extend(neg_item_hist_flat)
all_cates.extend(neg_cate_hist_flat)

item_cate_df = pd.DataFrame({'item': all_items, 'token_level_1': all_cates}).drop_duplicates().sort_values(by=['token_level_1', 'item']).reset_index(drop=True)

# Step 2: Create the two-level token mapping
item_cate_df['token_level_2'] = item_cate_df.groupby('token_level_1').cumcount()

# Step 3: Save only item, token_level_1, token_level_2 columns to parquet
item_cate_df[['item', 'token_level_1', 'token_level_2']].to_parquet(f'{path}/tokenized_items.parquet')

print("Tokenized item-cate DataFrame:")
print(item_cate_df[['item', 'token_level_1', 'token_level_2']])
print("\nTokenized data has been saved to 'tokenized_items.parquet'")