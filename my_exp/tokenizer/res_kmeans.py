import numpy as np
import pandas as pd
import faiss
import torch
from tqdm import tqdm

def residual_kmeans_tokenizer(embeddings: np.ndarray, 
                              n_clusters: int = 512, 
                              n_layers: int = 3,
                              output_filename: str = "item_tokens.parquet") -> pd.DataFrame:
    """
    使用残差K-Means对Item Embedding进行分层tokenizer。

    Args:
        embeddings (np.ndarray): 输入的Item Embedding矩阵，形状为 (n_items, n_dims)。
                                 为了faiss的性能，推荐使用float32类型。
        n_clusters (int): 每层聚类的簇数 (token词汇表大小)。
        n_layers (int): 聚类的层数。

    Returns:
        pd.DataFrame: 包含item_id和各层token的DataFrame。
    """
    if embeddings.dtype != np.float32:
        print("Embeddings are not float32. Converting for faiss compatibility...")
        embeddings = embeddings.astype(np.float32)

    num_items, dim = embeddings.shape
    print(f"Starting Residual K-Means for {num_items} items with {dim}-dim embeddings.")
    print(f"Parameters: {n_layers} layers, {n_clusters} clusters per layer.")

    # 初始化
    all_tokens = np.zeros((num_items, n_layers), dtype=np.int32)
    residual_embeddings = np.copy(embeddings)
    codebooks = [] # 存储每一层的质心

    # --- 分层聚类 ---
    for layer in range(n_layers):
        print(f"\n--- Processing Layer {layer + 1}/{n_layers} ---")
        
        # 1. 配置并训练K-Means
        # faiss的Kmeans非常高效，适合大规模数据
        # 使用GPU的话，可以设置 gpu=True
        kmeans = faiss.Kmeans(d=dim, k=n_clusters, niter=20, verbose=True, gpu=False, max_points_per_centroid=4096)
        # kmeans = faiss.Kmeans(d=dim, k=n_clusters, niter=20, verbose=True, gpu=False)
        kmeans.train(residual_embeddings)
        
        # 2. 分配token
        print("Assigning clusters to get tokens...")
        _, labels = kmeans.index.search(residual_embeddings, 1)
        labels = labels.flatten()
        all_tokens[:, layer] = labels
        
        # 3. 保存质心并计算下一层的残差
        codebooks.append(kmeans.centroids)
        print("Calculating residuals for the next layer...")
        residual_embeddings -= kmeans.centroids[labels]

    # --- 组装初始结果 ---
    print("\nAssembling initial results...")
    df = pd.DataFrame({
        'item_id': range(num_items),
        'token_1': all_tokens[:, 0],
        'token_2': all_tokens[:, 1],
        'token_3': all_tokens[:, 2],
    })

    # --- 碰撞处理 ---
    print("Checking for token collisions...")
    # 找出所有重复的 (t1, t2, t3) 组合的行，除了每组的第一个
    duplicates_mask = df.duplicated(subset=['token_1', 'token_2', 'token_3'], keep='first')
    duplicate_indices = df[duplicates_mask].index

    if not duplicate_indices.empty:
        print(f"Found {len(duplicate_indices)} items with duplicate token combinations. Starting reassignment...")
        
        # 为质心构建FAISS索引以便快速、准确地搜索
        print("Building faiss indexes for codebooks for fast searching...")
        # 我们需要 L2 和 L3 的质心索引来重新分配 t2 和 t3
        index_l2 = faiss.IndexFlatL2(dim)
        index_l2.add(codebooks[1])
        index_l3 = faiss.IndexFlatL2(dim)
        index_l3.add(codebooks[2])

        # 创建一个现有token组合的集合，用于快速查找
        existing_tokens_set = {tuple(x) for x in df[['token_1', 'token_2', 'token_3']].itertuples(index=False, name=None)}

        for item_idx in tqdm(duplicate_indices, desc="Reassigning tokens"):
            original_tokens = tuple(df.loc[item_idx, ['token_1', 'token_2', 'token_3']])
            
            # 找到一个新的、唯一的token组合
            found_new_token = False
            
            # 策略：优先修改token_3，然后token_2，最后token_1
            # 计算该item在第三层聚类前的残差
            residual_l2 = embeddings[item_idx] - codebooks[0][original_tokens[0]] - codebooks[1][original_tokens[1]]
            
            # --- 尝试修改 token_3 ---
            # 找到第三层质心中离残差最近的k个
            distances_l3, new_tokens_l3 = index_l3.search(np.expand_dims(residual_l2, 0), n_clusters)
            
            for new_t3 in new_tokens_l3.flatten():
                new_combo = (original_tokens[0], original_tokens[1], new_t3)
                if new_combo not in existing_tokens_set:
                    df.loc[item_idx, 'token_3'] = new_t3
                    existing_tokens_set.add(new_combo)
                    found_new_token = True
                    break
            
            if found_new_token:
                continue

            # --- 如果修改token_3失败，尝试修改token_2 ---
            # (这种情况在实践中非常罕见，但为了代码的鲁棒性我们处理它)
            print(f"Warning: Could not find unique token for item {item_idx} by only changing token_3. Trying to change token_2.")
            residual_l1 = embeddings[item_idx] - codebooks[0][original_tokens[0]]
            distances_l2, new_tokens_l2 = index_l2.search(np.expand_dims(residual_l1, 0), n_clusters)

            for new_t2 in new_tokens_l2.flatten():
                # 对于每个新的token_2，我们需要找到最优的token_3
                new_residual_l2 = residual_l1 - codebooks[1][new_t2]
                _, new_t3_arr = index_l3.search(np.expand_dims(new_residual_l2, 0), 1)
                new_t3 = new_t3_arr.flatten()[0]

                new_combo = (original_tokens[0], new_t2, new_t3)
                if new_combo not in existing_tokens_set:
                    df.loc[item_idx, 'token_2'] = new_t2
                    df.loc[item_idx, 'token_3'] = new_t3
                    existing_tokens_set.add(new_combo)
                    found_new_token = True
                    break
            
            if not found_new_token:
                # 如果连修改 token_2 都失败了，可以抛出错误或采取其他策略
                print(f"FATAL: Could not resolve collision for item {item_idx} even after trying to change token_2. "
                      f"Consider increasing n_clusters or review your data. Item assigned a potentially non-unique token.")

    else:
        print("No token collisions found.")

    # --- 最后处理与保存 ---
    print("\nFinalizing and saving the results...")
    
    # 将token从 0-511 映射到 1-512
    for col in ['token_1', 'token_2', 'token_3']:
        df[col] = df[col] + 1
        
    # 保存为parquet文件
    df.to_parquet(output_filename, index=False)
    
    print(f"Successfully generated and saved item tokens to '{output_filename}'.")
    return df


if __name__ == '__main__':
    model_path = "/home/liqun03/FuxiCTR/my_exp/base/checkpoints/taobao/DIN_Long.model"
    save_path = "/home/liqun03/FuxiCTR/my_datasets/taobao/item_tokens.parquet"
    max_id = 4162024
    # 1. 处理Embedding数据
    model_dict = torch.load(model_path, map_location='cpu')
    item_embeddings = model_dict['embedding_layer.embedding_layer.embedding_layers.target_item.weight'][1:max_id + 1, :]
    item_embeddings = item_embeddings.numpy()

    # 2. 运行tokenizer
    final_tokens_df = residual_kmeans_tokenizer(item_embeddings, n_clusters=1024, n_layers=3)

    # 3. 验证输出
    print("\n--- Verification ---")
    print("First 5 rows of the output DataFrame:")
    print(final_tokens_df.head())
    
    print("\nToken value range check:")
    for col in ['token_1', 'token_2', 'token_3']:
        min_val, max_val = final_tokens_df[col].min(), final_tokens_df[col].max()
        print(f"'{col}' range: [{min_val}, {max_val}]")
        assert 1 <= min_val <= 512
        assert 1 <= max_val <= 512

    print("\nFinal collision check:")
    num_duplicates = final_tokens_df.duplicated(subset=['token_1', 'token_2', 'token_3']).sum()
    print(f"Number of items with duplicate token combinations after reassignment: {num_duplicates}")
    assert num_duplicates == 0, "Collision resolution failed!"

    print("\nItem tokens has been created.")