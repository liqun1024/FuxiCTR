import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_and_process_data(data_path):
    # 加载评分数据
    ratings = pd.read_csv(f'{data_path}/ratings.dat', sep='::', 
                         engine='python', 
                         names=['user_id', 'movie_id', 'rating', 'timestamp'],
                         encoding='latin-1')

    # 将评分转换为二分类标签 (1: rating>3, 0: rating<=3)
    ratings['label'] = (ratings['rating'] > 3).astype(int)
    
    # 按时间戳对每个用户的评分进行排序
    ratings_sorted = ratings.sort_values(['user_id', 'timestamp'])
    
    # 对user_id和movie_id进行编码，使其从0开始连续编号
    # 先读取用户侧信息
    users = pd.read_csv(f'{data_path}/users.dat', sep='::', engine='python',
                        names=['user_id', 'gender', 'age', 'occupation', 'zip'], encoding='latin-1')

    # 重新编码user_id，确保所有user_id都被编码
    all_user_ids = pd.concat([ratings['user_id'], users['user_id']]).unique()
    user_encoder = LabelEncoder()
    user_encoder.fit(all_user_ids)
    ratings_sorted['user_id'] = user_encoder.transform(ratings_sorted['user_id']) + 1  # 从1开始编号
    users['user_id'] = user_encoder.transform(users['user_id']) + 1  # 与ratings编码对齐

    # 读取电影侧信息
    movies = pd.read_csv(f'{data_path}/movies.dat', sep='::', engine='python',
                         names=['movie_id', 'title', 'genres'], encoding='latin-1')

    # 重新编码movie_id，确保所有movie_id都被编码
    all_movie_ids = pd.concat([ratings['movie_id'], movies['movie_id']]).unique()
    movie_encoder = LabelEncoder()
    movie_encoder.fit(all_movie_ids)
    ratings_sorted['movie_id'] = movie_encoder.transform(ratings_sorted['movie_id']) + 1  # 从1开始编号
    movies['movie_id'] = movie_encoder.transform(movies['movie_id']) + 1  # 与ratings编码对齐
    
    # 按用户分组
    user_groups = ratings_sorted.groupby('user_id')
    
    # 初始化各数据集
    train_data = {'user_id': [], 'movie_id': [], 'label': [], 'history_seq': [], 'history_ts': []}
    val_data = {'user_id': [], 'movie_id': [], 'label': [], 'history_seq': [], 'history_ts': []}
    test_data = {'user_id': [], 'movie_id': [], 'label': [], 'history_seq': [], 'history_ts': []}
    
    # 统计过滤的用户
    total_users = len(user_groups)
    filtered_users = 0
    
    for user_id, group in user_groups:
        n_ratings = len(group)

        # 过滤掉评分少于60的用户
        if n_ratings < 60:
            filtered_users += 1
            continue
        
        # 对超过50的部分才可以划分
        n_ratings = len(group) - 50

        # 计算划分点
        train_end = 50 + int(n_ratings * 0.7)
        val_end = train_end + int(n_ratings * 0.2)
        
        # 划分数据
        train = group.iloc[:train_end]
        val = group.iloc[train_end:val_end]
        test = group.iloc[val_end:]
        
        # 处理训练集
        for i in range(50, len(train)):  # 从第50个开始，保证历史序列至少有50个
            record = train.iloc[i]
            history = train.iloc[:i]['movie_id'].values
            history_ts = train.iloc[:i]['timestamp'].values
            train_data['user_id'].append(record['user_id'])
            train_data['movie_id'].append(record['movie_id'])
            train_data['label'].append(record['label'])
            train_data['history_seq'].append(history[-50:])  # 只保留最近的50个
            train_data['history_ts'].append(history_ts[-50:])  # 添加对应的timestamp

        # 处理验证集 (历史序列为滑动窗口)
        for i, record in enumerate(val.itertuples(index=False)):
            idx = train_end + i
            full_group = group.iloc[:idx]
            val_data['user_id'].append(record.user_id)
            val_data['movie_id'].append(record.movie_id)
            val_data['label'].append(record.label)
            val_data['history_seq'].append(full_group['movie_id'].values[-50:])  # 滑动窗口
            val_data['history_ts'].append(full_group['timestamp'].values[-50:])  # 滑动窗口

        # 处理测试集 (历史序列为滑动窗口)
        for i, record in enumerate(test.itertuples(index=False)):
            idx = val_end + i
            full_group = group.iloc[:idx]
            test_data['user_id'].append(record.user_id)
            test_data['movie_id'].append(record.movie_id)
            test_data['label'].append(record.label)
            test_data['history_seq'].append(full_group['movie_id'].values[-50:])  # 滑动窗口
            test_data['history_ts'].append(full_group['timestamp'].values[-50:])  # 滑动窗口
    
    print(f"总用户数: {total_users}, 过滤用户数: {filtered_users}, 保留用户数: {total_users - filtered_users}")
    
    # 编码性别
    users['gender'] = users['gender'].map({'M': 0, 'F': 1}).astype(np.int8)
    # 年龄和职业本身就是数字，无需再编码
    # 只保留 user_id, gender, age, occupation
    user_features = users[['user_id', 'gender', 'age', 'occupation']].copy()

    # 处理Genres为多热编码
    all_genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                  'Sci-Fi', 'Thriller', 'War', 'Western']
    for genre in all_genres:
        movies[genre] = movies['genres'].apply(lambda x: int(genre in x.split('|')))
    # 只保留 movie_id 及多热编码
    movie_features = movies[['movie_id'] + all_genres].copy()

    # 将数据转换为DataFrame
    def prepare_data(data_dict):
        return pd.DataFrame({
            'user_id': data_dict['user_id'],
            'movie_id': data_dict['movie_id'],
            'label': data_dict['label'],
            'history_seq': data_dict['history_seq'],
            'history_ts': data_dict['history_ts']
        })

    train_data = prepare_data(train_data)
    val_data = prepare_data(val_data)
    test_data = prepare_data(test_data)

    return {
        'user_features': user_features,
        'movie_features': movie_features,
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

def save_as_parquet(data, output_path):
    # 保存用户特征
    data['user_features'].to_parquet(f'{output_path}/user_features.parquet', index=False)

    # 保存电影特征
    data['movie_features'].to_parquet(f'{output_path}/movie_features.parquet', index=False)

    # 保存训练集
    data['train'].to_parquet(f'{output_path}/train.parquet', index=False)

    # 保存验证集
    data['val'].to_parquet(f'{output_path}/valid.parquet', index=False)

    # 保存测试集
    data['test'].to_parquet(f'{output_path}/test.parquet', index=False)

if __name__ == '__main__':
    # 设置路径
    data_path = 'my_datasets/raw/ml-1m'  # 替换为你的MovieLens-1M数据集路径
    output_path = 'my_datasets/ml-1m'      # 输出目录
    
    # 创建输出目录
    import os
    os.makedirs(output_path, exist_ok=True)
    
    # 处理数据
    processed_data = load_and_process_data(data_path)

    # 保存数据
    save_as_parquet(processed_data, output_path)

    print("数据处理完成并已保存为parquet格式!")