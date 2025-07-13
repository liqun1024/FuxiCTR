import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

def load_and_process_data(data_path):
    # 加载评分数据
    ratings = pd.read_csv(f'{data_path}/ratings.dat', sep='::', 
                         engine='python', 
                         names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # 将评分转换为二分类标签 (1: rating>3, 0: rating<=3)
    ratings['label'] = (ratings['rating'] > 3).astype(int)
    
    # 按时间戳对每个用户的评分进行排序
    ratings_sorted = ratings.sort_values(['user_id', 'timestamp'])
    
    # 对user_id和movie_id进行编码，使其从0开始连续编号
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    ratings_sorted['user_id'] = user_encoder.fit_transform(ratings_sorted['user_id']) + 1  # 从1开始编号
    ratings_sorted['movie_id'] = movie_encoder.fit_transform(ratings_sorted['movie_id']) + 1  # 从1开始编号
    
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

        # 处理验证集 (历史序列是所有训练数据)
        for _, record in val.iterrows():
            val_data['user_id'].append(record['user_id'])
            val_data['movie_id'].append(record['movie_id'])
            val_data['label'].append(record['label'])
            val_data['history_seq'].append(train['movie_id'].values[-50:])  # 只保留最近的50个
            val_data['history_ts'].append(train['timestamp'].values[-50:])  # 添加对应的timestamp

        # 处理测试集 (历史序列是所有训练数据)
        for _, record in test.iterrows():
            test_data['user_id'].append(record['user_id'])
            test_data['movie_id'].append(record['movie_id'])
            test_data['label'].append(record['label'])
            test_data['history_seq'].append(train['movie_id'].values[-50:])  # 只保留最近的50个
            test_data['history_ts'].append(train['timestamp'].values[-50:])  # 添加对应的timestamp
    
    print(f"总用户数: {total_users}, 过滤用户数: {filtered_users}, 保留用户数: {total_users - filtered_users}")
    
    # 创建用户特征 (这里简单使用用户ID作为特征)
    user_features = np.arange(len(user_encoder.classes_), dtype=np.int32)
    
    # 创建电影特征 (这里简单使用电影ID作为特征)
    movie_features = np.arange(len(movie_encoder.classes_), dtype=np.int32)
    
    # 将数据转换为numpy数组
    def prepare_data(data_dict):
        return {
            'user_id': np.array(data_dict['user_id'], dtype=np.int32),
            'movie_id': np.array(data_dict['movie_id'], dtype=np.int32),
            'label': np.array(data_dict['label'], dtype=np.int32),
            'history_seq': np.array(data_dict['history_seq'], dtype=np.int32),  # 已经是固定长度50
            'history_ts': np.array(data_dict['history_ts'], dtype=np.int64)    # 已经是固定长度50
        }
    
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

def save_as_npz(data, output_path):
    # 保存用户特征
    np.savez(f'{output_path}/user_features.npz', user_id=data['user_features'])
    
    # 保存电影特征
    np.savez(f'{output_path}/movie_features.npz', movie_id=data['movie_features'])
    
    # 保存训练集
    np.savez(f'{output_path}/train.npz', 
             user_id=data['train']['user_id'],
             movie_id=data['train']['movie_id'],
             label=data['train']['label'],
             history_seq=data['train']['history_seq'],
             history_ts=data['train']['history_ts'])

    # 保存验证集
    np.savez(f'{output_path}/valid.npz', 
             user_id=data['val']['user_id'],
             movie_id=data['val']['movie_id'],
             label=data['val']['label'],
             history_seq=data['val']['history_seq'],
             history_ts=data['val']['history_ts'])

    # 保存测试集
    np.savez(f'{output_path}/test.npz', 
             user_id=data['test']['user_id'],
             movie_id=data['test']['movie_id'],
             label=data['test']['label'],
             history_seq=data['test']['history_seq'],
             history_ts=data['test']['history_ts'])

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
    save_as_npz(processed_data, output_path)
    
    print("数据处理完成并已保存为npz格式!")