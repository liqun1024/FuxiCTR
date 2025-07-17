import pickle as pkl
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import os

MAX_LEN_ITEM = 200

def to_df(file_name):
    df = pd.read_csv(file_name, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    return df

def remap(df):
    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(1, item_len + 1)))
    df['iid'] = df['iid'].map(lambda x: item_map[x])

    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(1, user_len + 1)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(1, cate_len + 1)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(1, btag_len + 1)))
    df['btag'] = df['btag'].map(lambda x: btag_map[x])

    print(f"ID重映射完成: item_len={item_len}, user_len={user_len}, cate_len={cate_len}, btag_len={btag_len}")
    return df, item_len


def gen_user_item_group(df):
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')

    print("group completed")
    return user_df, item_df


def analyze(user_df, item_df, item_cnt):
    # get each user's last touch point time
    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])
    print("get user last touch time completed")

    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time_train = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)]

    cnt = 0
    train_item_hist_lens = []  # 新增：用于收集train序列长度
    for uid, hist in tqdm(user_df):
        cnt += 1
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        btag_hist = hist['btag'].tolist()
        target_item_time = hist['time'].tolist()[-1]

        if target_item_time <= split_time_train:
            # 统计train序列长度
            train_item_hist_lens.append(len(item_hist))
    # 绘制分布图
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(train_item_hist_lens, bins=50)
    plt.xlabel('item_hist length')
    plt.ylabel('count')
    plt.title('Train序列中item_hist长度分布')
    plt.savefig('train_item_hist_length_distribution.png')


def main():
    # 设置路径
    data_path = 'my_datasets/raw/taobao/UserBehavior.csv'  # 替换为你的MovieLens-1M数据集路径

    df = to_df(data_path)
    df, item_cnt = remap(df)

    user_df, item_df = gen_user_item_group(df)
    analyze(user_df, item_df, item_cnt)

if __name__ == '__main__':
    main()