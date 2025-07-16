import pickle as pkl
import pandas as pd
import random
import numpy as np

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


def gen_dataset(user_df, item_df, item_cnt):
    train_rows, val_rows, test_rows = [], [], []

    # get each user's last touch point time
    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])
    print("get user last touch time completed")

    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time_train = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)]
    split_time_val = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.9)]

    cnt = 0
    for uid, hist in user_df:
        cnt += 1
        print(cnt)
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        btag_hist = hist['btag'].tolist()
        target_item_time = hist['time'].tolist()[-1]

        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        label = 1

        # neg sampling
        neg = random.randint(0, 1)
        if neg == 1:
            label = 0
            while target_item == item_hist[-1]:
                target_item = random.randint(1, item_cnt)
                target_item_cate = item_df.get_group(target_item)['cid'].tolist()[0]


        # the item history part of the sample
        item_part = []
        for i in range(len(item_hist) - 1):
            item_part.append([uid, item_hist[i], cate_hist[i]])
        item_part.append([uid, target_item, target_item_cate])
        # item_part_len = min(len(item_part), MAX_LEN_ITEM)

        # choose the item side information: which user has clicked the target item
        # padding history with 0
        if len(item_part) <= MAX_LEN_ITEM:
            item_part_pad =  [[0] * 3] * (MAX_LEN_ITEM - len(item_part)) + item_part
        else:
            item_part_pad = item_part[len(item_part) - MAX_LEN_ITEM:len(item_part)]
        
        # gen sample
        # sample = (label, item_part_pad, item_part_len, user_part_pad, user_part_len)

        padded_item_hist = [row[1] for row in item_part_pad]
        padded_cate_hist = [row[2] for row in item_part_pad]
        sample_row = {
            'uid': uid,
            'target_item': target_item,
            'target_item_cate': target_item_cate,
            'label': label,
            'item_hist': padded_item_hist,
            'cate_hist': padded_cate_hist
        }

        if target_item_time <= split_time_train:
            train_rows.append(sample_row)
        elif target_item_time <= split_time_val:
            val_rows.append(sample_row)
        else:
            test_rows.append(sample_row)

    print("数据集生成完毕。")
    return train_rows, val_rows, test_rows


def add_neg_samples(train_data, val_data, test_data):
    """
    为每个样本生成一个负采样历史序列 (negative history)。
    此函数保留了原始代码 `produce_neg_item_hist_with_cate` 的核心逻辑。
    """
    print("开始为样本添加负采样历史序列...")
    all_data = train_data + val_data + test_data
    if not all_data:
        print("无数据可处理。")
        return

    # 1. 收集所有出现过的 (item, cate) 对
    item_cate_dict = {}
    for row in all_data:
        hist_pairs = zip(row['item_hist'], row['cate_hist'])
        for pair in hist_pairs:
            item_cate_dict.setdefault(str(pair), 0)
    
    # 移除由padding产生的 (0, 0) 对
    if "('0', '0')" in item_cate_dict:
        del item_cate_dict["(0, 0)"]
    
    unique_item_cate_pairs = list(item_cate_dict.keys())
    if not unique_item_cate_pairs:
        print("警告: 找不到唯一的 (item, cate) 对，无法生成负采样历史。")
        return
        
    # 2. 一次性生成所有需要的随机负样本候选项
    sample_count = len(all_data)
    hist_len = len(all_data[0]['item_hist'])
    # `hist_len + 20` 是为了增加找到不重复负样本的概率，保留原逻辑
    print("正在生成随机负样本候选项...")
    neg_array = np.random.choice(unique_item_cate_pairs, (sample_count, hist_len + 20))
    print("负样本候选项生成完毕。")

    # 3. 遍历所有数据，添加负采样历史
    current_sample_idx = 0
    for dataset in [train_data, val_data, test_data]:
        for row in dataset:
            pos_hist_set = set(zip(row['item_hist'], row['cate_hist']))
            neg_choices_for_sample = neg_array[current_sample_idx]
            current_sample_idx += 1

            neg_hist_list = []
            for item_str in neg_choices_for_sample:
                # 使用 ast.literal_eval 会更安全，但为保持一致使用 eval
                item_tuple = eval(item_str)
                if item_tuple not in pos_hist_set:
                    neg_hist_list.append(item_tuple)
                if len(neg_hist_list) == hist_len:
                    break
            
            # 如果负样本不足，用 (0, 0) 填充
            while len(neg_hist_list) < hist_len:
                neg_hist_list.append((0, 0))

            neg_item_list, neg_cate_list = zip(*neg_hist_list)
            
            row['neg_item_hist'] = list(neg_item_list)
            row['neg_cate_hist'] = list(neg_cate_list)

    print("负采样历史序列添加完成。")

def main():

    # 设置路径
    data_path = 'my_datasets/raw/taobao/UserBehavior.csv'  # 替换为你的MovieLens-1M数据集路径
    output_path = 'my_datasets/taobao'      # 输出目录


    df = to_df(data_path)
    df, item_cnt = remap(df)

    user_df, item_df = gen_user_item_group(df)
    train_data, val_data, test_data = gen_dataset(user_df, item_df, item_cnt)
    add_neg_samples(train_data, val_data, test_data)

    # 转换为DataFrame
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    # 保存为Parquet格式
    print(f"正在保存训练集到 {output_path}/train.parquet...")
    train_df.to_parquet(f"{output_path}/train.parquet", index=False)

    print(f"正在保存验证集到 {output_path}/val.parquet...")
    val_df.to_parquet(f"{output_path}/val.parquet", index=False)

    print(f"正在保存测试集到 {output_path}/test.parquet...")
    test_df.to_parquet(f"{output_path}/test.parquet", index=False)


if __name__ == '__main__':
    main()