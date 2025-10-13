import os
import pandas as pd

def data_filter(dir, file):
    # 读取 parquet 文件（确保 cate_hist 被正确解析为 list）
    df = pd.read_parquet(os.path.join(dir, file))

    # 定义函数：计算非零元素个数
    def count_nonzero_in_list(lst):
        return sum(1 for x in lst if x != 0)

    # 应用函数，创建新列或直接过滤
    df_filtered = df[df['item_hist'].apply(count_nonzero_in_list) > 100]

    # 保存结果
    df_filtered.to_parquet(os.path.join(dir, f"filter_{file}"))

dir = "/home/liqun03/FuxiCTR/my_datasets/taobao"
file = ["train.parquet", "valid.parquet", "test.parquet"]
for f in file:
    data_filter(dir, f)