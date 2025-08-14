序列的数据格式：\
demo/exp6: \
是npz格式，即多个npy打包的内容。 \
对于特征是seq的部分，二维dict，其他是一维dict。

# 0813
RL使用评估集？ 模型需要交替迭代？

---
# GenRec
注意，之前组织数据集的时候，按照0是padding，大于1开始才是item。
从生成式模型开始，固定使用item从0开始编号，同时对于输入的数据，不固定具体的长度，使用tokenizer实现padding。
SEP_TOKEN: 0-padding, 1-history和target之间的间隔token, 2-decoder的start token

计算NDCG的时候，对于重复样本只计算排名靠前的次数（和labe次数一致）
Recall@20: 0.3831
NDCG@20: 0.4855

---
# SASRec
~~最终训练：~~
~~Step 1301: Train Loss = 0.0072, Val Loss = 0.1702, NDCG@1 = 0.6577, NDCG@5 = 0.7560~~
~~上述因为最后一个item（target item）被重复两次，需要重新训练。~~

Step 1301: Train Loss = 0.0073, Val Loss = 0.1797, NDCG@1 = 0.6573, NDCG@5 = 0.7579
Model saved. (NDCG1 = 0.6573)
Test Loss: 0.1855
---
# taobao
**！！！注意！！！**：由于LongCTR的模型中，默认item侧的序列中最后一个是target，因此我们处理的数据集的item_hist的最后一个是target，其他模型处理的时候需要删除。
item编号: 1-4162024
使用DIN训练的Embedding由于没有正负样本的对比loss，因此相似性更高，在进行tokenizer的时候几乎存在40%的item撞id（设置codesize=1024，layer=3，同时强制均匀聚类max_points_per_centroid也不可以）
改为使用SASRec获得embedding。设置codesize=512，并均匀聚类，有1/7的冲突，仍然不够好。

最终选择设置1024*3的码本，并且设置max_points_per_centroid=4096，冲突为675730个，均在最后一层重置解决。

---
# MovieLens-1M数据处理
使用ml-1m数据，将其处理成npz格式数据，保存user item以及cross的特征。\
其中，首先过滤掉用户交互不足60的，剩余的，在超过50的部分划分7:2:1作为train val和test。
```
(FuxiCTR) lucio@Lucio-Mac FuxiCTR % python my_exp/data_process/movielens-1m.py
总用户数: 6040, 过滤用户数: 2102, 保留用户数: 3938

>>> np.load("train.npz")
NpzFile 'train.npz' with keys: user_id, movie_id, label, history_seq, history_ts
```
train valid test分别有507186，143849，76118条数据。

**Note**:潜在的问题是，在测试集，用户序列不是最新的（即测试集的部分没有被训练，作为流式数据不知道会不会有问题。
这样过滤是否存在穿越问题？ 即按照每条用户自己的时间，但是训练集每条用户的截止时间不相同。
