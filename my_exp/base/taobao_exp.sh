#!/bin/bash

# 实验ID（expid），保持不变
EXPID="SIM"

# 学习率列表（网格搜索范围）
LRS=(0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)

# 循环执行训练脚本
for lr in "${LRS[@]}"
do
  echo "=================================================================="
  echo "Running experiment with learning_rate = $lr"
  echo "=================================================================="
  
  python taobao_run_expid.py --expid $EXPID --learning_rate $lr
  
done

echo "Grid search completed!"