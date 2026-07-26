# SSA优化严格封闭50窗口孪生光储备池

## 优化边界

- 光储备池、50维状态缓存、mask、输入增益、虚拟节点和延迟参数全部冻结。
- SSA只优化孪生模型独有参数：`gap、K、β、p、M`。
- 特征固定为50维`h_i-h_j`，没有使用此前表现较差的100维特征。
- 每个SSA种群都加入原始孪生基线，防止搜索结果无意中弱于已知起点。
- SSA搜索只读取训练池和验证集；最佳配置冻结后才加载测试集。

## 参数定义

- `gap`：目标窗口与参考窗口的最小时间间隔。
- `K`：每个目标使用的历史参考数量。
- `β`：绝对水平距离权重；形状距离权重为`1-β`。
- `p`：参考聚合权重`1/(distance+ε)^p`中的指数。
- `M`：每个delta区间最多选择的训练样本对数量。

## SSA设置与每次运行结果

| seed | gap_months | k_references | level_weight | distance_power | max_pairs_per_bin | shape_weight | fitness | val_mae | val_rmse | validation_block_rmse_std | selected_alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 12 | 8 | 0.000000 | 0.604000 | 2 | 1.000000 | 0.703566 | 0.511686 | 0.688381 | 0.151850 | 1.000000 |
| 43 | 12 | 8 | 0.001000 | 0.835000 | 2 | 0.999000 | 0.705336 | 0.512356 | 0.690146 | 0.151904 | 1.000000 |
| 44 | 12 | 10 | 0.006000 | 0.504000 | 2 | 0.994000 | 0.709920 | 0.514714 | 0.693995 | 0.159249 | 1.000000 |

共实际评估了 **336** 组不重复参数。

## 验证集冻结的最佳参数

```text
gap = 12
K = 8
β = 0.000
形状权重 = 1.000
p = 0.604
M = 2
```

## 最终比较

| model | optimized_by_ssa | val_mae | val_rmse | test_mae | test_rmse |
| --- | --- | --- | --- | --- | --- |
| ordinary_optical_reservoir_closed50 | False | 0.582056 | 0.781584 | 0.910261 | 0.988836 |
| siamese_closed50_baseline | False | 0.516337 | 0.696738 | 0.799978 | 0.886371 |
| ssa_optimized_siamese_closed50 | True | 0.511686 | 0.688381 | 0.804664 | 0.888942 |

相对原始孪生基线，验证RMSE下降 **1.20%**。
测试RMSE上升 **0.29%**。
相对单光储备池，SSA孪生模型的测试RMSE仍下降 **10.10%**。

**结论：SSA优化后测试RMSE没有低于孪生基线，不能宣称取得改善。**

当前测试区间此前已经被查看，因此仍属于探索性复分析。正式结论应冻结本次参数后，使用新的未见时间区间检验。

关键文件：

- `tables/ssa_all_unique_evaluations.csv`：所有不重复候选参数及验证结果。
- `tables/ssa_iteration_history.csv`：不同随机种子的收敛过程。
- `tables/ssa_model_comparison.csv`：单光、孪生基线和SSA孪生的对比。
- `tables/ssa_test_prediction_comparison.csv`：47个测试月份逐月结果。
- `figures/ssa_validation_convergence.png`：SSA收敛曲线。
- `figures/ssa_test_prediction_comparison.png`：最终预测曲线。
- `experiment_manifest.json`：搜索边界和防泄漏记录。
