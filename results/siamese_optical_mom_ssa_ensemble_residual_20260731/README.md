# 环比严格封闭50：SSA稳定集成与残差式孪生修正

本实验冻结光储备池及其50维状态，只优化孪生分支的组合方式。
三个独立SSA种子的验证集最优配置分别重训后等权平均，再把孪生集成与单光预测之差作为受约束修正量。

## 防泄漏边界

- 模型只访问固定的50个训练窗口；验证集45个目标、测试集47个目标。
- 验证和测试所用的历史参考全部来自训练50窗口。
- SSA成员、集成规则和残差强度只依据训练集与验证集确定。
- 测试集在成员及残差强度冻结后才进入本流程；但该历史测试区间此前已被项目查看，因此结果属于探索性复分析。

## 方法

- SSA成员种子：42, 43, 44。
- 集成：三个成员等权平均，不根据测试表现分配权重。
- 残差修正：`最终值 = 单光预测 + λ × (孪生集成预测 - 单光预测)`。
- 验证集选择得到 `λ = 1.00`；λ被限制在[0, 1]，最终值不会越过两种基础预测。
- 按验证稳定性目标最终选中的候选：`ssa_seed_winner_ensemble`。

## 结果

| model | split | mae | rmse | fitness |
| --- | --- | --- | --- | --- |
| ordinary_optical_reservoir | val | 0.386178 | 0.487698 | 0.498674 |
| ssa_best_single | val | 0.353049 | 0.446791 | 0.452017 |
| ssa_seed_winner_ensemble | val | 0.348384 | 0.444837 | 0.451821 |
| ordinary_plus_siamese_residual | val | 0.348384 | 0.444837 | 0.451821 |
| ordinary_optical_reservoir | test | 0.311511 | 0.397189 | 0.398380 |
| ssa_best_single | test | 0.303929 | 0.400142 | 0.404253 |
| ssa_seed_winner_ensemble | test | 0.316162 | 0.404185 | 0.406856 |
| ordinary_plus_siamese_residual | test | 0.316162 | 0.404185 | 0.406856 |

- SSA等权集成相对单光测试RMSE变化：+1.76%。
- 残差式修正相对单光测试RMSE变化：+1.76%。
- 验证集最终选中方案相对单光测试RMSE变化：+1.76%。
- 结论：验证集选择的方案在这段探索性测试上未优于单光储备池，因此暂不替换当前主结果。

这里不能因为反复查看同一测试区间而宣称获得新的无偏提升。
正式结论应冻结当前方案，等待2026年5月之后的新月份，或另做滚动回测。

## 关键文件

- `tables/ensemble_member_configurations.csv`：三个SSA成员及验证表现。
- `tables/residual_strength_selection.csv`：λ的验证集选择过程。
- `tables/validation_predictions.csv`：45个验证目标的逐月结果。
- `tables/test_predictions.csv`：47个测试目标的逐月探索性结果。
- `tables/model_comparison.csv`：单光、最佳单个SSA、SSA集成和残差修正的统一比较。
- `figures/validation_residual_strength.png`：残差强度选择曲线。
- `figures/test_prediction_comparison.png`：测试预测曲线。
- `figures/test_metric_comparison.png`：测试MAE/RMSE柱状图。
