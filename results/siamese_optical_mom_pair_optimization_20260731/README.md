# 环比严格封闭50样本：孪生配对结构优化

本实验只改变孪生模型的配对规则和训练权重，不改变光储备池、状态或50个月数据预算。
配置只根据45个月验证集选择；测试集随后评估一次。由于旧测试结果已经看过，本结果属于探索性分析。

- 最终配置：target_gap=12，training_mode=antisymmetric
- alpha=1000.0
- 参考聚合=trimmed_mean
- 候选训练对=741，训练行=1482，覆盖配对目标=38
- 验证MAE/RMSE=0.309749/0.406930
- 测试MAE/RMSE=0.319753/0.422209
- 测试RMSE相对单光变化=+6.30%
- 测试RMSE相对当前SSA孪生变化=+5.51%

详细验证消融见 `tables/validation_pair_structure_comparison.csv`。
