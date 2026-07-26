# 严格封闭 50 窗口孪生光储备池：单因素消融报告

## 实验设计

基线固定为：`gap=12 + 纯形状参考距离 + 50维[h_i-h_j]`。每个消融配置相对基线只改变一个概念因素：

- 只改 gap：保持纯形状距离和50维状态差，在1、3、6、12中按验证集选择。
- 只改混合距离：保持gap=12和50维状态差，改为形状与绝对水平各占0.5。
- 只改100维特征：保持gap=12和纯形状距离，改为`[h_i,h_i-h_j]`。
- 所有配置只能访问同一批最后50个训练窗口；验证和测试参考均来自该固定窗口库。

## 只改 gap：验证集选择

| configuration | changed_factor | gap_months | reference_mode | feature_mode | feature_dimension | legal_train_candidates | train_pair_targets | selected_train_pairs | selected_alpha | val_mae | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| only_gap_1 | gap_months | 1 | shape | signed_diff | 50 | 741 | 38 | 173 | 10.000000 | 0.574994 | 0.768890 |
| only_gap_3 | gap_months | 3 | shape | signed_diff | 50 | 666 | 36 | 165 | 10.000000 | 0.599899 | 0.790288 |
| only_gap_6 | gap_months | 6 | shape | signed_diff | 50 | 561 | 33 | 150 | 10.000000 | 0.586801 | 0.781840 |
| only_gap_12 | gap_months | 12 | shape | signed_diff | 50 | 378 | 27 | 118 | 1.000000 | 0.516337 | 0.696738 |

只改 gap 时，验证集选出的最佳值为 **gap=12**。

## 单因素最终比较

| configuration | changed_factor | gap_months | reference_mode | feature_mode | feature_dimension | legal_train_candidates | train_pair_targets | selected_train_pairs | selected_alpha | val_mae | val_rmse | aggregation | test_mae | test_rmse | val_rmse_change_vs_baseline | test_rmse_change_vs_baseline | test_rmse_change_percent_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | none | 12 | shape | signed_diff | 50 | 378 | 27 | 118 | 1.000000 | 0.516337 | 0.696738 | inverse_distance | 0.799978 | 0.886371 | 0.000000 | 0.000000 | 0.000000 |
| only_gap | gap_months | 12 | shape | signed_diff | 50 | 378 | 27 | 118 | 1.000000 | 0.516337 | 0.696738 | inverse_distance | 0.799978 | 0.886371 | 0.000000 | 0.000000 | 0.000000 |
| only_hybrid_distance | reference_mode | 12 | hybrid | signed_diff | 50 | 378 | 27 | 118 | 1.000000 | 0.550141 | 0.741536 | inverse_hybrid_distance | 0.845661 | 0.933889 | 0.044798 | 0.047517 | 5.360890 |
| only_100d_feature | feature_mode | 12 | shape | target_plus_diff | 100 | 378 | 27 | 118 | 10.000000 | 0.698537 | 0.926594 | inverse_distance | 1.234966 | 1.353783 | 0.229856 | 0.467412 | 52.733166 |

相对基线的测试 RMSE 变化：

- 只改 gap：验证集仍选择基线设置，测试 RMSE 保持不变。
- 只改混合距离：测试 RMSE 上升（退化） 5.36%。
- 只改100维特征：测试 RMSE 上升（退化） 52.73%。

## 结论

1. **不应缩小 gap。** gap=1、3、6虽然增加了训练配对目标，但验证误差都高于gap=12，因此严格按验证集选择后仍保留gap=12。
2. **当前等权混合距离不优于纯形状距离。** 单独加入绝对水平后，验证和测试误差同时上升，说明水平接近不一定代表下一月CPI变化规律接近。
3. **100维特征是性能下降的主要来源。** 在只有118个训练样本对时，读出维度从50增加到100，测试RMSE明显上升，表现出小样本高维过拟合。
4. 因此上一轮三项组合实验的退化不是单纯由gap造成；100维特征贡献了最大的负面影响，等权混合距离也有较小的负面影响。

当前测试区间在此前工作中已经被查看过，因此仍属于探索性复分析。正式结论需要在冻结最终配置后使用新的未见时间区间验证。

关键文件：

- `tables/only_gap_validation_comparison.csv`：只改gap的验证筛选。
- `tables/single_factor_model_comparison.csv`：三个单因素与基线的验证/测试指标。
- `tables/single_factor_test_predictions.csv`：47个测试月份逐月预测。
- `figures/single_factor_rmse_comparison.png`：单因素RMSE柱状图。
- `experiment_manifest.json`：配置隔离与防泄漏记录。
