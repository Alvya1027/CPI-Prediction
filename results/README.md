# 结果文件说明

本目录保存 baseline、普通光储备池和孪生光储备池的指标、预测值、模型参数与图片。

## 核心结果

- `tables/optical_model_comparison.csv`：普通光储备池与孪生光储备池的统一 MAE、RMSE 对照。
- `tables/optical_reservoir_metrics.csv`：普通光储备池结果。
- `tables/siamese_optical_reservoir_metrics.csv`：孪生光储备池结果。
- `tables/siamese_optical_configuration_comparison.csv`：孪生特征与聚合方式的验证集选择记录。
- `figures/optical_model_test_comparison.png`：两种光储备池在测试集上的预测曲线。
- `figures/siamese_optical_test_predictions.png`：孪生光储备池测试预测曲线。

`predictions_<split>.csv` 保存每个月的真实 CPI、预测 CPI 和误差；`pair_predictions_<split>.csv` 额外保存孪生样本对的 CPI 差值预测。

当前真实状态结果中，普通光储备池测试 MAE/RMSE 为 `0.2798/0.3552`，孪生光储备池为 `0.3300/0.4261`。孪生结构已经跑通，但第一版尚未超过普通光储备池。
