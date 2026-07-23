# 近50个月训练集：传统模型对照

同一套CPI滑动窗口数据：训练集取原训练集最后50个目标，验证集45个、测试集47个保持不变。
模型参数只使用验证集选择，测试集仅用于最终评估。该目录独立于单网络光储备池结果。

- `tables/classical_models_recent50_metrics.csv`：指标
- `tables/classical_models_recent50_validation_trials.csv`：验证集选参记录
- `tables/classical_models_recent50_predictions.csv`：逐样本预测
- `tables/*_selected_params.json`：各模型最终参数
- `figures/classical_models_recent50_test_predictions.png`：测试预测对比图
