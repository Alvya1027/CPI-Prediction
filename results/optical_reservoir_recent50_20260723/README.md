# 单网络光储备池：近50个月训练集实验

本目录是独立实验结果，不覆盖 `results/tables/` 下原有212条训练集结果。
训练集取原始时间顺序训练集的最后50个目标窗口；验证集45个目标、测试集47个目标保持不变。
本次只运行普通/单网络光储备池读出，没有运行孪生模型。

- `tables/optical_reservoir_metrics.csv`：各划分指标
- `tables/optical_reservoir_predictions_*.csv`：预测与误差
- `tables/optical_reservoir_alpha_selection.csv`：验证集选alpha记录
- `tables/optical_reservoir_run_summary.json`：运行配置
- `data_manifest.json`：实际使用的样本范围与状态维度
- `input_states_recent50/`：本实验独立使用的状态缓存
- `figures/`：验证集、测试集预测图和测试残差图
