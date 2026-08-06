# 历史连续状态方案：固定共享光储备池 + 共享线性输出权重

> **LEGACY 提醒：这不是老师最新要求的显式 MATLAB 双分支结果。** 本目录使用的是旧 `continuous_serial_shared_fixed_reservoir` 状态缓存；MATLAB 中没有让每个目标窗口和参考窗口同时通过两个 Model Reference。下列数值只保留为历史对照，不能作为新 `Twin_SL_RC + isolated_repeated_window + R=4` 方案的正式指标。新方案协议见 `docs/teacher_explicit_twin_matlab_protocol.md`，其状态尚待有 MATLAB/Simulink 的电脑实际生成。

本实验严格使用 50/45/47 个环比目标，储备池内部参数、掩码和状态全部固定，只闭式训练同一组输出权重 `(b,w)`。孪生训练目标为绝对 CPI 损失与关系差值损失的联合；741 对仅由 50 个训练月份内部组合，不是新增样本。

冻结配置：`alpha=100.0`、`lambda_pair=0.1`、`K=5`、聚合=`mean`、`gap=1`。验证选参期间没有读取测试数据或测试状态，之后按冻结配置评价测试集一次。

| 模型 | 验证 MAE | 验证 RMSE | 测试 MAE | 测试 RMSE |
| --- | ---: | ---: | ---: | ---: |
| 同状态单光读出 | 0.386178 | 0.487698 | 0.311511 | 0.397189 |
| 老师方案孪生参考还原 | 0.323680 | 0.420836 | 0.336503 | 0.436302 |

验证 RMSE 相对单光下降 `13.71%`，但测试 MAE/RMSE 分别上升 `8.02%` / `9.85%`。因此该配置在验证区间表现更好，却没有迁移到测试区间，不能得出孪生方案优于单光的结论。验证中 `lambda_pair=0` 的参考校准消融与正关系损失几乎相同，说明验证增益主要来自历史参考校准，而非差值监督本身。

关键文件：

- `tables/validation_model_comparison.csv`
- `tables/test_model_comparison.csv`
- `tables/test_prediction_comparison.csv`
- `tables/selected_configuration.json`
- `figures/validation_prediction_comparison.png`
- `figures/test_prediction_comparison.png`
- `experiment_manifest.json`

方法与边界详见 `docs/teacher_shared_optical_reservoir_protocol.md`。
