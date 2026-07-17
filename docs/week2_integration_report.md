# 第二周孪生光储备池回归联调记录

## 已完成

1. 保留老师 `ESN/` 原始文件，建立独立 CPI 工作副本。
2. 将 baseline 的 12 个月窗口、时间划分和全局样本索引导出到 MATLAB。
3. 固定 12 x 50 mask、50 个虚拟节点、40 ps 节点间隔、2.04 ns 延迟反馈和 0.004 输入增益。
4. 补齐训练、验证、测试三划分的统一 Simulink 输入接口。
5. 根据 Scope 时间轴动态提取 `样本数 x 50` 状态，删除 NARMA 样本数和响应区间硬编码。
6. 完成孪生回归读出：`h_i-h_j -> Ridge -> delta_cpi_hat`。
7. 完成 `cpi_j + delta_cpi_hat` 还原和多参考窗口聚合。
8. 修正验证/测试参考选择中的目标泄漏问题，并保证验证 45/45、测试 47/47 个目标完整覆盖。

## 已验证

- 样本对生成可重复执行，运行时间约 4 秒。
- 训练集 972 对、验证集 225 对、测试集 235 对。
- 验证和测试每个目标使用 5 个历史参考窗口。
- Python 端到端接口测试通过：可以读取三份状态、训练读出层、输出全部 47 个测试月份并保存指标和预测图。
- 测试同时验证了有符号状态差的方向和 `sample_id` 映射。

## 尚需在有 MATLAB/Simulink 的电脑上执行

当前开发环境没有可调用的 MATLAB，因此尚未生成真实 `SL_RC.slx` 响应，也不能把模拟状态测试结果当作光储备池预测结果。

正式运行步骤：

```text
仓库根目录：python -m src.export_cpi_to_matlab
MATLAB 目录：outputs = run_all_cpi_simulations();
仓库根目录：python -m src.siamese_reservoir_regression
```

运行后重点检查：

1. Scope 输出是否为 `ScopeData` 或 `ScopeData1`；
2. 响应时间是否覆盖 4 us 延迟后的全部输入；
3. 三个状态矩阵是否分别为 212 x 50、45 x 50、47 x 50；
4. 状态是否存在 NaN、Inf、全零或异常饱和；
5. 正式测试指标是否优于 Seasonal Naive、ARIMA/SARIMA、ETS 等小样本时序 baseline。

## 第一版明确不做

- 不把 `similar_label` 当作最终标签；
- 不训练 Logistic 相似度分类器；
- 不加入 Contrastive Loss；
- 不额外挂载随机森林或 LSTM 作为最终预测器；
- 不训练光储备池内部参数，只训练回归读出层。
