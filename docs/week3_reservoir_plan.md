# 第3周储备池实现计划

> 目标：实现普通储备池计算（Echo State Network），能读取 CPI 数据并完成预测，与 Ridge、SVR、LSTM 等基线模型对比。
> 时间参考：6月22日—7月5日（两周合并为第3阶段核心工作）
> 验收标准：`src/reservoir.py` 能运行，RC 能预测 CPI，误差指标可与基线模型对比。

---

## 里程碑 1：数据适配（1-2天）

**目标**：让 ESN 能读取项目已有的 CPI 滑动窗口数据。

**任务清单**：
- [ ] 在 `src/reservoir.py` 中实现 `load_cpi_data()` 函数，读取 `data_processed/` 下的 `X_train.npy`、`y_train.npy` 等
- [ ] 数据标准化：CPI 值范围通常在 90~110 之间，需要归一化到 [-1, 1] 或 [0, 1]
- [ ] 验证数据加载正确：print 训练集/验证集/测试集的形状
- [ ] 确认数据按时间顺序划分，没有随机打乱

**验收标准**：
- 运行 `python src/reservoir.py` 能打印出各数据集的形状
- 打印出的形状与 `data_description.md` 中描述一致

**依赖**：B 的 `data_processed/` 数据已就绪（已完成）

---

## 里程碑 2：ESN 核心实现（2-3天）

**目标**：实现 ESN 的储备池生成、状态收集、输出权重训练。

**任务清单**：
- [ ] 实现 `init_reservoir(N, spectral_radius, sparsity, leak_rate, input_scaling, random_seed)`
  - 生成随机输入权重 `W_in`（均匀分布 [-1, 1]）
  - 生成稀疏随机储备池矩阵 `W`（稀疏度约 5%）
  - 缩放 `W` 使谱半径 = `spectral_radius`
- [ ] 实现 `collect_states(X, W_in, W, leak_rate)` 收集所有时间步的储备池状态
  - 注意：前 `washout` 步（如 10 步）丢弃，用于消除初始瞬态
- [ ] 实现 `train_output(X_states, y, ridge_beta)` 用岭回归训练输出权重
- [ ] 实现 `predict(X, W_out, ...)` 用训练好的权重做预测
- [ ] 实现 `esn_pipeline(X_train, y_train, X_test, y_test, params)` 端到端 pipeline

**验收标准**：
- 用正弦波 `sin(t)` 作为玩具数据：输入过去 12 个点预测下 1 个点
- 预测 NRMSE < 0.3（正弦波任务很简单，ESN 应该能轻松做到）
- 如果没有过拟合，说明 ESN 实现正确

**参考公式**（详见 `reservoir_computing_notes.md`）：

```
x(t) = (1-α)·x(t-1) + α·tanh(W_in·u(t) + W·x(t-1))
W_out = Y_target · X^T · (X·X^T + β·I)^(-1)
```

---

## 里程碑 3：CPI 预测与对比（2-3天）

**目标**：用 ESN 预测 CPI，并与基线模型对比。

**任务清单**：
- [ ] 用默认参数跑 ESN 在 CPI 数据上的预测
  - 节点数 N=200, 谱半径=0.9, 泄露率=0.3, 输入缩放=0.5, 正则化=1e-4
- [ ] 调用 A 的 `src/metrics.py` 计算 MAE、RMSE、MAPE、Directional Accuracy
- [ ] 参数扫描：尝试不同的 N、谱半径、泄露率、正则化系数
  - N: [50, 100, 200, 300]
  - 谱半径: [0.5, 0.7, 0.9, 0.99]
  - 泄露率: [0.1, 0.3, 0.5, 0.7, 1.0]
- [ ] 记录最优参数组合
- [ ] 将 ESN 结果与 `first_baseline_results.csv` 中的 Naive、MA、Ridge、SVR 对比
- [ ] 画出真实值 vs ESN 预测值图，保存到 `results/figures/esn_prediction.png`
- [ ] 保存 ESN 误差结果到 `results/tables/esn_results.csv`

**验收标准**：
- ESN 的 MAE 和 RMSE 至少不差于 Ridge 和 SVR
- 如果 ESN 明显优于简单移动平均，说明储备池的非线性映射确实有帮助
- 参数扫描结果记录在 `results/tables/esn_param_search.csv`

---

## 里程碑 4：结果整理与文档（1天）

**目标**：整理结果，更新文档，为第4阶段做准备。

**任务清单**：
- [ ] 更新 `docs/reservoir_computing_notes.md`，补充实际调参经验
- [ ] 将 ESN 最优参数和误差写入 `results/experiment_log.xlsx`
- [ ] 在 `notebooks/03_reservoir_draft.ipynb` 中更新为实际 CPI 预测的结果
- [ ] 协助 A 更新 `docs/weekly_report.md`（储备池部分）
- [ ] 检查时间序列对齐：确保 X 和 y 的位置正确，没有未来数据泄漏
- [ ] 为第4阶段准备：阅读 L-K 方程、RK4 数值方法、虚节点读取

**验收标准**：
- `results/tables/esn_results.csv` 包含 ESN 的最优参数和误差
- `results/figures/esn_prediction.png` 包含真实值与预测值对比
- 确认没有数据泄漏

---

## 里程碑依赖关系

```
M1(数据适配) → M2(ESN核心) → M3(CPI预测对比) → M4(结果整理)
```

M1 必须先完成，后续都是串行依赖。M2 的玩具数据验证（正弦波）可以在 M1 完成前独立进行。

---

## 风险与应对

| 风险 | 应对方案 |
|------|----------|
| ESN 在 CPI 上表现不如简单基线 | 增加节点数、调整谱半径和泄露率、尝试不同的储备池稀疏度 |
| 岭回归过拟合 | 增大正则化系数 β，或使用交叉验证选 β |
| 训练时间过长 | 节点数不宜超过 500，用 `numpy.linalg.solve` 替代 `inv` 加速 |
| 与 B/C 的数据接口不兼容 | 提前确认数据格式，用 `assert` 检查形状 |