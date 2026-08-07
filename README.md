# CPI Prediction

本项目研究 CPI 月度小样本预测。当前核心方法为孪生光储备池回归：两个共享参数的光储备池分支提取目标窗口和历史参考窗口的动态状态，回归读出层预测连续 CPI 差值，再利用已知参考 CPI 还原目标 CPI。

## 模型定义

```text
x_i -> shared optical reservoir -> h_i --+
                                          +-> Ridge readout -> delta_cpi_hat
x_j -> shared optical reservoir -> h_j --+

cpi_i_hat = cpi_j + delta_cpi_hat
```

相似标签只用于参考窗口分析，不是最终分类目标。第一版不使用 Contrastive Loss，也不训练光储备池内部参数，只训练回归读出层。

## 当前正式划分：Train45 / Test47，无验证集（2026-08-07）

当前环比实验已取消独立验证集：训练目标改为 `2018-09—2022-05`
共45个，测试目标保持 `2022-06—2026-04` 共47个。测试集不得用于选参，
所以读出层和参考策略参数必须在测试状态生成前固定。

严格流程为：

1. MATLAB：`run_teacher_twin_train45_noval()`，只生成45个训练状态；
2. Python：`python scripts/run_teacher_explicit_twin_mom_train45_noval.py`，冻结固定配置并写测试状态授权；
3. MATLAB：`run_teacher_twin_test45_frozen()`，生成47个测试状态；
4. Python：向同一入口传入 `--frozen-test <fixed_configuration.json>`，只评价测试集一次。

新数据和状态独立存放在
`matlab/optical_reservoir_cpi_mom_train45_noval_20260807/`，不覆盖历史实验。
完整边界见
[`docs/teacher_explicit_twin_train45_noval_protocol.md`](docs/teacher_explicit_twin_train45_noval_protocol.md)。

## 历史正式方案：Train50 / Val45 / Test47（2026-08-02）

最终实验不再把旧的单分支连续状态直接当成孪生方案。MATLAB 会从同一个
`SL_RC_shared_branch.slx` 建立两个 Model Reference：目标窗口和参考窗口
同时进入两条动态独立、结构与参数完全共享的分支。每个12个月窗口从同一
初态开始重复4轮，按节点结束时刻截取第4轮50维状态。储备池内部、mask和
物理参数固定，监督训练只闭式求一组共享线性输出权重。

该历史流程为：

1. MATLAB：`run_teacher_twin_train_validation()`，只生成50个训练状态和45个验证状态，并完成双分支等价性审计。
2. Python：`python scripts/run_teacher_explicit_twin_mom_closed50.py`，只用训练/验证选择配置，冻结后写测试状态授权。
3. MATLAB：`run_teacher_twin_test_frozen()`，生成47个测试状态。
4. Python：给同一入口传入 `--frozen-test <selected_configuration.json>`，只评价测试集一次。

单光基线与孪生模型必须读取同一批新 Twin 状态；旧连续状态数值只作历史
对照。完整方法、文件合同和运行边界见
[`docs/teacher_explicit_twin_matlab_protocol.md`](docs/teacher_explicit_twin_matlab_protocol.md)。

当前电脑未安装 MATLAB/Simulink，因此仓库已完成可执行接口与静态/Python
验证，但尚未生成新方案的正式状态和性能指标。

## 旧版通用流程

1. `python -m src.create_siamese_pairs`：生成无目标泄漏的训练、验证、测试样本对。
2. `python -m src.export_cpi_to_matlab`：导出与 baseline 相同的 12 个月窗口。
3. 在 MATLAB 中运行 `run_all_cpi_simulations`：生成固定 mask、运行 `SL_RC.slx` 并提取状态。
4. `python -m src.siamese_reservoir_regression`：训练差值回归读出层并计算 MAE、RMSE。

该流程保留用于旧连续状态复现。详细接口见 `docs/siamese_optical_reservoir_interface.md`。

## 仓库结构

- `data_raw/`：原始 CPI 数据。
- `data_processed/`：清洗数据、滑动窗口和孪生样本对。
- `src/`：Python 数据处理、baseline 和孪生回归代码。
- `ESN/`：老师提供的原始 MATLAB/Simulink 文件，只作保留。
- `matlab/optical_reservoir_cpi/`：CPI 光储备池工作副本和动态接口。
- `results/`：指标、预测表和图片。
- `docs/`：项目说明、报告和周计划。

## 路径约定

不要在代码中写个人电脑的绝对路径。Python 统一从 `src/config.py` 读取项目路径；MATLAB 统一使用脚本所在目录构造路径。

