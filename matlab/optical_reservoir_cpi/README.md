# CPI 孪生光储备池回归接口

本目录用于把现有 CPI 小样本数据送入老师提供的光储备池 Simulink 模型。`SL_RC.slx` 是工作副本，原始 `ESN/SL_RC.slx` 不做任何修改。

## 当前Train45 / Test47无验证集入口（2026-08-07）

```matlab
% 阶段A：只生成2018-09至2022-05的train45
outputs = run_teacher_twin_train45_noval();

% Python固定配置并写出test_generation_authorization.json之后
% 阶段C：生成保持不变的2022-06至2026-04 test47
outputs = run_teacher_twin_test45_frozen();
```

新输出位于相邻的
`optical_reservoir_cpi_mom_train45_noval_20260807/`。没有独立验证集，
测试集不能用于调参；所有配置必须在阶段C之前冻结。

## 历史Train50 / Val45 / Test47显式双分支入口

正式方案使用 `Twin_SL_RC.slx` 中两个 Model Reference，它们都引用同一个
`SL_RC_shared_branch.slx`。每个目标/参考窗口同时经过两支固定动力学；窗口
重复4轮，按节点结束时刻截取第4轮50维状态。只有共享线性 `W_out` 接受
监督训练，光储备池内部参数不训练。

MATLAB 依次运行：

```matlab
% 阶段A：只生成train50和val45，并建立不可变审计
outputs = run_teacher_twin_train_validation();

% Python验证冻结并写出test_generation_authorization.json之后
% 才允许阶段C：生成test47
outputs = run_teacher_twin_test_frozen();
```

正式状态位于相邻 profile 的 `states_twin/state_cache_<split>.mat`，并带有
`.manifest.json` 与 `audits_twin/` 原始审计数组。状态文件不保存 `y` 或
`target`；Python 会按 `sample_id` 连接标签，训练单光与孪生读出，并强制
两者使用同一批状态。

关键新文件：

- `config_twin_cpi_rc.m`：固定 R=4、node-end、求解器与正式协议；
- `build_shared_reservoir_branch_model.m`：把工作副本封装成可多实例引用分支；
- `build_twin_shared_reservoir_model.m`：建立两个并行且同源的 Model Reference；
- `prepare_twin_window_cache.m`：仅加载输入字段，训练拟合 mask/缩放；
- `run_one_twin_window_pair.m`：执行一次真实双支仿真；
- `run_twin_state_cache.m`：每两个唯一窗口生成缓存，并写严格 manifest；
- `audit_twin_equivalence.m`：执行 A/B、B/A、A/A、复现与缓存重放审计；
- `fit_twin_shared_output_weights.m`：展示只求共享 `W_out` 的 MATLAB 闭式训练。

完整边界见 `docs/teacher_explicit_twin_matlab_protocol.md`。当前开发电脑没有
MATLAB/Simulink，因此这些入口已通过静态合同检查，但尚未在本机产生正式
Twin 状态或指标。

## 旧连续状态结构（LEGACY）

1. 直接复用 baseline 使用的 12 个月滑动窗口和训练/验证/测试划分。
2. 用训练集标准化后的输入生成固定 mask。
3. 把每个 12 维样本展开成 50 个连续虚拟节点输入。
4. 为 Simulink 生成 `simin_train.mat`、`simin_val.mat` 和 `simin_test.mat`。
5. 分别运行模型、动态提取储备池状态，并按 `sample_id` 缓存。
6. Python 根据样本对取出 `h_i` 和 `h_j`，用回归读出层预测 `delta_cpi`。

该流程按 split 连续运行一个 `SL_RC.slx`，没有在 MATLAB 中显式搭建目标/参考双分支；只保留为历史对照，不能作为老师最终方案指标。

## 旧流程使用方法

先在仓库根目录导出 baseline 的同一批样本：

```powershell
python -m src.export_cpi_to_matlab
```

然后在 MATLAB 中进入本目录并一键执行：

```matlab
outputs = run_all_cpi_simulations();
```

也可以逐步执行 `prepare_cpi_inputs`、`run_cpi_simulation` 和 `extract_cpi_states`，方便定位问题。

得到 `states/states_train.mat`、`states_val.mat` 和 `states_test.mat` 后，在仓库根目录执行：

```powershell
python -m src.siamese_reservoir_regression
python -m src.optical_reservoir_regression
```

第一条命令训练孪生状态差读出，第二条命令训练普通单状态读出并生成两种模型的统一对比表和预测图。

## 新版 MATLAB 兼容

老师模型保存于 R2016a。当前脚本在工作副本中自动完成以下兼容处理：

- `repair_legacy_noise_block.m`：将缺失的旧版激光高斯噪声模块替换为等价的 Simulink 随机数模块；
- `remove_optional_spectrum_analyzer.m`：将只负责显示的频谱分析仪替换为 Terminator；
- `ensure_cpi_state_logger.m`：增加独立 `CPIStateData` 记录器，以 Timeseries 格式保存时间和响应。

原始 `ESN/SL_RC.slx` 始终保留不变。模型内部已有 `4 us` 输入延迟，脚本在延迟后按 40 ps 间隔提取状态。

## 固定参数

- 输入窗口：12 个月
- 虚拟节点：50 个
- mask：`12 x 50` 的固定二值矩阵，随机种子为 42
- 节点间隔：40 ps
- 延迟反馈：2.04 ns
- 预热时间：4 us
- 输入增益：0.004

mask 的幅度只根据训练集确定，验证集和测试集沿用同一个 mask 与缩放系数，避免提前看到测试数据。三个数据集都经过完全相同的映射，这一点和后续正式预测时的流程一致。

## 文件说明

- `config_cpi_rc.m`：集中保存仿真参数和路径。
- `prepare_cpi_inputs.m`：生成固定 mask 与三个数据划分的 `simin`。
- `run_cpi_simulation.m`：调用 Simulink 并保存模型输出。
- `repair_legacy_noise_block.m`：兼容新版 MATLAB 中已移除的旧噪声模块。
- `remove_optional_spectrum_analyzer.m`：移除额外工具箱依赖的显示模块。
- `ensure_cpi_state_logger.m`：记录真正的储备池输出及时间轴。
- `extract_cpi_states.m`：根据时间轴动态截取响应，生成 `样本数 x 50` 状态矩阵。
- `run_all_cpi_simulations.m`：依次完成输入、仿真和状态提取。
- `data/cpi_windows.mat`：Python 导出的 CPI 窗口、标签、日期和样本索引。
- `inputs/`：生成的 Simulink 输入。
- `responses/`：仿真得到的储备池响应。
- `states/`：供 Python 孪生回归读出的状态缓存。

不要在训练、验证、测试之间重新生成 mask，也不要用测试集重新计算缩放系数。
