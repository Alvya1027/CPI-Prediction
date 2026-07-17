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

## 当前流程

1. `python -m src.create_siamese_pairs`：生成无目标泄漏的训练、验证、测试样本对。
2. `python -m src.export_cpi_to_matlab`：导出与 baseline 相同的 12 个月窗口。
3. 在 MATLAB 中运行 `run_all_cpi_simulations`：生成固定 mask、运行 `SL_RC.slx` 并提取状态。
4. `python -m src.siamese_reservoir_regression`：训练差值回归读出层并计算 MAE、RMSE。

详细接口见 `docs/siamese_optical_reservoir_interface.md`。

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

