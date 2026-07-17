# 孪生光储备池回归接口说明

## 1. 最终任务

本项目预测连续 CPI，而不是判断两个窗口是否相似。对目标窗口 `i` 和历史参考窗口 `j`：

```text
x_i -> shared SL_RC -> h_i
x_j -> shared SL_RC -> h_j

z_ij = h_i - h_j
delta_cpi_hat = Ridge(z_ij)
cpi_i_hat = cpi_j + delta_cpi_hat
```

真实回归标签为 `delta_cpi = cpi_i - cpi_j`。第一版损失为差值均方误差，不使用二分类交叉熵和 Contrastive Loss。

## 2. “两个共享分支”的实现

两个分支必须使用相同的：

- `SL_RC.slx` 模型；
- 12 x 50 固定 mask；
- 输入增益和缩放系数；
- 50 个虚拟节点及 40 ps 节点间隔；
- 延迟反馈和非线性响应；
- 状态截取方法。

Simulink 仿真成本较高，因此工程实现不会为每个样本对重复运行两个模型。每个唯一 `sample_id` 只通过共享光储备池一次并缓存状态。样本对随后按 ID 读取 `h_i` 和 `h_j`。这与两个共享参数分支逐对重复计算使用的是同一个映射，同时避免同一窗口被重复仿真。

## 3. 文件流转

### Python 到 MATLAB

运行：

```bash
python -m src.export_cpi_to_matlab
```

输出 `matlab/optical_reservoir_cpi/data/cpi_windows.mat`，包含与 baseline 完全一致的 12 个月窗口、标签、日期和全局 `sample_id`。

### MATLAB 光储备池

在 `matlab/optical_reservoir_cpi/` 中运行：

```matlab
outputs = run_all_cpi_simulations();
```

脚本依次完成：

1. `prepare_cpi_inputs`：用训练集确定 mask 缩放，生成三个 `simin`；
2. `run_cpi_simulation`：运行老师模型的工作副本；
3. `extract_cpi_states`：根据 Scope 时间轴和 4 us 延迟动态提取状态。

状态文件包含：

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| `state_matrix` | `样本数 x 50` | 每个窗口的光储备池状态 |
| `sample_id` | `样本数 x 1` | 全局样本编号 |
| `target` | `样本数 x 1` | 原始 CPI 标签 |
| `target_date` | `样本数 x 1` | 目标月份 |
| `mask` | `12 x 50` | 三个划分共享的固定 mask |

状态提取不再使用 `100001:200000`、`NTrain=2000` 等 NARMA 专用硬编码，而是根据样本数、虚拟节点数、时间轴和延迟自动计算。

### MATLAB 到 Python

状态文件准备好后运行：

```bash
python -m src.siamese_reservoir_regression
```

Python 使用 `pair_indices_<split>.csv` 取出状态差，训练 Ridge 读出层，并输出：

- 每个样本对的 `delta_cpi` 预测；
- 每个参考窗口还原出的 CPI；
- 多参考窗口平均后的目标 CPI；
- train/val/test 的 MAE、RMSE；
- 测试集预测曲线和可复现模型参数。

## 4. 数据泄漏边界

- 标准化和 mask 缩放只在训练集上拟合。
- `target_j_date` 必须早于 `target_i_date`。
- 训练集可用真实差值控制训练对分布。
- 验证集和测试集只能根据已知输入窗口距离选择参考，不能用 `cpi_i` 或 `delta_cpi` 选参考。
- 测试指标按目标月份计算，不能把同一目标的多个参考对当成多个独立测试样本。

## 5. 默认设计选择

- 状态组合：默认 `h_i - h_j`，保留差值方向；可比较 `signed_abs` 或拼接版本。
- 读出层：Ridge，只训练这一层。
- 多参考聚合：默认算术平均，可选按窗口距离倒数加权。
- 评价指标：目标 CPI 的 MAE、RMSE；差值误差仅作为诊断指标。
