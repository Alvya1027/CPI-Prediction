# 老师最终显式双分支 MATLAB 实验协议

> **协议状态：待 MATLAB 实际执行。**  本文定义老师最新要求下的正式实验流程。当前电脑没有安装 MATLAB/Simulink，因此目前只完成了模型构建脚本、状态文件合同和分阶段运行接口；尚未生成正式的显式双分支状态，也没有该方案的正式验证集或测试集指标。旧连续状态实验的数值不能冒充本文方案的结果。

## 1. 最终任务定义

本轮最终方案解决的是严格封闭 50 个训练月份下的 CPI 环比预测。对目标窗口 `x_i` 和历史参考窗口 `x_j`，MATLAB/Simulink 中必须存在两个可见的光储备池分支：

```text
目标窗口 x_i ──> Target Shared Optical Reservoir ──> h_i ──┐
                                                              ├─> 同一组 W_out
参考窗口 x_j ──> Reference Shared Optical Reservoir ──> h_j ─┘
```

两个分支不是两套独立设计的模型，而是同一个固定光储备池模型的两个动态实例：

- 两个 Model Reference 都指向同一个 `SL_RC_shared_branch.slx`；
- 两支具有各自独立的动态状态，但模型结构和全部物理参数相同；
- 掩码、输入增益、激光器参数、反馈参数、初始条件、噪声设置和状态截取方式全部共享并冻结；
- 不训练光储备池内部参数，不增加可训练投影网络或多层神经网络；
- 唯一通过监督目标求解的模型参数是一组共享线性输出权重 `W_out`，以及可选偏置。

这里所说的“训练光储备池”，采用的是储备池计算领域的标准含义：Simulink 负责固定动力学的前向仿真并产生状态，训练阶段只根据这些状态求线性输出权重。MATLAB 仿真不是可有可无的数据预处理；没有经过新的显式双分支 SL_RC 前向仿真，就没有本文方案的正式状态。

## 2. 与旧连续状态方案的边界

老师原始 NARMA 示例采用连续串行信号：每个输入占 50 个虚拟节点，全部输入首尾相接后只运行一次储备池。原程序中的反馈延迟为 2.04 ns，而 50 个节点的输入周期为 2.00 ns，因此状态会跨输入周期保留前序信息。原程序还只在整条序列开头进行稳定处理和 `nForgetPoints` 丢弃，不会对每个样本单独复位。

该旧语义可写成：

```text
h_s = R(h_(s-1), x_s)
```

它适合保留为 `legacy_continuous_stream` 对照，但不再作为老师最新方案。老师最新要求强调目标窗口与参考窗口同时进入权重共享分支，因此本文重新定义独立窗口状态：

```text
h_s = R_isolated(x_s)
```

每个窗口从相同声明初态开始，重复输入若干周期后截取状态。这样 `h_s` 只由该窗口、固定模型和固定仿真协议决定，不依赖窗口在某个 split 中的前后排列，也不依赖它与哪个窗口配对。

因此必须遵守以下结论：

- 旧目录中的 `states_train.mat`、`states_val.mat`、`states_test.mat` 是连续状态，不能直接用于本文正式方案；
- 旧连续状态得到的共享读出指标只能作为初步实验或状态语义消融；
- 正式单光和正式孪生模型都必须重新使用本文产生的同一批独立窗口状态；
- 报告中不得把两种状态语义的指标放在同一行后声称是公平的“单光 vs 孪生”比较。

## 3. 为什么每个窗口重复 4 轮

固定物理参数为：

```text
虚拟节点数 N          = 50
节点间隔 theta        = 40 ps
单轮窗口时间 N*theta  = 2.00 ns
反馈延迟 tau          = 2.04 ns
```

如果一个窗口只输入一轮，仿真在 2.00 ns 处就开始截取或结束，而输入相关信号尚未完整经过一次 2.04 ns 反馈回路。这样得到的状态不能充分体现延迟反馈储备池的动力学。

正式协议将同一 masked window 连续重复 4 轮，并仅保留第 4 轮的 50 个节点。第 4 轮的时间区间为 6–8 ns。对该轮最早时刻也有：

```text
6.00 ns - 2 * 2.04 ns = 1.92 ns > 0
```

所以第 4 轮的全部节点至少已经受到两次与当前窗口有关的延迟反馈。相比之下，第 3 轮从 4 ns 开始，在该轮开头仍有 `4.00 - 2*2.04 < 0`，不能保证整轮节点都经历两次输入相关反馈。

因此固定：

```text
sequence_protocol = isolated_repeated_window
repeat_count       = 4
capture_cycle      = 4
state_width        = 50
sample_phase       = node_end
```

第 4 轮每个虚拟节点在该节点保持区间结束时采样，时间为
`4 μs + (151:200)×40 ps`；最后一个状态点恰好位于 `4.008 μs`。
MATLAB manifest 和 Python 合同会共同核验这 50 个采样时刻，不能把旧连续
缓存使用的边界相位混入本实验。

需要准确表述：4 轮是依据反馈覆盖范围确定的工程协议，不代表已经证明进入严格稳态。正式读取测试集前，可仅在训练集和验证集上进行第 4、5 轮状态差异审计；该审计只能依据预先规定的状态差阈值判断是否需要更长驱动，不能根据测试误差选择轮数。一旦配置冻结，测试阶段不得再修改重复轮数和截取位置。

原 SL_RC 中的 4 μs Transport Delay 继续保留。它用于每次仿真开始时让无输入的光反馈系统先稳定，并使输入到达时间对齐；它不是“每个窗口重复 4 轮”的替代，也不能被误写成每窗口监督样本的 washout。

## 4. 显式 Twin Model Reference 结构

正式 MATLAB 结构由两个层级组成。

### 4.1 共享分支模型

`build_shared_reservoir_branch_model.m` 从指定的 SL_RC 工作副本建立 `SL_RC_shared_branch.slx`：

- 保留激光器、延迟反馈、光注入和原有物理核心；
- 将根层的 `From Workspace` 换成一个输入端口；
- 将原 Scope 所接的储备池响应同时引到一个输出端口；
- 清除会让两个实例写入同一工作区变量的分支内 logger；
- 设置为允许多个 Model Reference 实例；
- 禁止跨仿真加载上次最终状态。

原始 `ESN/SL_RC.slx` 不应被直接覆盖；所有修改只发生在工作副本和生成的共享分支文件中。

### 4.2 Twin 顶层模型

`build_twin_shared_reservoir_model.m` 建立 `Twin_SL_RC.slx`，其中包含：

- `Target Window Sequence` 输入；
- `Reference Window Sequence` 输入；
- `Target Shared Optical Reservoir` Model Reference；
- `Reference Shared Optical Reservoir` Model Reference；
- 两个彼此独立命名的状态 logger。

构建脚本必须检查：顶层恰好存在两个 Model Reference，并且两者的 `ModelName` 完全一致。两条储备池分支之间不得存在状态或信号交叉连接。

“共享”必须通过同一个被引用模型文件和相同参数哈希来证明，不能只凭两张看起来相似的 Simulink 截图声称共享。

## 5. 输入、数据划分与严格隔离

每个样本使用过去 12 个月 `actual` 环比值预测下一个月：

```text
x(t) = [actual(t-12), ..., actual(t-1)]
y(t) = actual(t)
```

统一划分固定为：

| split | 目标月份 | 目标数 | 允许用途 |
| --- | --- | ---: | --- |
| train | 2014-07 至 2018-08 | 50 | 拟合输入/状态标准化、mask 缩放、输出权重和训练关系 |
| val | 2018-09 至 2022-05 | 45 | 选择输出层正则化、关系损失权重、参考数和聚合方式 |
| test | 2022-06 至 2026-04 | 47 | 冻结后生成状态并只评价一次 |

物理隔离输入文件为：

```text
matlab/optical_reservoir_cpi_mom_recent50_20260730/data/
├── cpi_train_isolated.mat
├── cpi_val_isolated.mat
└── cpi_test_isolated.mat
```

输入处理必须满足：

1. 输入均值和尺度只在 train 的 50 行上拟合；
2. 12×50 二值 mask 只生成一次，随机种子固定；
3. mask 幅值缩放只根据 train 投影计算；
4. val/test 只能复用训练阶段保存的变换，不能重新拟合；
5. MATLAB 状态生成文件只能包含输入、月份、样本 ID 和仿真溯源，不得包含 `y`、`target`、`delta_cpi` 等标签字段；
6. 验证配置冻结前，不允许运行或读取测试状态。

训练关系对无论有多少行，都只是这 50 个原始训练月份内部的关系约束，不能宣传为增加了同等数量的独立训练样本。

## 6. 唯一窗口缓存为何不改变双分支语义

逐个重跑全部训练关系对会重复计算相同窗口。本文采用显式 Twin 生成唯一窗口缓存：每次 Simulink 仿真将两个唯一窗口分别送入 A、B 分支，两个分支同时执行；每个唯一 `sample_id` 只保存一条状态。之后关系对通过 ID 查找：

```text
(sample_i_id, sample_j_id) -> lookup(h_i, h_j)
```

由于每个窗口都使用相同初态、相同4轮输入协议、相同模型和固定噪声流，独立窗口映射不依赖同次仿真的另一窗口，因此缓存是确定性前向计算的复用，不是用 Python 代替光储备池。

同时必须如实记录实现边界：

- MATLAB 中确实存在两个同时运行、引用同一模型的光储备池分支；
- 缓存生成时同次仿真的两个窗口只是计算调度伙伴；
- 后续所有语义训练对并没有逐对重新运行一次 Simulink；
- manifest 应声明 `semantic_pairs_simulated_simultaneously=false`、`pair_states_resolved_by_sample_id=true`；
- 必须通过 A/B 交换、相同输入和伙伴独立性审计证明这种复用与逐对运行等价。

这比笼统地说“两个缓存状态就是双分支”更严谨，也避免把741个训练关系对重复仿真1482次。

## 7. 只训练共享输出权重

令训练集状态标准化后的向量为 `z_s`，共享输出函数为：

```text
f(h_s) = b + w^T z_s
```

两个分支调用同一个 `(b,w)`。因此差值为：

```text
f(h_i) - f(h_j) = w^T (z_i - z_j)
```

偏置在差值中自动消失。单参考预测为：

```text
delta_y_hat_ij = w^T (z_i - z_j)
y_hat_i^(j)    = y_j + delta_y_hat_ij
```

可用多个训练参考窗口产生若干候选预测，再按照验证阶段冻结的方式求平均或输入距离加权。参考选择只能使用输入窗口、月份和已冻结距离规则，不能使用目标 `y_i`、真实差值或目标预测误差。

共享输出权重可以按老师原 `train.m` 的伪逆方式求解，也可以采用 Ridge/Tikhonov 闭式求解。联合绝对值和差值监督时可写为：

```text
L_abs  = mean_s (b + w^T z_s - y_s)^2
L_pair = balanced_mean_(i,j) (w^T(z_i-z_j) - (y_i-y_j))^2
L      = L_abs + lambda_pair*L_pair + alpha*||w||^2
```

无论采用纯差值或联合目标，全部监督项只能求同一组 `(b,w)`。禁止加入共享投影网络、隐藏层、激活函数或用 Adam 更新储备池内部参数。输入标准化、固定 mask 和状态标准化属于只在训练集拟合并随后冻结的确定性变换，不应冒充额外监督样本，也不属于激光储备池内部权重训练。

## 8. 强制的分阶段运行顺序

正式流程必须严格遵守以下顺序，不能为了节省时间一次性生成 train/val/test。

### 阶段 A：MATLAB 只生成训练和验证状态

在 MATLAB 中进入：

```text
matlab/optical_reservoir_cpi/
```

然后运行：

```matlab
outputs = run_teacher_twin_train_validation();
```

入口会准备标签隔离输入、建立共享分支和 Twin 顶层、先用训练窗口完成不可变的全局等价性审计，再运行验证分支并写出状态缓存与 manifest。此时禁止调用 `test`。如需逐步排错，可再分别调用 `run_twin_state_cache('train', cfg, true)` 和 `run_twin_state_cache('val', cfg, false)`；验证阶段不得重写训练审计。

主要输出目录为：

```text
matlab/optical_reservoir_cpi_mom_recent50_20260730/
├── inputs_twin/
├── states_twin/
└── audits_twin/
```

其中应包含 train/val 的窗口输入、`state_cache_<split>.mat`、对应 manifest，以及 Twin 等价性审计文件。具体文件名以运行入口生成并由 manifest 记录的名称为准。

### 阶段 B：Python 只用 train/val 选择并冻结配置

在仓库根目录运行：

```powershell
python scripts/run_teacher_explicit_twin_mom_closed50.py
```

该命令必须只加载 train/val 隔离数据和 Twin 状态，完成验证选择后写出：

```text
results/siamese_optical_mom_teacher_explicit_twin_20260802/
└── tables/selected_configuration.json
```

冻结文件必须记录状态协议、train/val 状态哈希、选定输出层参数、参考选择、聚合方式以及“尚未测试”状态。随后程序才允许写出：

```text
inputs_twin/test_generation_authorization.json
```

该授权文件只允许下一阶段生成 `test` 状态，不能修改已冻结配置。

### 阶段 C：MATLAB 在冻结后生成测试状态

确认冻结文件和测试生成授权已经存在后，才能在 MATLAB 中运行：

```matlab
outputs = run_teacher_twin_test_frozen();
```

测试状态的模型、mask、输入变换、重复轮数、截取周期、求解器、噪声和全部协议哈希必须与 train/val 完全一致。任何漂移都应使后续 Python 合同拒绝加载。

### 阶段 D：Python 只评价一次测试集

使用阶段 B 输出的冻结配置运行：

```powershell
python scripts/run_teacher_explicit_twin_mom_closed50.py `
  --frozen-test results/siamese_optical_mom_teacher_explicit_twin_20260802/tables/selected_configuration.json
```

测试脚本只能加载冻结配置和新生成的 test 状态，输出一次最终指标。看见测试指标后不得继续修改 `repeat_count`、截取相位、`alpha`、`lambda_pair`、参考数、参考距离或聚合方式再重测。

## 9. 必须通过的审计

### 9.1 模型结构审计

- Twin 顶层恰好有两个 Model Reference；
- 两个 Model Reference 指向同一个共享分支文件；
- 两支模型哈希、参数哈希和 mask 哈希一致；
- 两条储备池分支之间无交叉状态连接；
- 原始源模型、共享分支模型和 Twin 顶层模型均保存 SHA-256；
- MATLAB、Simulink版本、求解器和固定步长均有记录。

### 9.2 动态等价性审计

至少执行以下实验：

1. **相同输入审计**：A、B同时输入同一窗口，检查 `h_A(x)` 与 `h_B(x)`；
2. **交换审计**：先输入 `(x_a,x_b)`，再输入 `(x_b,x_a)`，检查状态按分支交换；
3. **伙伴独立性审计**：固定 A 的窗口、改变 B 的窗口，检查 A 的缓存状态不变，反向同理；
4. **重复运行审计**：相同输入和种子重复执行，检查状态可复现；
5. **反对称审计**：训练输出层后检查 `delta_y_hat(a,b) = -delta_y_hat(b,a)`；
6. **缓存重放审计**：抽取若干真实关系对直接送入 Twin，检查直接状态与按 ID 查出的缓存状态一致。

所有比较必须使用预先固定的绝对、相对容差，并在读取状态时重新计算，不能只相信 MATLAB 写出的 `audit_passed=true`。

### 9.3 数据与文件合同审计

- train/val/test 行数分别为50/45/47，且月份、`sample_id` 与隔离数据逐行一致；
- 状态矩阵均为 `样本数 × 50`，无 NaN、无无穷值；
- 状态缓存不包含 CPI 标签字段；
- 每个唯一样本只对应一条缓存记录；
- 缓存记录包含实际执行分支、同次仿真伙伴和运行 ID；
- state、manifest、audit和输入变换的哈希相互一致；
- train/val/test 的固定协议字段完全相同；
- 验证阶段的文件访问记录证明没有打开 test 状态或 test 标签。

任一审计失败都只能报告“协议未通过”，不能继续输出正式测试指标。

## 10. 与单光储备池的公平比较

正式对比必须只改变读出关系，不能改变光储备池状态：

| 比较项 | 单光储备池 | 显式双分支孪生光储备池 |
| --- | --- | --- |
| 原始窗口和月份 | 同一套50/45/47 | 完全相同 |
| 可访问训练月份 | 50 | 50；关系对不增加独立月份 |
| 状态来源 | 新 `states_twin` 缓存 | 同一批新 `states_twin` 缓存 |
| 状态协议 | isolated，重复4轮，取第4轮 | 完全相同 |
| 光模型、mask和物理参数 | 固定 | 完全共享、固定 |
| 可训练参数 | 线性 `W_out` | 同一形式的共享线性 `W_out` |
| 输入到读出 | `h_i -> y_i` | `h_i-h_j -> y_i-y_j`，再用 `y_j` 还原 |
| 测试指标 | 47个唯一目标的MAE/RMSE | 同样按47个唯一目标计算 |

同一目标使用多个参考时，必须先聚合为一个目标预测，再计算目标级 MAE/RMSE；不能把多个参考对当成多个独立测试样本扩大样本数。

至少应报告：

1. 同状态的普通单光线性读出；
2. 纯差值共享读出；
3. 联合绝对值/差值共享读出；
4. 联合读出的直接输出 `f(h_i)`，仅用于判断增益来自关系监督还是参考值还原。

旧连续状态的单光、孪生结果可另设“状态语义消融”表，但不得与新 isolated 状态结果混成正式公平比较。

## 11. 当前完成度与下一步

当前电脑没有 MATLAB/Simulink，所以此刻只能确认：

- 显式 Twin 模型的构建逻辑已经定义；
- 独立窗口4轮输入协议已经固定；
- train/val/test 分阶段接口和状态合同已经定义；
- Python 将拒绝旧连续状态冒充 Twin 状态；
- 当前尚不存在由该新 Twin 模型实际生成并通过审计的正式状态；
- 当前尚不存在本文方案的正式验证指标和测试指标。

下一步需要由安装了 MATLAB/Simulink 且能够运行原 SL_RC 的同学先执行阶段 A，只返回 train/val 状态及全部审计文件。Python 完成验证冻结并产生授权以后，再请其执行阶段 C。只有阶段 D 成功结束后，才能回答“老师最终方案效果是否优于同状态单光储备池”。

## 12. 报告中的标准表述

在 MATLAB 尚未运行时，应表述为：

> 已按照老师最新要求完成显式双分支实验接口：两个 Simulink Model Reference 指向同一个固定 SL_RC 分支，目标和参考窗口采用独立复位、重复4轮并截取末轮50维状态，监督训练只求一组共享线性输出权重。由于本机未安装 MATLAB，目前尚未生成该方案的正式状态和性能指标；旧连续缓存状态的结果已降级为对照，待 MATLAB 训练/验证状态通过审计并冻结配置后，再生成测试状态并进行一次最终评价。

不得表述为：

- “已经在本机完整跑完显式 Twin MATLAB”；
- “旧 `states_train/val/test.mat` 就是新 Twin 状态”；
- “已有旧连续状态指标就是老师最终方案结果”；
- “741个关系对等于741个独立训练样本”；
- “两个分支训练了两套输出权重”或“储备池内部参数已经通过反向传播训练”。
