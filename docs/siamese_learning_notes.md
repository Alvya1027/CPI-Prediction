# 孪生光储备池回归笔记

## 1. 当前项目定义

本项目使用孪生结构完成连续 CPI 回归，不是相似/不相似二分类。

```text
目标窗口 x_i -> 共享光储备池 -> h_i --+
                                      +-> 回归读出层 -> delta_cpi_hat
参考窗口 x_j -> 共享光储备池 -> h_j --+

cpi_i_hat = cpi_j + delta_cpi_hat
```

真实标签为：

```text
delta_cpi = cpi_i - cpi_j
```

其中 `cpi_j` 是预测时已经公布的历史 CPI，`cpi_i` 是需要预测的目标 CPI。

## 2. 为什么不是相似度分类

原方案用阈值把样本对标为相似 1、不相似 0，并训练 Contrastive Loss。这种模型主要学习两个时期是否相似，不能直接给出具体 CPI 数值。

当前方案把连续差值 `delta_cpi` 作为主监督信号。相似标签可以保留，但只用于参考窗口分析或后期辅助任务，不是第一版模型的最终输出。

## 3. 共享分支是什么意思

两个分支必须使用相同的光储备池映射，包括：

- 相同的 `SL_RC.slx`；
- 相同的固定 mask；
- 相同的虚拟节点数、延迟反馈和输入增益；
- 相同的标准化和状态提取方法。

光储备池参数固定，不通过反向传播更新。每个窗口得到一个 50 维状态，只有后面的回归读出层需要训练。

工程上可以把每个唯一窗口的状态提前计算并缓存。样本对只按 `sample_i_id` 和 `sample_j_id` 读取两个状态，不必为每一对重复运行 Simulink。

## 4. 状态如何组合

第一版默认使用有符号差：

```text
z_ij = h_i - h_j
```

有符号差保留了 `i-j` 的方向，与 `cpi_i-cpi_j` 一致。后续可以对比：

- `[h_i-h_j, |h_i-h_j|]`；
- `[h_i, h_j]` 拼接；
- 多个参考窗口的加权聚合。

只使用 `|h_i-h_j|` 会丢失方向，因此不建议单独作为第一版输入。

## 5. 样本对构造

每对数据至少保存：

- 目标窗口和参考窗口的全局样本编号；
- 两个窗口的起止月份和目标月份；
- `cpi_i`、`cpi_j`；
- `delta_cpi`；
- 输入窗口距离；
- 可选的 `similar_label`。

训练集可以使用真实 `delta_cpi` 分层采样，使小、中、大差值都有足够训练样本。

验证集和测试集不能根据真实 `cpi_i` 或 `delta_cpi` 选择参考，否则会提前看到预测目标。当前使用预测时已知的两个历史窗口之间的距离选择参考。

## 6. 训练与预测

第一版使用 Ridge 回归读出层，训练目标是最小化差值均方误差：

```text
L_reg = mean((delta_cpi_hat - delta_cpi)^2)
```

对每个参考窗口都可以还原一个 CPI：

```text
cpi_i_hat_j = cpi_j + delta_cpi_hat_ij
```

使用多个参考窗口时，对这些结果求平均或按窗口距离加权，得到目标月份的最终预测。

评价指标必须按唯一目标月份计算 MAE、RMSE，不能把同一目标的多个参考对当成多个独立测试样本。

## 7. 相似标签和对比损失

`similar_label` 当前只用于：

1. 检查参考窗口是否具有相似趋势；
2. 分析不同参考类型对预测误差的影响；
3. 后期可能的辅助任务。

第一版不使用 Contrastive Loss。如果后期实验需要辅助相似任务，可以再考虑：

```text
L = L_reg + lambda * L_sim
```

但必须始终以连续 CPI 回归误差为主任务。

## 8. 关键泄漏边界

- `target_j_date` 必须早于 `target_i_date`。
- 参考 CPI 在预测目标月份时必须已经公布。
- 标准化和 mask 缩放只用训练集拟合。
- 验证和测试参考选择不能使用目标 CPI。
- 测试集不能参与 Ridge 参数选择。

## 9. 相关文献背景

1. Bromley, J. et al. “Signature verification using a Siamese time delay neural network.” NIPS, 1993.
2. Hadsell, R., Chopra, S. & LeCun, Y. “Dimensionality reduction by learning an invariant mapping.” CVPR, 2006.
3. Franceschi, J. Y. et al. “Unsupervised scalable representation learning for multivariate time series.” NeurIPS, 2019.

这些文献主要提供共享分支和度量学习背景。当前 CPI 项目的最终实现以孪生回归和连续差值预测为准。
