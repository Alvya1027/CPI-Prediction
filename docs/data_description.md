# CPI 数据说明

## 1. 数据用途

本数据集用于 CPI 小样本月度预测任务，是传统 baseline、ESN 和后续光储备池计算仿真的共同输入。

当前阶段的建模任务：

- 输入：过去 12 个月 CPI。
- 输出：未来 1 个月 CPI。
- 目标列：`cpi`。
- 评价指标：MAE、RMSE。由于当前 CPI 指数值在 100 附近，不接近 0，MAPE 可以作为参考指标使用。

## 2. 清洗后的月度数据

主文件：

```text
data_processed/cpi_monthly.csv
```

当前时间范围：

```text
2000-01 至 2026-04
```

当前检查结果：

- 共 316 条月度数据。
- 月份连续。
- 没有重复月份。
- 建模字段没有缺失值。
- CPI 指数范围约为 98.19 至 108.74。

字段说明：

| 字段 | 含义 |
| --- | --- |
| `date` | 月份，格式为 `YYYY-MM` |
| `cpi` | 建模目标列，当前等于 `cpi_yoy` |
| `cpi_yoy_growth` | CPI 同比增长率，单位为百分点 |
| `cpi_yoy` | CPI 同比指数，以上年同月为 100 |

后续模型统一读取 `cpi` 作为预测目标。

## 3. 原始数据来源

原始文件：

```text
data_raw/cpi_raw.xlsx
```

来源说明记录在 Excel 的 `来源说明` 工作表中。

需要注意：`2000-01` 至 `2005-12` 这一段是根据 MacroMicro 折线图进行像素读数估算得到的。第一周跑通数据流程和 baseline 对比可以使用，但正式论文或最终汇报前建议换成国家统计局、Wind、CEIC 等权威来源核验后的数据。

## 4. 数据清洗流程

清洗脚本：

```text
src/clean_cpi_data.py
```

处理步骤：

1. 读取 `data_raw/cpi_raw.xlsx` 中的 `CPI数据` 工作表。
2. 将月份统一为 `YYYY-MM`。
3. 统一 CPI 相关字段名。
4. 如果原始数据中有 `cpi_yoy_growth`，则计算 `cpi_yoy = 100 + cpi_yoy_growth`。
5. 设置 `cpi = cpi_yoy`，作为统一建模目标列。
6. 删除重复月份，保留第一次出现的记录。
7. 按月份升序排列。
8. 检查缺失值和月份连续性。
9. 保存到 `data_processed/cpi_monthly.csv`。

重新生成清洗数据：

```bash
python src/clean_cpi_data.py
```

## 5. 滑动窗口样本构造

默认生成脚本：

```bash
python src/create_datasets.py
```

默认参数：

| 参数 | 当前值 |
| --- | ---: |
| 输入窗口长度 | 12 |
| 预测步长 | 1 |
| 训练集比例 | 0.70 |
| 验证集比例 | 0.15 |
| 测试集比例 | 剩余 0.15 |

生成文件：

| 文件 | 含义 |
| --- | --- |
| `X_train.npy` | 训练集输入窗口 |
| `y_train.npy` | 训练集目标值 |
| `X_val.npy` | 验证集输入窗口 |
| `y_val.npy` | 验证集目标值 |
| `X_test.npy` | 测试集输入窗口 |
| `y_test.npy` | 测试集目标值 |
| `sample_index.csv` | 每个样本对应的日期索引 |

当前默认数据形状：

| 数据集 | X 形状 | y 形状 |
| --- | ---: | ---: |
| 训练集 | `(212, 12)` | `(212,)` |
| 验证集 | `(45, 12)` | `(45,)` |
| 测试集 | `(47, 12)` | `(47,)` |

划分方式严格按时间顺序，不随机打乱。

## 6. 样本索引说明

`sample_index.csv` 用于把模型预测结果对齐回真实月份。

字段说明：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 滑动窗口样本编号 |
| `split` | 所属数据集，取值为 `train`、`val` 或 `test` |
| `x_start_date` | 输入窗口起始月份 |
| `x_end_date` | 输入窗口结束月份 |
| `target_date` | 被预测的目标月份 |
| `y` | 目标 CPI 值 |

默认 12 个月窗口中，第 0 个样本使用 `2000-01` 至 `2000-12` 预测 `2001-01`。

## 7. 多窗口数据

如果要比较不同输入窗口长度，可以运行：

```bash
python src/generate_datasets_for_windows.py
```

当前已经生成：

- `_ws6`：6 个月窗口，共 310 个样本。
- `_ws12`：12 个月窗口，共 304 个样本。
- `_ws24`：24 个月窗口，共 292 个样本。

每个窗口长度都有对应的 `X`、`y` 和 `sample_index` 文件。

## 8. 给 baseline 组的读取方式

建议 baseline 代码直接这样读取默认数据：

```python
from src.config import DATA_PROCESSED_DIR
from src.data_utils import load_window_dataset

data = load_window_dataset(DATA_PROCESSED_DIR)
```

SVR 等对尺度敏感的模型需要标准化时，只能用训练集拟合标准化器，再应用到验证集和测试集，避免未来信息泄漏。
