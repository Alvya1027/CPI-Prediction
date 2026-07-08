# 处理后数据说明

本文件夹是传统 baseline、ESN 和后续光储备池模型共同使用的数据入口。

## 主数据文件

- `cpi_monthly.csv`：正式清洗后的 CPI 月度数据，后续模型优先读取这个文件。
- `cpi_monthly_checked.csv`：检查后保存的同结构副本，便于核对。

字段说明：

| 字段 | 含义 |
| --- | --- |
| `date` | 月份，格式为 `YYYY-MM` |
| `cpi` | 建模目标列，当前等于 `cpi_yoy` |
| `cpi_yoy_growth` | CPI 同比增长率，单位为百分点 |
| `cpi_yoy` | CPI 同比指数，以上年同月为 100 |

当前项目统一使用 `cpi` 作为预测目标列。后续同学写模型时，不需要再猜是用同比增长率还是 CPI 指数，直接读 `cpi` 即可。

## 默认滑动窗口数据集

建模任务：用过去 12 个月 CPI 预测未来 1 个月 CPI。

默认文件：

- `X_train.npy`、`y_train.npy`
- `X_val.npy`、`y_val.npy`
- `X_test.npy`、`y_test.npy`
- `sample_index.csv`

当前划分后的形状：

| 数据集 | X 形状 | y 形状 |
| --- | ---: | ---: |
| 训练集 | `(212, 12)` | `(212,)` |
| 验证集 | `(45, 12)` | `(45,)` |
| 测试集 | `(47, 12)` | `(47,)` |

数据按时间顺序划分，比例为训练集 70%、验证集 15%、测试集 15%。时间序列预测不能随机打乱，否则会造成未来信息泄漏。

## 样本索引文件

`sample_index.csv` 用来把 `.npy` 里的每一行样本对应回具体月份，方便后续画预测曲线、保存预测结果表和检查时间对齐。

字段说明：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 滑动窗口样本编号 |
| `split` | 所属数据集，取值为 `train`、`val` 或 `test` |
| `x_start_date` | 输入窗口的第一个月份 |
| `x_end_date` | 输入窗口的最后一个月份 |
| `target_date` | 需要预测的目标月份 |
| `y` | 目标 CPI 值 |

示例：第 0 个样本使用 `2000-01` 到 `2000-12` 的 CPI，预测 `2001-01` 的 CPI。

## 多窗口数据集

为了后续做窗口长度敏感性分析，已经额外生成了 6、12、24 个月窗口的数据：

- 6 个月窗口：文件名后缀为 `_ws6`，索引文件为 `sample_index_ws6.csv`。
- 12 个月窗口：文件名后缀为 `_ws12`，索引文件为 `sample_index_ws12.csv`。
- 24 个月窗口：文件名后缀为 `_ws24`，索引文件为 `sample_index_ws24.csv`。

第一周 baseline 建议优先使用默认 12 个月窗口，也就是不带 `_ws12` 后缀的 `X_train.npy`、`y_train.npy` 等文件。

## 推荐读取方式

后续写 baseline 时建议直接使用项目里的读取函数：

```python
from src.config import DATA_PROCESSED_DIR
from src.data_utils import load_window_dataset

data = load_window_dataset(DATA_PROCESSED_DIR)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
sample_index = data["sample_index"]
```

SVR、Linear Regression 等模型如果需要标准化，请只在 `X_train` 上拟合 scaler，然后再 transform `X_val` 和 `X_test`，不要用验证集或测试集参与 scaler 拟合。
