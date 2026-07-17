# CPI 月间比值数据说明

## 数据含义

`cpi_month_ratio.csv` 是根据项目现有的 CPI 同比指数计算出的相邻月份比值：

```text
cpi_month_ratio = 本月 cpi_yoy / 上月 cpi_yoy * 100
```

例如结果为 `100.3009`，表示本月的 CPI 同比指数是上月的约 `100.3009%`。

注意：这个字段是“现有 CPI 同比指数的相邻月份比值”，不等于国家统计局发布的“居民消费价格环比指数（上月=100）”。如果后续取得官方环比数据，应单独保存并替换建模目标，不能将两者混用。

## CSV 字段

| 字段 | 含义 |
| --- | --- |
| `date` | 本月，格式为 `YYYY-MM` |
| `cpi_yoy` | 本月 CPI 同比指数 |
| `previous_date` | 上一个月 |
| `previous_cpi_yoy` | 上月 CPI 同比指数 |
| `cpi_month_ratio` | 本月值与上月值的比值，以 100 为基准 |

`2000-01` 是现有序列的第一个月，没有上月数据，所以该行比值为空。模型数据从 `2000-02` 开始。

## 可直接建模的文件

以下文件使用过去 12 个月比值预测未来 1 个月比值，并按时间顺序以 70%/15%/15% 划分：

| 文件 | 形状或用途 |
| --- | --- |
| `X_train_month_ratio.npy` | `(212, 12)` |
| `y_train_month_ratio.npy` | `(212,)` |
| `X_val_month_ratio.npy` | `(45, 12)` |
| `y_val_month_ratio.npy` | `(45,)` |
| `X_test_month_ratio.npy` | `(46, 12)` |
| `y_test_month_ratio.npy` | `(46,)` |
| `sample_index_month_ratio.csv` | 每个样本对应的输入月份、目标月份和数据集划分 |

模型读取示例：

```python
from src.config import DATA_PROCESSED_DIR
from src.data_utils import load_window_dataset

data = load_window_dataset(DATA_PROCESSED_DIR, suffix="_month_ratio")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
```

## 重新生成

现有 CPI 数据更新后运行：

```bash
python -m src.create_cpi_month_ratio_dataset
```
