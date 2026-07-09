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

## 标准化数据集（`_scaled` 后缀）

为了避免每个模型重复写标准化代码，已经预先生成了标准化后的数据，文件命名规则：在原始文件名中的后缀前插入 `_scaled`。

默认 12 个月窗口的标准化文件：

| 文件 | 说明 |
| --- | --- |
| `X_train_scaled.npy` | 标准化训练集特征（z-score） |
| `y_train_scaled.npy` | 标准化训练集标签 |
| `X_val_scaled.npy` | 标准化验证集特征 |
| `y_val_scaled.npy` | 标准化验证集标签 |
| `X_test_scaled.npy` | 标准化测试集特征 |
| `y_test_scaled.npy` | 标准化测试集标签 |
| `scaler_params_scaled.json` | 标准化器参数（供逆变换和复现） |

多窗口版本同样可用：`_scaled_ws6`、`_scaled_ws12`、`_scaled_ws24`。

### 标准化方法

使用 **StandardScaler（z-score 标准化）**：

$$X_{scaled} = \frac{X - \mu_{train}}{\sigma_{train}}$$

其中 $\mu_{train}$ 和 $\sigma_{train}$ 只在训练集上计算，验证集和测试集使用相同的参数 transform。

### 关键原则

- **只用训练集拟合标准化器**：验证集和测试集不参与参数计算，避免数据泄漏
- **不覆盖原始数据**：原始 `.npy` 文件保留不变，标准化版本另存
- **参数可复现**：`scaler_params_scaled.json` 中记录了 X_mean、X_std、y_mean、y_std，预测时可用 `inverse_transform_y()` 还原到原始 CPI 尺度

### 推荐读取方式

```python
from src.data_utils import load_scaled_dataset, inverse_transform_y

# 加载标准化数据
data = load_scaled_dataset(DATA_PROCESSED_DIR)
X_train = data["X_train"]  # 标准化后的数据
y_train = data["y_train"]
scaler = data["scaler_params"]

# 模型训练...
# y_pred_scaled = model.predict(X_test)

# 预测值还原为原始 CPI 尺度
y_pred = inverse_transform_y(y_pred_scaled, scaler)
```

### 示例：scaler_params_scaled.json 内容

```json
{
  "scaler_type": "standard",
  "X_mean": 102.2127,
  "X_std": 2.0732,
  "y_mean": 102.2642,
  "y_std": 2.0468
}
```

### 生成标准化数据

如果原始数据有更新，重新运行：

```bash
python -m src.create_scaled_datasets
```

## 同比 / 环比一维数据集

为了满足 Naive、Auto-Regression 等简单 baseline 的需求，额外生成了按月份分组的一维数据。所有数据均为 1D 数组，按时间顺序切分训练集/验证集/测试集（70%/15%/15%）。

### 同比数据（`cpi_m01` ~ `cpi_m12`）

每个文件包含该月份所有年份的 CPI 值，按时间顺序排列。每组 3 个文件：

| 文件 | 说明 | 点数 |
|------|------|:---:|
| `cpi_m01_train.npy` | 1月训练集 | 18 |
| `cpi_m01_val.npy` | 1月验证集 | 4 |
| `cpi_m01_test.npy` | 1月测试集 | 5 |
| … | 2月~12月同理 | … |
| `cpi_m12_train.npy` | 12月训练集 | 18 |
| `cpi_m12_val.npy` | 12月验证集 | 3 |
| `cpi_m12_test.npy` | 12月测试集 | 5 |

标准化版本：文件名插入 `_scaled`（如 `cpi_m01_train_scaled.npy`），标准化参数独立保存为 `scaler_params_m01.json`。每个月份独立拟合 scaler，因为不同月份 CPI 的分布特征不同。

### 环比数据（`cpi_seq`）

所有 CPI 按时间顺序排成一维序列：

| 文件 | 说明 | 点数 |
|------|------|:---:|
| `cpi_seq_train.npy` | 训练集 | 221 |
| `cpi_seq_val.npy` | 验证集 | 47 |
| `cpi_seq_test.npy` | 测试集 | 48 |

同样有 `_scaled` 标准化版本和 `scaler_params_seq.json`。

### 推荐读取方式

```python
import numpy as np

# 同比：读取 1 月数据
train = np.load("data_processed/cpi_m01_train.npy")
val = np.load("data_processed/cpi_m01_val.npy")
test = np.load("data_processed/cpi_m01_test.npy")

# 环比：读取全序列
train = np.load("data_processed/cpi_seq_train.npy")
val = np.load("data_processed/cpi_seq_val.npy")
test = np.load("data_processed/cpi_seq_test.npy")
```

### 生成数据

```bash
python -m src.create_monthly_datasets
```
