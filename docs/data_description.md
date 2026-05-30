# CPI 数据说明

## 1. 数据名称

中国月度居民消费价格指数 CPI 数据。

## 2. 数据用途

本数据用于 CPI 小样本时间序列预测任务。当前阶段主要用于：

1. 绘制 CPI 原始时间序列图；
2. 检查缺失值和日期连续性；
3. 后续构造滑动窗口数据集；
4. 为 Naive Forecast、Moving Average、Ridge、SVR、ARIMA、LSTM 以及储备池模型提供输入数据。

## 3. 数据时间范围

当前数据时间范围为：

```text
2000-01 至 2026-04

## 数据集划分说明（用于预测模型）

### 滑动窗口参数
- **窗口大小 (window_size)**：12（默认，用过去12个月预测未来1个月）
- **预测步长 (horizon)**：1（预测下一个月）
- 代码已支持灵活调整窗口大小（如6、24），可通过修改 `WINDOW_SIZE` 变量重新生成。

### 数据集划分比例
- 训练集 (Training set)：70%
- 验证集 (Validation set)：15%
- 测试集 (Test set)：15%
- **划分原则**：严格按照时间顺序，**不使用随机打乱**，避免未来信息泄露。

### 生成的文件（位于 `data_processed/` 目录）
- `X_train.npy` : 训练集特征，形状 (样本数, window_size)
- `y_train.npy` : 训练集标签，形状 (样本数,)
- `X_val.npy`   : 验证集特征
- `y_val.npy`   : 验证集标签
- `X_test.npy`  : 测试集特征
- `y_test.npy`  : 测试集标签

若生成多窗口数据集，文件名会包含 `_ws6`、`_ws12`、`_ws24` 等后缀。

### 生成数据集的代码
- 主脚本：`create_datasets.py`（或 `generate_datasets_for_windows.py`）
- 依赖函数：`src/data_utils.py` 中的 `create_sliding_window` 和 `split_sequence`