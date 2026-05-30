import numpy as np
import pandas as pd
from typing import Tuple, Optional


def create_sliding_window(
        series: pd.Series,
        window_size: int = 12,
        horizon: int = 1,
        target_column: str = "cpi"
) -> Tuple[np.ndarray, np.ndarray]:
    """
        将时间序列转换为滑动窗口格式的监督学习样本。

        参数:
            series: 一维时间序列（pandas Series 或 numpy 数组）
            window_size: 输入窗口长度（使用过去多少个点）
            horizon: 预测步长（预测未来第几个点，默认1）

        返回:
            X: 特征数组，形状为 (样本数, window_size)
            y: 标签数组，形状为 (样本数,)
        """
    # 确保数据是numpy数组
    data = np.array(series)

    X, y = [], []
    # 计算能生成多少个样本
    num_samples = len(data) - window_size - horizon + 1

    for i in range(num_samples):
        # X是窗口内的数据
        X.append(data[i:i + window_size])
        # y是窗口后第'horizon'个点的数据
        y.append(data[i + window_size + horizon - 1])

    return np.array(X), np.array(y)


def split_sequence(
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按时间顺序划分训练集、验证集和测试集"""
    n_samples = len(X)
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test
