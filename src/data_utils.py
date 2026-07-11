from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def create_sliding_window(
    series: pd.Series,
    window_size: int = 12,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 1-D time series into supervised learning samples."""
    data = np.asarray(series, dtype=float)
    num_samples = len(data) - window_size - horizon + 1
    if num_samples <= 0:
        raise ValueError(
            "Series is too short for the requested window_size and horizon."
        )

    X, y = [], []
    for i in range(num_samples):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size + horizon - 1])

    return np.asarray(X), np.asarray(y)


def create_sample_index(
    dates: pd.Series,
    y: np.ndarray,
    window_size: int = 12,
    horizon: int = 1,
) -> pd.DataFrame:
    """Create date metadata for each sliding-window sample."""
    date_values = pd.to_datetime(dates).dt.strftime("%Y-%m").to_numpy()
    rows = []

    for i, target in enumerate(y):
        target_pos = i + window_size + horizon - 1
        rows.append(
            {
                "sample_id": i,
                "x_start_date": date_values[i],
                "x_end_date": date_values[i + window_size - 1],
                "target_date": date_values[target_pos],
                "y": float(target),
            }
        )

    return pd.DataFrame(rows)


def split_sequence(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split samples in chronological order into train, validation, and test sets."""
    n_samples = len(X)
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_sample_index(
    sample_index: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> pd.DataFrame:
    """Add chronological split labels to a sample-index table."""
    n_samples = len(sample_index)
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)

    indexed = sample_index.copy()
    indexed["split"] = "test"
    indexed.loc[: train_end - 1, "split"] = "train"
    indexed.loc[train_end : val_end - 1, "split"] = "val"

    return indexed[
        ["sample_id", "split", "x_start_date", "x_end_date", "target_date", "y"]
    ]


def save_window_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    window_size: int = 12,
    horizon: int = 1,
    target_col: str = "cpi",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    suffix: str = "",
) -> dict:
    """Create, split, and save one sliding-window dataset."""
    if target_col not in df.columns:
        if target_col == "cpi" and "cpi_yoy" in df.columns:
            target_col = "cpi_yoy"
        else:
            raise ValueError(f"Missing target column: {target_col}")
    if "date" not in df.columns:
        raise ValueError("Missing date column.")

    X, y = create_sliding_window(
        df[target_col], window_size=window_size, horizon=horizon
    )
    X_train, y_train, X_val, y_val, X_test, y_test = split_sequence(
        X, y, train_ratio=train_ratio, val_ratio=val_ratio
    )

    sample_index = create_sample_index(
        df["date"], y, window_size=window_size, horizon=horizon
    )
    sample_index = split_sample_index(
        sample_index, train_ratio=train_ratio, val_ratio=val_ratio
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"X_train{suffix}.npy", X_train)
    np.save(output_dir / f"y_train{suffix}.npy", y_train)
    np.save(output_dir / f"X_val{suffix}.npy", X_val)
    np.save(output_dir / f"y_val{suffix}.npy", y_val)
    np.save(output_dir / f"X_test{suffix}.npy", X_test)
    np.save(output_dir / f"y_test{suffix}.npy", y_test)
    sample_index.to_csv(
        output_dir / f"sample_index{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "window_size": window_size,
        "horizon": horizon,
        "n_samples": len(X),
        "train": X_train.shape,
        "val": X_val.shape,
        "test": X_test.shape,
    }


def load_window_dataset(data_dir: Path, suffix: str = "") -> dict:
    """Load saved X/y arrays and sample index for model scripts."""
    return {
        "X_train": np.load(data_dir / f"X_train{suffix}.npy"),
        "y_train": np.load(data_dir / f"y_train{suffix}.npy"),
        "X_val": np.load(data_dir / f"X_val{suffix}.npy"),
        "y_val": np.load(data_dir / f"y_val{suffix}.npy"),
        "X_test": np.load(data_dir / f"X_test{suffix}.npy"),
        "y_test": np.load(data_dir / f"y_test{suffix}.npy"),
        "sample_index": pd.read_csv(data_dir / f"sample_index{suffix}.csv"),
    }


# ============================================================
# 标准化 / 缩放函数
# 原则：只用训练集拟合标准化器，再用同一个标准化器 transform
#       验证集和测试集，避免任何数据泄漏。
# ============================================================


def fit_scaler(
    X_train: NDArray,
    y_train: NDArray,
    scaler_type: str = "standard",
) -> Dict[str, object]:
    """在训练集上拟合标准化器，返回包含 scaler 参数的字典。

    参数：
        X_train: 训练集特征，形状 (n_samples, window_size)
        y_train: 训练集标签，形状 (n_samples,)
        scaler_type: "standard" → z-score 标准化（默认）
                     "minmax"  → MinMax 缩放到 [0, 1]

    返回：
        dict: {
            "X_mean": float,   # X 的均值（standard 模式）
            "X_std": float,    # X 的标准差（standard 模式）
            "X_min": float,    # X 的最小值（minmax 模式）
            "X_max": float,    # X 的最大值（minmax 模式）
            "y_mean": float,
            "y_std": float,
            "y_min": float,
            "y_max": float,
            "scaler_type": str,
        }
    """
    # 将 X 展平为一维，因为所有时间步的 CPI 值来自同一分布
    X_flat = X_train.ravel()

    params = {"scaler_type": scaler_type}

    if scaler_type == "standard":
        params["X_mean"] = float(np.mean(X_flat))
        params["X_std"] = float(np.std(X_flat))
        params["y_mean"] = float(np.mean(y_train))
        params["y_std"] = float(np.std(y_train))
    elif scaler_type == "minmax":
        params["X_min"] = float(np.min(X_flat))
        params["X_max"] = float(np.max(X_flat))
        params["y_min"] = float(np.min(y_train))
        params["y_max"] = float(np.max(y_train))
    else:
        raise ValueError(
            f"scaler_type 必须是 'standard' 或 'minmax'，收到: {scaler_type}"
        )

    return params


def apply_scaler(
    X: NDArray,
    y: Optional[NDArray],
    scaler_params: Dict[str, object],
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    """用已拟合的标准化器参数 transform 数据。

    参数：
        X: 特征数组，形状 (n_samples, window_size)
        y: 标签数组，形状 (n_samples,)，如果为 None 则只 transform X
        scaler_params: fit_scaler 返回的参数字典

    返回：
        如果 y 为 None: 只返回 X_scaled
        否则: 返回 (X_scaled, y_scaled)
    """
    scaler_type = scaler_params["scaler_type"]

    if scaler_type == "standard":
        X_scaled = (X - scaler_params["X_mean"]) / scaler_params["X_std"]
        if y is None:
            return X_scaled
        y_scaled = (y - scaler_params["y_mean"]) / scaler_params["y_std"]
    elif scaler_type == "minmax":
        X_range = scaler_params["X_max"] - scaler_params["X_min"]
        X_scaled = (X - scaler_params["X_min"]) / X_range
        if y is None:
            return X_scaled
        y_range = scaler_params["y_max"] - scaler_params["y_min"]
        y_scaled = (y - scaler_params["y_min"]) / y_range
    else:
        raise ValueError(
            f"scaler_type 必须是 'standard' 或 'minmax'，收到: {scaler_type}"
        )

    if y is None:
        return X_scaled
    return X_scaled, y_scaled


def inverse_transform_y(
    y_scaled: NDArray,
    scaler_params: Dict[str, object],
) -> NDArray:
    """将标准化后的 y 预测值还原为原始 CPI 尺度。

    参数：
        y_scaled: 标准化后的预测值
        scaler_params: 拟合时保存的参数字典

    返回：
        原始 CPI 尺度的预测值
    """
    scaler_type = scaler_params["scaler_type"]

    if scaler_type == "standard":
        return y_scaled * scaler_params["y_std"] + scaler_params["y_mean"]
    elif scaler_type == "minmax":
        y_range = scaler_params["y_max"] - scaler_params["y_min"]
        return y_scaled * y_range + scaler_params["y_min"]
    else:
        raise ValueError(
            f"scaler_type 必须是 'standard' 或 'minmax'，收到: {scaler_type}"
        )


def save_scaled_dataset(
    data_dir: Path,
    suffix: str = "",
    scaler_type: str = "standard",
    output_suffix: str = "_scaled",
) -> Dict[str, object]:
    """加载原始 .npy 数据，在训练集上拟合标准化器，保存标准化版本。

    原则：
    1. 只在训练集上拟合标准化器参数
    2. 用同一个标准化器 transform 训练集、验证集、测试集
    3. 标准化后的数据另存为带 _scaled 后缀的新文件，不覆盖原始数据
    4. 标准化器参数保存为 JSON 文件，供后续模型复现

    参数：
        data_dir: 数据目录路径
        suffix: 原始数据文件后缀（如 "", "_ws6", "_ws12", "_ws24"）
        scaler_type: 标准化类型，"standard" 或 "minmax"
        output_suffix: 输出文件后缀，默认 "_scaled"

    返回：
        dict: 包含 scaler_params 和各数据集形状
    """
    # 加载原始数据
    X_train = np.load(data_dir / f"X_train{suffix}.npy")
    y_train = np.load(data_dir / f"y_train{suffix}.npy")
    X_val = np.load(data_dir / f"X_val{suffix}.npy")
    y_val = np.load(data_dir / f"y_val{suffix}.npy")
    X_test = np.load(data_dir / f"X_test{suffix}.npy")
    y_test = np.load(data_dir / f"y_test{suffix}.npy")

    # 1. 只在训练集上拟合标准化器
    scaler_params = fit_scaler(X_train, y_train, scaler_type=scaler_type)

    # 2. 用同一个标准化器 transform 所有数据集
    X_train_scaled, y_train_scaled = apply_scaler(
        X_train, y_train, scaler_params
    )
    X_val_scaled, y_val_scaled = apply_scaler(
        X_val, y_val, scaler_params
    )
    X_test_scaled, y_test_scaled = apply_scaler(
        X_test, y_test, scaler_params
    )

    # 3. 保存标准化后的数据
    np.save(data_dir / f"X_train{output_suffix}{suffix}.npy", X_train_scaled)
    np.save(data_dir / f"y_train{output_suffix}{suffix}.npy", y_train_scaled)
    np.save(data_dir / f"X_val{output_suffix}{suffix}.npy", X_val_scaled)
    np.save(data_dir / f"y_val{output_suffix}{suffix}.npy", y_val_scaled)
    np.save(data_dir / f"X_test{output_suffix}{suffix}.npy", X_test_scaled)
    np.save(data_dir / f"y_test{output_suffix}{suffix}.npy", y_test_scaled)

    # 4. 保存标准化器参数为 JSON
    import json

    scaler_file = data_dir / f"scaler_params{output_suffix}{suffix}.json"
    with open(scaler_file, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, ensure_ascii=False, indent=2)

    return {
        "scaler_params": scaler_params,
        "scaler_file": str(scaler_file),
        "train": X_train_scaled.shape,
        "val": X_val_scaled.shape,
        "test": X_test_scaled.shape,
    }


def load_scaled_dataset(
    data_dir: Path,
    suffix: str = "",
    output_suffix: str = "_scaled",
) -> dict:
    """加载标准化后的 X/y 数组、sample_index 和 scaler_params。

    参数：
        data_dir: 数据目录路径
        suffix: 原始数据文件后缀
        output_suffix: 标准化数据文件后缀，默认 "_scaled"

    返回：
        dict: 包含标准化后的 X_train, y_train, ... 以及 scaler_params
    """
    import json

    data = load_window_dataset(data_dir, suffix=suffix)
    scaler_file = data_dir / f"scaler_params{output_suffix}{suffix}.json"

    if scaler_file.exists():
        with open(scaler_file, "r", encoding="utf-8") as f:
            scaler_params = json.load(f)
    else:
        scaler_params = None

    return {
        **data,
        "scaler_params": scaler_params,
    }
