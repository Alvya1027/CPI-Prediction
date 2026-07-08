from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


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
