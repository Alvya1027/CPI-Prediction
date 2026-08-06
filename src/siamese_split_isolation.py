"""Physically isolated split loaders for strict pre-test experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def load_isolated_split(data_dir: Path, split: str) -> dict[str, object]:
    """Load one split without opening any file containing another split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    index_path = data_dir / f"sample_index_{split}.csv"
    x_path = data_dir / f"X_{split}.npy"
    y_path = data_dir / f"y_{split}.npy"
    missing = [path for path in (index_path, x_path, y_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "isolated split files are missing: "
            + ", ".join(str(path) for path in missing)
        )
    index = pd.read_csv(index_path).reset_index(drop=True)
    X = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False).reshape(-1)
    if not index["split"].eq(split).all():
        raise ValueError(f"{index_path} contains a row outside split={split}")
    if len(X) != len(y) or len(X) != len(index):
        raise ValueError(f"isolated {split} split is misaligned")
    if not np.array_equal(
        index["sample_id"].to_numpy(dtype=int),
        np.sort(index["sample_id"].to_numpy(dtype=int)),
    ):
        raise ValueError(f"isolated {split} IDs are not chronological")
    if not np.allclose(y, index["y"].to_numpy(dtype=float)):
        raise ValueError(f"isolated {split} labels do not match the index")
    return {"X": X, "y": y, "index": index}


def build_isolated_closed_train_pool(
    data_dir: Path,
    count: int = 50,
) -> dict[str, object]:
    """Select the latest count targets from the physically isolated train split."""
    train = load_isolated_split(data_dir, "train")
    index = train["index"].copy()
    if len(index) < count:
        raise ValueError(f"train has only {len(index)} targets, fewer than {count}")
    dates = pd.PeriodIndex(index["target_date"].astype(str), freq="M")
    order = np.argsort(dates.asi8, kind="stable")
    selected_positions = np.asarray(order[-count:], dtype=int)
    selected_positions.sort()
    return {
        "X": np.asarray(train["X"])[selected_positions],
        "y": np.asarray(train["y"])[selected_positions],
        "index": index.iloc[selected_positions].reset_index(drop=True),
    }
