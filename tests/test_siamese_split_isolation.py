from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.siamese_shared_projection import DATA_DIR
from src.siamese_split_isolation import load_isolated_split


def test_train_and_validation_loaders_never_open_combined_or_test_files(
    monkeypatch,
) -> None:
    opened: list[str] = []
    original_read_csv = pd.read_csv
    original_np_load = np.load

    def guarded_read_csv(path: str | Path, *args, **kwargs):
        name = Path(path).name
        opened.append(name)
        assert name not in {"sample_index.csv", "sample_index_test.csv"}
        return original_read_csv(path, *args, **kwargs)

    def guarded_np_load(path: str | Path, *args, **kwargs):
        name = Path(path).name
        opened.append(name)
        assert "test" not in name
        return original_np_load(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr(np, "load", guarded_np_load)

    train = load_isolated_split(DATA_DIR, "train")
    validation = load_isolated_split(DATA_DIR, "val")

    assert len(train["index"]) == 50
    assert len(validation["index"]) == 45
    assert opened == [
        "sample_index_train.csv",
        "X_train.npy",
        "y_train.npy",
        "sample_index_val.csv",
        "X_val.npy",
        "y_val.npy",
    ]
