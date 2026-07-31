from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from src.siamese_closed50_experiment import _load_split


def test_load_split_falls_back_to_mat(tmp_path: Path) -> None:
    index = pd.DataFrame(
        {
            "sample_id": [0, 1],
            "split": ["train", "train"],
            "x_start_date": ["2013-07", "2013-08"],
            "x_end_date": ["2014-06", "2014-07"],
            "target_date": ["2014-07", "2014-08"],
            "y": [100.1, 99.9],
        }
    )
    index.to_csv(tmp_path / "sample_index.csv", index=False)
    X = np.arange(24, dtype=float).reshape(2, 12)
    y = np.asarray([[100.1], [99.9]])
    savemat(tmp_path / "cpi_windows.mat", {"X_train": X, "y_train": y})

    split = _load_split(tmp_path, "train")

    assert np.array_equal(split["X"], X)
    assert np.allclose(split["y"], y.reshape(-1))
    assert split["index"]["sample_id"].tolist() == [0, 1]
