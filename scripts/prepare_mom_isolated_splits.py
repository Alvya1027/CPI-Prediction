"""Materialize independent train/val/test files for strict load isolation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.siamese_shared_projection import DATA_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source_index = pd.read_csv(DATA_DIR / "sample_index.csv")
    payload = loadmat(DATA_DIR / "cpi_windows.mat")
    outputs: list[Path] = []
    summary: dict[str, object] = {}
    for split in ("train", "val", "test"):
        index = source_index.loc[source_index["split"].eq(split)].reset_index(
            drop=True
        )
        X = np.asarray(payload[f"X_{split}"], dtype=np.float64)
        y = np.asarray(payload[f"y_{split}"], dtype=np.float64).reshape(-1)
        sample_ids = np.asarray(payload[f"sample_id_{split}"], dtype=int).reshape(-1)
        if len(index) != len(X) or len(index) != len(y):
            raise ValueError(f"source {split} split is misaligned")
        if not np.array_equal(sample_ids, index["sample_id"].to_numpy(dtype=int)):
            raise ValueError(f"source {split} sample IDs disagree")
        if not np.allclose(y, index["y"].to_numpy(dtype=float)):
            raise ValueError(f"source {split} labels disagree")
        index_path = DATA_DIR / f"sample_index_{split}.csv"
        x_path = DATA_DIR / f"X_{split}.npy"
        y_path = DATA_DIR / f"y_{split}.npy"
        matlab_path = DATA_DIR / f"cpi_{split}_isolated.mat"
        index.to_csv(index_path, index=False)
        np.save(x_path, X, allow_pickle=False)
        np.save(y_path, y, allow_pickle=False)
        savemat(
            matlab_path,
            {
                "X": X,
                "y": y.reshape(-1, 1),
                "X_scaled": np.asarray(
                    payload[f"X_{split}_scaled"], dtype=np.float64
                ),
                "y_scaled": np.asarray(
                    payload[f"y_{split}_scaled"], dtype=np.float64
                ).reshape(-1, 1),
                "sample_id": sample_ids.reshape(-1, 1),
                "x_start_date": index["x_start_date"].astype(str).to_numpy(
                    dtype=object
                ).reshape(-1, 1),
                "x_end_date": index["x_end_date"].astype(str).to_numpy(
                    dtype=object
                ).reshape(-1, 1),
                "target_date": index["target_date"].astype(str).to_numpy(
                    dtype=object
                ).reshape(-1, 1),
                "window_size": np.asarray([[X.shape[1]]], dtype=np.int32),
                "x_mean": np.asarray(payload["x_mean"], dtype=np.float64),
                "x_scale": np.asarray(payload["x_scale"], dtype=np.float64),
                "y_mean": np.asarray(payload["y_mean"], dtype=np.float64),
                "y_scale": np.asarray(payload["y_scale"], dtype=np.float64),
            },
            do_compression=True,
            long_field_names=True,
        )
        outputs.extend((index_path, x_path, y_path, matlab_path))
        summary[split] = {
            "num_targets": int(len(index)),
            "first_target_date": str(index["target_date"].iloc[0]),
            "last_target_date": str(index["target_date"].iloc[-1]),
            "X_shape": list(X.shape),
        }
    manifest = {
        "purpose": "physical split isolation for pre-test experiment loading",
        "source_files": {
            "sample_index.csv": _sha256(DATA_DIR / "sample_index.csv"),
            "cpi_windows.mat": _sha256(DATA_DIR / "cpi_windows.mat"),
        },
        "splits": summary,
        "outputs": {
            path.name: _sha256(path) for path in outputs
        },
    }
    (DATA_DIR / "isolated_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
