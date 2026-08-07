"""Prepare the user's continuous-window YoY train45/test47 dataset.

The source is ``cpi_data_lastyear=100.csv`` and only its ``actual`` sequence
is used.  The legacy phase-space columns are deliberately ignored: every
sample is twelve consecutive months used to predict the following month.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PROCESSED_DIR, ROOT_DIR


PROFILE_ROOT = ROOT_DIR / "matlab" / "optical_reservoir_cpi_yoy_train45_noval_20260807"
DATA_DIR = PROFILE_ROOT / "data"
WINDOW_SIZE = 12
SPLITS = ("train", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matlab_strings(values: pd.Series) -> np.ndarray:
    return np.asarray(values.astype(str).tolist(), dtype=object).reshape(-1, 1)


def build_dataset() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    source = DATA_PROCESSED_DIR / "cpi_data_lastyear=100.csv"
    raw = pd.read_csv(source)
    raw["date"] = pd.to_datetime(
        raw["year"].astype(str) + "-" + raw["month"].astype(str) + "-01"
    )
    values = (
        raw.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")["actual"]
        .astype(float)
    )

    target_dates = pd.date_range("2018-09-01", "2026-04-01", freq="MS")
    rows: list[dict[str, object]] = []
    windows: list[np.ndarray] = []
    targets: list[float] = []
    for sample_id, target_date in enumerate(target_dates):
        window_dates = pd.date_range(
            target_date - pd.DateOffset(months=WINDOW_SIZE),
            target_date - pd.DateOffset(months=1),
            freq="MS",
        )
        required = [*window_dates, target_date]
        missing = [date for date in required if date not in values.index]
        if missing:
            raise ValueError(
                f"Missing YoY values for {target_date:%Y-%m}: "
                + ", ".join(date.strftime("%Y-%m") for date in missing)
            )
        split = "train" if target_date <= pd.Timestamp("2022-05-01") else "test"
        windows.append(values.loc[window_dates].to_numpy(dtype=np.float64))
        targets.append(float(values.loc[target_date]))
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "x_start_date": window_dates[0].strftime("%Y-%m"),
                "x_end_date": window_dates[-1].strftime("%Y-%m"),
                "target_date": target_date.strftime("%Y-%m"),
                "y": float(values.loc[target_date]),
            }
        )

    index = pd.DataFrame(rows)
    X = np.vstack(windows)
    y = np.asarray(targets, dtype=np.float64)
    train_mask = index["split"].eq("train").to_numpy()
    if int(train_mask.sum()) != 45 or int((~train_mask).sum()) != 47:
        raise AssertionError("Expected exactly 45 train and 47 test targets")

    # Fit transforms on the 45 training targets only.
    x_scaler = StandardScaler().fit(X[train_mask])
    y_scaler = StandardScaler().fit(y[train_mask].reshape(-1, 1))
    arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        mask = index["split"].eq(split).to_numpy()
        arrays[f"X_{split}"] = X[mask]
        arrays[f"y_{split}"] = y[mask]
        arrays[f"X_{split}_scaled"] = x_scaler.transform(X[mask])
        arrays[f"y_{split}_scaled"] = y_scaler.transform(y[mask].reshape(-1, 1)).ravel()
    arrays["window_size"] = np.asarray([[WINDOW_SIZE]], dtype=np.int32)
    arrays["x_mean"] = x_scaler.mean_
    arrays["x_scale"] = x_scaler.scale_
    arrays["y_mean"] = np.asarray([[float(y_scaler.mean_[0])]])
    arrays["y_scale"] = np.asarray([[float(y_scaler.scale_[0])]])
    return index, arrays


def main() -> None:
    index, arrays = build_dataset()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index.to_csv(DATA_DIR / "sample_index.csv", index=False)

    combined: dict[str, object] = dict(arrays)
    isolated_outputs: list[Path] = []
    split_metadata: dict[str, object] = {}
    for split in SPLITS:
        rows = index.loc[index["split"].eq(split)].reset_index(drop=True)
        sample_ids = rows["sample_id"].to_numpy(dtype=np.int32)
        for column in ("x_start_date", "x_end_date", "target_date"):
            combined[f"{column}_{split}"] = _matlab_strings(rows[column])
        combined[f"sample_id_{split}"] = sample_ids.reshape(-1, 1)

        rows.to_csv(DATA_DIR / f"sample_index_{split}.csv", index=False)
        np.save(DATA_DIR / f"X_{split}.npy", arrays[f"X_{split}"], allow_pickle=False)
        np.save(DATA_DIR / f"y_{split}.npy", arrays[f"y_{split}"], allow_pickle=False)
        mat_path = DATA_DIR / f"cpi_{split}_isolated.mat"
        savemat(
            mat_path,
            {
                "X": arrays[f"X_{split}"],
                "y": arrays[f"y_{split}"].reshape(-1, 1),
                "X_scaled": arrays[f"X_{split}_scaled"],
                "y_scaled": arrays[f"y_{split}_scaled"].reshape(-1, 1),
                "sample_id": sample_ids.reshape(-1, 1),
                "x_start_date": _matlab_strings(rows["x_start_date"]),
                "x_end_date": _matlab_strings(rows["x_end_date"]),
                "target_date": _matlab_strings(rows["target_date"]),
                "window_size": arrays["window_size"],
                "x_mean": arrays["x_mean"],
                "x_scale": arrays["x_scale"],
                "y_mean": arrays["y_mean"],
                "y_scale": arrays["y_scale"],
            },
            do_compression=True,
            long_field_names=True,
        )
        isolated_outputs.extend(
            [
                DATA_DIR / f"sample_index_{split}.csv",
                DATA_DIR / f"X_{split}.npy",
                DATA_DIR / f"y_{split}.npy",
                mat_path,
            ]
        )
        split_metadata[split] = {
            "num_samples": int(len(rows)),
            "first_target_date": str(rows["target_date"].iloc[0]),
            "last_target_date": str(rows["target_date"].iloc[-1]),
            "X_shape": list(arrays[f"X_{split}"].shape),
        }

    combined_path = DATA_DIR / "cpi_windows.mat"
    savemat(combined_path, combined, do_compression=True, long_field_names=True)
    source_path = DATA_PROCESSED_DIR / "cpi_data_lastyear=100.csv"
    metadata = {
        "source": str(source_path.relative_to(ROOT_DIR)),
        "source_column": "actual",
        "target_scale": "CPI YoY index (previous year same month=100)",
        "window_size": WINDOW_SIZE,
        "forecast_horizon_months": 1,
        "window_protocol": "continuous previous 12 actual values; no legacy phase-space columns",
        "split_method": "train45/test47 explicit target-date ranges; no validation split",
        "splits": split_metadata,
        "scaler_fit_scope": "train targets 2018-09 through 2022-05 only",
        "hyperparameter_policy": "train-only time-series cross-validation; test labels never used for selection",
    }
    (DATA_DIR / "cpi_windows_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "purpose": "user YoY continuous-window train/test isolation without validation",
        "source_files": {source_path.name: _sha256(source_path)},
        "splits": split_metadata,
        "outputs": {path.name: _sha256(path) for path in isolated_outputs},
        "combined_mat_sha256": _sha256(combined_path),
    }
    (DATA_DIR / "isolated_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
