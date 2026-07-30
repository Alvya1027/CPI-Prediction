"""Prepare an isolated 12-month MoM window dataset for the optical reservoir.

The official ``cpi_data_lastmonth=100.csv`` file is used as the source of the
target sequence.  This script does not overwrite the shared ``.npy`` files or
the existing YoY reservoir inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PROCESSED_DIR, ROOT_DIR


PROFILE_ROOT = ROOT_DIR / "matlab" / "optical_reservoir_cpi_mom_recent50_20260730"
DATA_DIR = PROFILE_ROOT / "data"
WINDOW_SIZE = 12


def _matlab_strings(values: pd.Series) -> np.ndarray:
    return np.asarray(values.astype(str).tolist(), dtype=object).reshape(-1, 1)


def build_dataset() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    raw = pd.read_csv(DATA_PROCESSED_DIR / "cpi_data_lastmonth=100.csv")
    raw["date"] = pd.to_datetime(
        raw["year"].astype(str) + "-" + raw["month"].astype(str) + "-01"
    )
    raw = raw.sort_values("date").reset_index(drop=True)
    values = raw.set_index("date")["actual"].astype(float)

    target_dates = pd.date_range("2014-07-01", "2026-04-01", freq="MS")
    rows: list[dict[str, object]] = []
    windows: list[np.ndarray] = []
    targets: list[float] = []
    for sample_id, target_date in enumerate(target_dates):
        window_dates = pd.date_range(
            target_date - pd.DateOffset(months=WINDOW_SIZE),
            target_date - pd.DateOffset(months=1),
            freq="MS",
        )
        if not all(date in values.index for date in window_dates):
            raise ValueError(f"Missing MoM history for target {target_date:%Y-%m}")
        if target_date not in values.index:
            raise ValueError(f"Missing MoM target {target_date:%Y-%m}")

        if target_date <= pd.Timestamp("2018-08-01"):
            split = "train"
        elif target_date <= pd.Timestamp("2022-05-01"):
            split = "val"
        else:
            split = "test"

        windows.append(values.loc[window_dates].to_numpy(dtype=float))
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
    y = np.asarray(targets, dtype=float)
    arrays: dict[str, np.ndarray] = {}
    x_scaler = StandardScaler().fit(X[index["split"].eq("train")])
    y_scaler = StandardScaler().fit(y[index["split"].eq("train")].reshape(-1, 1))

    for split in ("train", "val", "test"):
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

    payload: dict[str, object] = {key: value for key, value in arrays.items()}
    for split in ("train", "val", "test"):
        rows = index.loc[index["split"].eq(split)].reset_index(drop=True)
        payload[f"sample_id_{split}"] = rows[["sample_id"]].to_numpy(dtype=np.int32)
        for column in ("x_start_date", "x_end_date", "target_date"):
            payload[f"{column}_{split}"] = _matlab_strings(rows[column])
    savemat(DATA_DIR / "cpi_windows.mat", payload, do_compression=True, long_field_names=True)

    split_metadata = {}
    for split in ("train", "val", "test"):
        rows = index.loc[index["split"].eq(split)]
        split_metadata[split] = {
            "num_samples": int(len(rows)),
            "first_target_date": str(rows["target_date"].iloc[0]),
            "last_target_date": str(rows["target_date"].iloc[-1]),
        }
    metadata = {
        "source": "data_processed/cpi_data_lastmonth=100.csv",
        "target_scale": "CPI MoM index (previous month=100)",
        "window_size": WINDOW_SIZE,
        "forecast_horizon_months": 1,
        "split_method": "explicit target-date ranges",
        "splits": split_metadata,
        "scaler_fit_scope": "train targets 2014-07 through 2018-08 only",
    }
    (DATA_DIR / "cpi_windows_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"MoM MAT 数据已生成：{DATA_DIR / 'cpi_windows.mat'}")


if __name__ == "__main__":
    main()
