"""Export the baseline CPI windows to a MATLAB-compatible MAT file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from src.config import DATA_PROCESSED_DIR, DEFAULT_WINDOW_SIZE, ROOT_DIR


SPLITS = ("train", "val", "test")
OUTPUT_DIR = ROOT_DIR / "matlab" / "optical_reservoir_cpi" / "data"


def _as_column(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(-1, 1)


def _as_matlab_strings(values: pd.Series) -> np.ndarray:
    return np.asarray(values.astype(str).tolist(), dtype=object).reshape(-1, 1)


def export_cpi_windows(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    """Export existing chronological splits without rebuilding the windows."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_index = pd.read_csv(DATA_PROCESSED_DIR / "sample_index.csv")
    required_columns = {
        "sample_id",
        "split",
        "x_start_date",
        "x_end_date",
        "target_date",
        "y",
    }
    missing_columns = required_columns.difference(sample_index.columns)
    if missing_columns:
        raise ValueError(f"sample_index.csv is missing columns: {sorted(missing_columns)}")

    payload: dict[str, np.ndarray | int] = {
        "window_size": np.asarray([[DEFAULT_WINDOW_SIZE]], dtype=np.int32),
    }
    split_metadata: dict[str, dict[str, object]] = {}

    for split in SPLITS:
        x_raw = np.load(DATA_PROCESSED_DIR / f"X_{split}.npy")
        x_scaled = np.load(DATA_PROCESSED_DIR / f"X_{split}_scaled.npy")
        y_raw = np.load(DATA_PROCESSED_DIR / f"y_{split}.npy")
        y_scaled = np.load(DATA_PROCESSED_DIR / f"y_{split}_scaled.npy")
        rows = sample_index.loc[sample_index["split"] == split].reset_index(drop=True)

        if x_raw.ndim != 2 or x_raw.shape[1] != DEFAULT_WINDOW_SIZE:
            raise ValueError(
                f"X_{split}.npy must have shape (n, {DEFAULT_WINDOW_SIZE}); "
                f"got {x_raw.shape}"
            )
        if x_scaled.shape != x_raw.shape:
            raise ValueError(
                f"X_{split}_scaled.npy shape {x_scaled.shape} does not match "
                f"X_{split}.npy shape {x_raw.shape}"
            )

        lengths = {x_raw.shape[0], y_raw.size, y_scaled.size, len(rows)}
        if len(lengths) != 1:
            raise ValueError(
                f"Split {split} has inconsistent sample counts: "
                f"X={x_raw.shape[0]}, y={y_raw.size}, y_scaled={y_scaled.size}, "
                f"index={len(rows)}"
            )
        if not np.allclose(y_raw.reshape(-1), rows["y"].to_numpy(dtype=float)):
            raise ValueError(f"Split {split} labels do not match sample_index.csv")

        payload[f"X_{split}"] = np.asarray(x_raw, dtype=np.float64)
        payload[f"X_{split}_scaled"] = np.asarray(x_scaled, dtype=np.float64)
        payload[f"y_{split}"] = _as_column(np.asarray(y_raw, dtype=np.float64))
        payload[f"y_{split}_scaled"] = _as_column(
            np.asarray(y_scaled, dtype=np.float64)
        )
        payload[f"sample_id_{split}"] = _as_column(
            rows["sample_id"].to_numpy(dtype=np.int32)
        )
        for date_column in ("x_start_date", "x_end_date", "target_date"):
            payload[f"{date_column}_{split}"] = _as_matlab_strings(rows[date_column])

        split_metadata[split] = {
            "num_samples": int(x_raw.shape[0]),
            "window_size": int(x_raw.shape[1]),
            "first_target_date": str(rows["target_date"].iloc[0]),
            "last_target_date": str(rows["target_date"].iloc[-1]),
        }

    mat_path = output_dir / "cpi_windows.mat"
    savemat(mat_path, payload, do_compression=True, long_field_names=True)

    scaler_path = DATA_PROCESSED_DIR / "scaler_params_scaled.json"
    with scaler_path.open("r", encoding="utf-8") as file:
        scaler_parameters = json.load(file)

    metadata = {
        "description": "CPI windows shared with the baseline models",
        "source_directory": "data_processed",
        "window_size": DEFAULT_WINDOW_SIZE,
        "forecast_horizon_months": 1,
        "split_method": "chronological 70/15/15",
        "input_for_reservoir": "X_<split>_scaled",
        "target_for_evaluation": "y_<split>",
        "scaler_fit_scope": "training split only",
        "scaler_parameters": scaler_parameters,
        "splits": split_metadata,
    }
    metadata_path = output_dir / "cpi_windows_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return mat_path, metadata_path


def main() -> None:
    mat_path, metadata_path = export_cpi_windows()
    print(f"Exported MATLAB data: {mat_path}")
    print(f"Exported metadata: {metadata_path}")


if __name__ == "__main__":
    main()
