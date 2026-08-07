"""Fit the single optical-reservoir readout on YoY train45/test47 states."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR


PROFILE_ROOT = ROOT_DIR / "matlab" / "optical_reservoir_cpi_yoy_train45_noval_20260807"
STATE_DIR = PROFILE_ROOT / "states"
SAMPLE_INDEX = PROFILE_ROOT / "data" / "sample_index.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "optical_reservoir_yoy_train45_noval_20260807"
ALPHAS = (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def _load_split(split: str, index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = loadmat(STATE_DIR / f"states_{split}.mat")
    states = np.asarray(data["state_matrix"], dtype=float)
    sample_ids = np.asarray(data["sample_id"]).reshape(-1).astype(int)
    targets = np.asarray(data["target"]).reshape(-1).astype(float)
    if states.shape != (len(sample_ids), 50) or not np.isfinite(states).all():
        raise ValueError(f"Invalid {split} state cache shape or values: {states.shape}")
    expected = index.loc[index["split"].eq(split)].set_index("sample_id").loc[sample_ids]
    if not np.allclose(expected["y"].to_numpy(float), targets):
        raise ValueError(f"Target mismatch in {split} reservoir states")
    return states, targets, sample_ids


def _select_alpha(X: np.ndarray, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    splitter = TimeSeriesSplit(n_splits=5)
    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        fold_rows = []
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
            scaler = StandardScaler().fit(X[train_idx])
            model = Ridge(alpha=float(alpha)).fit(
                scaler.transform(X[train_idx]), y[train_idx]
            )
            prediction = model.predict(scaler.transform(X[val_idx]))
            fold_rows.append({"fold": fold, **_metrics(y[val_idx], prediction)})
        rows.append(
            {
                "alpha": float(alpha),
                "cv_mae": float(np.mean([r["mae"] for r in fold_rows])),
                "cv_rmse": float(np.mean([r["rmse"] for r in fold_rows])),
            }
        )
    trials = pd.DataFrame(rows).sort_values(["cv_rmse", "cv_mae", "alpha"])
    return float(trials.iloc[0]["alpha"]), trials


def main() -> None:
    index = pd.read_csv(SAMPLE_INDEX)
    X_train, y_train, train_ids = _load_split("train", index)
    X_test, y_test, test_ids = _load_split("test", index)
    alpha, trials = _select_alpha(X_train, y_train)
    scaler = StandardScaler().fit(X_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(X_train), y_train)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_dir = OUTPUT_DIR / "tables"
    figure_dir = OUTPUT_DIR / "figures"
    model_dir = OUTPUT_DIR / "models"
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    rows = []
    prediction_tables = []
    for split, X, y, ids in (
        ("train", X_train, y_train, train_ids),
        ("test", X_test, y_test, test_ids),
    ):
        prediction = model.predict(scaler.transform(X))
        rows.append({"model": "single_optical_reservoir", "split": split, "num_targets": len(y), **_metrics(y, prediction)})
        meta = index.set_index("sample_id").loc[ids]
        prediction_tables.append(
            pd.DataFrame(
                {
                    "model": "single_optical_reservoir",
                    "split": split,
                    "sample_id": ids,
                    "target_date": meta["target_date"].to_numpy(),
                    "cpi_actual": y,
                    "cpi_predicted": prediction,
                    "error": prediction - y,
                    "absolute_error": np.abs(prediction - y),
                }
            )
        )
    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(table_dir / "optical_reservoir_metrics.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        table_dir / "optical_reservoir_predictions.csv", index=False
    )
    trials.to_csv(table_dir / "optical_reservoir_alpha_train_cv.csv", index=False)
    np.savez_compressed(
        model_dir / "optical_reservoir_readout.npz",
        alpha=np.asarray([alpha]),
        coefficient=model.coef_,
        intercept=np.asarray([model.intercept_]),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )

    test_table = prediction_tables[1]
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(test_table["target_date"], test_table["cpi_actual"], marker="o", label="Actual YoY CPI")
    axis.plot(test_table["target_date"], test_table["cpi_predicted"], marker="o", label="Single optical reservoir")
    axis.set_title("Single optical reservoir: YoY train45/test47")
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI YoY index (previous year same month=100)")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "optical_reservoir_yoy_test_predictions.png", dpi=180)
    plt.close(fig)

    summary = {
        "experiment": "single_optical_reservoir_yoy_train45_noval",
        "created_at": datetime.now().astimezone().isoformat(),
        "source": "data_processed/cpi_data_lastyear=100.csv actual sequence",
        "window_protocol": "continuous previous 12 actual YoY values; one-step-ahead target",
        "split_protocol": "train 2018-09..2022-05 (45), test 2022-06..2026-04 (47), no validation",
        "alpha_selection": "5-fold chronological TimeSeriesSplit on train45 only",
        "selected_alpha": alpha,
        "state_dir": str(STATE_DIR),
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics_table.to_string(index=False))
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
