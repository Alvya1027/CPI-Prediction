"""Train a standard optical-reservoir CPI readout and compare it with Siamese."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, ROOT_DIR
from src.siamese_reservoir_regression import DEFAULT_ALPHAS, SPLITS, regression_metrics


DEFAULT_STATE_DIR = ROOT_DIR / "matlab" / "optical_reservoir_cpi" / "states"


@dataclass
class OpticalReadout:
    scaler: StandardScaler
    model: Ridge


def load_state_splits(state_dir: Path) -> dict[str, pd.DataFrame]:
    """Load fixed reservoir states and align them with the shared sample index."""
    sample_index = pd.read_csv(DATA_PROCESSED_DIR / "sample_index.csv")
    outputs: dict[str, pd.DataFrame] = {}

    for split in SPLITS:
        state_path = state_dir / f"states_{split}.mat"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing {state_path}")
        data = loadmat(state_path)
        states = np.asarray(data["state_matrix"], dtype=float)
        sample_ids = np.asarray(data["sample_id"]).reshape(-1).astype(int)
        targets = np.asarray(data["target"]).reshape(-1).astype(float)
        if states.ndim != 2 or len(states) != len(sample_ids):
            raise ValueError(f"Invalid state shapes in {state_path}")
        if not np.all(np.isfinite(states)):
            raise ValueError(f"Non-finite states in {state_path}")

        metadata = (
            sample_index.loc[sample_index["split"] == split]
            .set_index("sample_id")
            .loc[sample_ids]
            .reset_index()
        )
        if not np.allclose(metadata["y"].to_numpy(dtype=float), targets):
            raise ValueError(f"Target mismatch in {state_path}")
        metadata["state"] = list(states)
        outputs[split] = metadata

    return outputs


def select_alpha(
    train_states: np.ndarray,
    train_targets: np.ndarray,
    val_states: np.ndarray,
    val_targets: np.ndarray,
    alphas: Iterable[float],
) -> tuple[OpticalReadout, list[dict[str, float]]]:
    scaler = StandardScaler().fit(train_states)
    train_scaled = scaler.transform(train_states)
    val_scaled = scaler.transform(val_states)
    trials: list[dict[str, float]] = []

    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(train_scaled, train_targets)
        metrics = regression_metrics(val_targets, model.predict(val_scaled))
        trials.append({"alpha": float(alpha), **metrics})

    best = min(trials, key=lambda item: (item["rmse"], item["mae"], item["alpha"]))
    model = Ridge(alpha=best["alpha"]).fit(train_scaled, train_targets)
    return OpticalReadout(scaler, model), trials


def _state_matrix(table: pd.DataFrame) -> np.ndarray:
    return np.vstack(table["state"].to_numpy())


def _save_model(bundle: OpticalReadout, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        coefficient=bundle.model.coef_,
        intercept=np.asarray([bundle.model.intercept_]),
        alpha=np.asarray([bundle.model.alpha]),
        scaler_mean=bundle.scaler.mean_,
        scaler_scale=bundle.scaler.scale_,
    )


def _save_comparison(output_dir: Path, optical_metrics: pd.DataFrame) -> None:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    siamese_metrics_path = table_dir / "siamese_optical_reservoir_metrics.csv"
    siamese_predictions_path = table_dir / "siamese_optical_predictions_test.csv"
    if not siamese_metrics_path.exists() or not siamese_predictions_path.exists():
        return

    siamese_metrics = pd.read_csv(siamese_metrics_path)
    comparison = pd.concat(
        [
            optical_metrics.assign(model="ordinary_optical_reservoir"),
            siamese_metrics[["split", "num_targets", "cpi_mae", "cpi_rmse"]]
            .rename(columns={"cpi_mae": "mae", "cpi_rmse": "rmse"})
            .assign(model="siamese_optical_reservoir"),
        ],
        ignore_index=True,
    )[["model", "split", "num_targets", "mae", "rmse"]]
    comparison.to_csv(table_dir / "optical_model_comparison.csv", index=False)

    ordinary = pd.read_csv(table_dir / "optical_reservoir_predictions_test.csv")
    siamese = pd.read_csv(siamese_predictions_path)
    merged = ordinary.merge(
        siamese[["target_date", "cpi_predicted"]],
        on="target_date",
        suffixes=("_ordinary", "_siamese"),
        validate="one_to_one",
    )
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(merged["target_date"], merged["cpi_actual"], label="Actual CPI")
    axis.plot(
        merged["target_date"],
        merged["cpi_predicted_ordinary"],
        label="Ordinary optical reservoir",
    )
    axis.plot(
        merged["target_date"],
        merged["cpi_predicted_siamese"],
        label="Siamese optical reservoir",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI")
    axis.set_title("Optical reservoir CPI regression: test comparison")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "optical_model_test_comparison.png", dpi=180)
    plt.close(fig)


def run_training(
    state_dir: Path = DEFAULT_STATE_DIR,
    output_dir: Path = RESULTS_DIR,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> pd.DataFrame:
    splits = load_state_splits(state_dir)
    train = splits["train"]
    val = splits["val"]
    bundle, trials = select_alpha(
        _state_matrix(train),
        train["y"].to_numpy(dtype=float),
        _state_matrix(val),
        val["y"].to_numpy(dtype=float),
        alphas,
    )

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    for split in SPLITS:
        table = splits[split]
        actual = table["y"].to_numpy(dtype=float)
        predicted = bundle.model.predict(bundle.scaler.transform(_state_matrix(table)))
        metrics = regression_metrics(actual, predicted)
        metric_rows.append(
            {"split": split, "num_targets": len(table), **metrics}
        )
        predictions = table[["sample_id", "target_date", "y"]].rename(
            columns={"y": "cpi_actual"}
        )
        predictions["cpi_predicted"] = predicted
        predictions["error"] = predictions["cpi_predicted"] - predictions["cpi_actual"]
        predictions["absolute_error"] = predictions["error"].abs()
        predictions.to_csv(
            table_dir / f"optical_reservoir_predictions_{split}.csv", index=False
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(table_dir / "optical_reservoir_metrics.csv", index=False)
    pd.DataFrame(trials).to_csv(
        table_dir / "optical_reservoir_alpha_selection.csv", index=False
    )
    _save_model(bundle, output_dir / "models" / "optical_reservoir_readout.npz")
    with (table_dir / "optical_reservoir_run_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "task": "ordinary optical reservoir direct CPI regression",
                "selected_alpha": float(bundle.model.alpha),
                "reservoir_training": "fixed; only the Ridge readout is trained",
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")
    _save_comparison(output_dir, metrics)
    return metrics


def main() -> None:
    metrics = run_training()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
