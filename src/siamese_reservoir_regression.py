"""Train the readout of a Siamese optical reservoir CPI regressor.

The optical reservoir states are fixed. Only a Ridge readout is trained to map
the shared-branch state difference to the continuous CPI difference.
"""

from __future__ import annotations

import argparse
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


SPLITS = ("train", "val", "test")
DEFAULT_STATE_DIR = ROOT_DIR / "matlab" / "optical_reservoir_cpi" / "states"
DEFAULT_ALPHAS = (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass
class ReadoutBundle:
    scaler: StandardScaler
    model: Ridge
    feature_mode: str
    aggregation: str


def load_state_lookup(state_dir: Path) -> dict[int, np.ndarray]:
    """Load MATLAB state caches and map each global sample_id to one state."""
    lookup: dict[int, np.ndarray] = {}
    state_width: int | None = None

    for split in SPLITS:
        state_path = state_dir / f"states_{split}.mat"
        if not state_path.exists():
            raise FileNotFoundError(
                f"Missing {state_path}. Run run_all_cpi_simulations in MATLAB first."
            )
        data = loadmat(state_path)
        if "state_matrix" not in data or "sample_id" not in data:
            raise ValueError(f"{state_path} must contain state_matrix and sample_id")

        states = np.asarray(data["state_matrix"], dtype=float)
        sample_ids = np.asarray(data["sample_id"]).reshape(-1).astype(int)
        if states.ndim != 2 or states.shape[0] != sample_ids.size:
            raise ValueError(
                f"Invalid shapes in {state_path}: states={states.shape}, "
                f"sample_id={sample_ids.shape}"
            )
        if not np.all(np.isfinite(states)):
            raise ValueError(f"{state_path} contains non-finite states")
        if state_width is None:
            state_width = states.shape[1]
        elif states.shape[1] != state_width:
            raise ValueError("All state files must use the same number of virtual nodes")

        for sample_id, state in zip(sample_ids, states):
            if int(sample_id) in lookup:
                raise ValueError(f"Duplicate sample_id across state files: {sample_id}")
            lookup[int(sample_id)] = state

    return lookup


def load_pair_tables(data_dir: Path = DATA_PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    required = {
        "sample_i_id",
        "sample_j_id",
        "target_i_date",
        "cpi_i",
        "cpi_j",
        "delta_cpi",
    }
    for split in SPLITS:
        table = pd.read_csv(data_dir / f"pair_indices_{split}.csv")
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{split} pair table is missing columns: {sorted(missing)}")
        if table.empty:
            raise ValueError(f"{split} pair table is empty")
        tables[split] = table
    return tables


def build_pair_features(
    pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
    feature_mode: str = "signed_diff",
) -> np.ndarray:
    """Build the shared-branch comparison features for each pair."""
    missing_ids = (
        set(pairs["sample_i_id"].astype(int))
        | set(pairs["sample_j_id"].astype(int))
    ).difference(state_lookup)
    if missing_ids:
        preview = sorted(missing_ids)[:10]
        raise ValueError(f"Reservoir states are missing sample IDs: {preview}")

    h_i = np.vstack([state_lookup[int(value)] for value in pairs["sample_i_id"]])
    h_j = np.vstack([state_lookup[int(value)] for value in pairs["sample_j_id"]])
    signed_diff = h_i - h_j

    if feature_mode == "signed_diff":
        return signed_diff
    if feature_mode == "signed_abs":
        return np.hstack([signed_diff, np.abs(signed_diff)])
    if feature_mode == "concat":
        return np.hstack([h_i, h_j])
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def aggregate_target_predictions(
    pair_predictions: pd.DataFrame,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Combine several reference-based CPI estimates for each target month."""
    rows: list[dict[str, object]] = []
    for sample_i_id, group in pair_predictions.groupby("sample_i_id", sort=False):
        actual_values = group["cpi_i"].to_numpy(dtype=float)
        if not np.allclose(actual_values, actual_values[0]):
            raise ValueError(f"Inconsistent cpi_i values for sample {sample_i_id}")

        estimates = group["cpi_pred_pair"].to_numpy(dtype=float)
        if aggregation == "mean":
            predicted = float(np.mean(estimates))
        elif aggregation == "inverse_distance":
            if "window_distance" not in group.columns:
                raise ValueError("inverse_distance aggregation needs window_distance")
            distances = group["window_distance"].to_numpy(dtype=float)
            weights = 1.0 / np.maximum(distances, 1e-6)
            predicted = float(np.average(estimates, weights=weights))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        rows.append({
            "sample_i_id": int(sample_i_id),
            "target_date": str(group["target_i_date"].iloc[0]),
            "cpi_actual": float(actual_values[0]),
            "cpi_predicted": predicted,
            "num_references": int(len(group)),
            "reference_prediction_std": float(np.std(estimates)),
        })

    result = pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)
    result["error"] = result["cpi_predicted"] - result["cpi_actual"]
    result["absolute_error"] = result["error"].abs()
    return result


def predict_pairs(
    bundle: ReadoutBundle,
    pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = build_pair_features(pairs, state_lookup, bundle.feature_mode)
    features_scaled = bundle.scaler.transform(features)
    delta_predicted = bundle.model.predict(features_scaled)

    pair_predictions = pairs.copy()
    pair_predictions["delta_cpi_predicted"] = delta_predicted
    pair_predictions["cpi_pred_pair"] = (
        pair_predictions["cpi_j"].to_numpy(dtype=float) + delta_predicted
    )
    pair_predictions["delta_error"] = (
        pair_predictions["delta_cpi_predicted"] - pair_predictions["delta_cpi"]
    )
    target_predictions = aggregate_target_predictions(
        pair_predictions, bundle.aggregation
    )
    return pair_predictions, target_predictions


def select_alpha(
    train_features: np.ndarray,
    train_delta: np.ndarray,
    val_pairs: pd.DataFrame,
    val_features: np.ndarray,
    alphas: Iterable[float],
    aggregation: str,
) -> tuple[float, list[dict[str, float]]]:
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    val_scaled = scaler.transform(val_features)
    trials: list[dict[str, float]] = []

    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(train_scaled, train_delta)
        pair_predictions = val_pairs.copy()
        pair_predictions["delta_cpi_predicted"] = model.predict(val_scaled)
        pair_predictions["cpi_pred_pair"] = (
            pair_predictions["cpi_j"].to_numpy(dtype=float)
            + pair_predictions["delta_cpi_predicted"]
        )
        targets = aggregate_target_predictions(pair_predictions, aggregation)
        metrics = regression_metrics(targets["cpi_actual"], targets["cpi_predicted"])
        trials.append({"alpha": float(alpha), **metrics})

    best = min(trials, key=lambda item: (item["rmse"], item["mae"], item["alpha"]))
    return float(best["alpha"]), trials


def train_readout(
    pair_tables: dict[str, pd.DataFrame],
    state_lookup: dict[int, np.ndarray],
    feature_mode: str = "signed_diff",
    aggregation: str = "mean",
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> tuple[ReadoutBundle, list[dict[str, float]]]:
    train_features = build_pair_features(
        pair_tables["train"], state_lookup, feature_mode
    )
    val_features = build_pair_features(pair_tables["val"], state_lookup, feature_mode)
    train_delta = pair_tables["train"]["delta_cpi"].to_numpy(dtype=float)

    best_alpha, trials = select_alpha(
        train_features,
        train_delta,
        pair_tables["val"],
        val_features,
        alphas,
        aggregation,
    )
    scaler = StandardScaler().fit(train_features)
    model = Ridge(alpha=best_alpha).fit(scaler.transform(train_features), train_delta)
    return ReadoutBundle(scaler, model, feature_mode, aggregation), trials


def save_model(bundle: ReadoutBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        coefficient=bundle.model.coef_,
        intercept=np.asarray([bundle.model.intercept_]),
        alpha=np.asarray([bundle.model.alpha]),
        scaler_mean=bundle.scaler.mean_,
        scaler_scale=bundle.scaler.scale_,
        feature_mode=np.asarray([bundle.feature_mode]),
        aggregation=np.asarray([bundle.aggregation]),
    )


def save_test_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(predictions["target_date"], predictions["cpi_actual"], label="Actual CPI")
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted"],
        label="Predicted CPI",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI")
    axis.set_title("Siamese optical reservoir regression: test predictions")
    axis.tick_params(axis="x", rotation=60)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_training(
    state_dir: Path = DEFAULT_STATE_DIR,
    output_dir: Path = RESULTS_DIR,
    feature_mode: str = "signed_diff",
    aggregation: str = "mean",
) -> pd.DataFrame:
    state_lookup = load_state_lookup(state_dir)
    pair_tables = load_pair_tables()
    bundle, alpha_trials = train_readout(
        pair_tables, state_lookup, feature_mode, aggregation
    )

    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    model_dir = output_dir / "models"
    table_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    target_outputs: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        pair_predictions, target_predictions = predict_pairs(
            bundle, pair_tables[split], state_lookup
        )
        pair_metrics = regression_metrics(
            pair_predictions["delta_cpi"],
            pair_predictions["delta_cpi_predicted"],
        )
        target_metrics = regression_metrics(
            target_predictions["cpi_actual"],
            target_predictions["cpi_predicted"],
        )
        metric_rows.append({
            "split": split,
            "num_pairs": len(pair_predictions),
            "num_targets": len(target_predictions),
            "delta_mae": pair_metrics["mae"],
            "delta_rmse": pair_metrics["rmse"],
            "cpi_mae": target_metrics["mae"],
            "cpi_rmse": target_metrics["rmse"],
        })
        pair_predictions.to_csv(
            table_dir / f"siamese_optical_pair_predictions_{split}.csv", index=False
        )
        target_predictions.to_csv(
            table_dir / f"siamese_optical_predictions_{split}.csv", index=False
        )
        target_outputs[split] = target_predictions

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(table_dir / "siamese_optical_reservoir_metrics.csv", index=False)
    pd.DataFrame(alpha_trials).to_csv(
        table_dir / "siamese_optical_alpha_selection.csv", index=False
    )
    save_model(bundle, model_dir / "siamese_optical_reservoir_readout.npz")
    save_test_plot(
        target_outputs["test"], figure_dir / "siamese_optical_test_predictions.png"
    )

    summary = {
        "task": "siamese optical reservoir CPI difference regression",
        "state_directory": str(state_dir),
        "feature_mode": feature_mode,
        "aggregation": aggregation,
        "selected_alpha": float(bundle.model.alpha),
        "reservoir_training": "fixed; only the Ridge readout is trained",
        "formula": "cpi_i_hat = cpi_j + delta_cpi_hat",
    }
    with (table_dir / "siamese_optical_run_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--feature-mode",
        choices=("signed_diff", "signed_abs", "concat"),
        default="signed_diff",
    )
    parser.add_argument(
        "--aggregation", choices=("mean", "inverse_distance"), default="mean"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_training(
        state_dir=args.state_dir,
        output_dir=args.output_dir,
        feature_mode=args.feature_mode,
        aggregation=args.aggregation,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
