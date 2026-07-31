"""Explore pair-structure improvements for the MoM strict-closed50 model.

Only Siamese-side choices are changed:

* target-month gap instead of non-overlapping-window gap;
* all legal causal pairs with target/bin-balanced weights;
* mirrored pairs plus a zero-intercept readout for exact antisymmetry.

The optical reservoir, its 50-dimensional states, and the accessible set of
50 training windows remain frozen. Validation selects one configuration before
the test arrays and test states are opened.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR
from src.create_siamese_pairs import (
    _assign_delta_bins,
    _compute_delta_bin_thresholds,
    _sample_pairs_by_bin,
)
from src.siamese_closed50_experiment import _load_split, build_closed_train_pool
from src.siamese_closed50_gap_hybrid_experiment import (
    HybridDistanceCalibration,
    _window_lookup,
    add_hybrid_distances,
    fit_hybrid_distance_calibration,
)
from src.siamese_closed50_ssa_optimization import (
    _validation_block_rmse,
    aggregate_power_predictions,
    select_search_references,
)
from src.siamese_mom_closed50_pipeline import (
    DATA_DIR,
    ORDINARY_DIR,
    STATE_DIR,
    audit_profile,
)
from src.siamese_reservoir_regression import (
    ReadoutBundle,
    build_pair_features,
    load_state_lookup,
    regression_metrics,
    save_model,
)
from src.siamese_validation_search import load_validation_state_lookup


OUTPUT_DIR = RESULTS_DIR / "siamese_optical_mom_pair_optimization_20260731"
CURRENT_SSA_DIR = RESULTS_DIR / "siamese_optical_mom_closed50_ssa_20260730"
TARGET_GAPS = (1, 3, 6, 12)
TRAINING_MODES = ("sampled", "balanced", "antisymmetric")
K_REFERENCES = 5
LEVEL_WEIGHT = 0.041
DISTANCE_POWER = 0.654
MAX_PAIRS_PER_BIN = 1
ALPHAS = (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1_000.0)
STABILITY_PENALTY = 0.10
AGGREGATION_MODES = ("power", "mean", "trimmed_mean", "median")


@dataclass(frozen=True)
class PairOptimizationConfig:
    target_gap_months: int
    training_mode: str
    k_references: int = K_REFERENCES
    level_weight: float = LEVEL_WEIGHT
    distance_power: float = DISTANCE_POWER


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    stds[stds == 0.0] = 1.0
    return (values - means) / stds


def build_target_gap_candidates(
    target_split: dict[str, object],
    reference_split: dict[str, object],
    target_gap_months: int,
) -> pd.DataFrame:
    """Build pairs using target-month chronology, allowing window overlap."""
    if target_gap_months < 1:
        raise ValueError("target_gap_months must be at least 1")
    target_index = target_split["index"]
    reference_index = reference_split["index"]
    target_dates = pd.to_datetime(target_index["target_date"])
    reference_dates = pd.to_datetime(reference_index["target_date"])
    target_month = (target_dates.dt.year * 12 + target_dates.dt.month).to_numpy()
    reference_month = (
        reference_dates.dt.year * 12 + reference_dates.dt.month
    ).to_numpy()
    gaps = target_month[:, None] - reference_month[None, :]
    i_rows, j_rows = np.where(gaps >= int(target_gap_months))
    if len(i_rows) == 0:
        return pd.DataFrame()

    X_i = np.asarray(target_split["X"], dtype=float)
    X_j = np.asarray(reference_split["X"], dtype=float)
    y_i = np.asarray(target_split["y"], dtype=float).reshape(-1)
    y_j = np.asarray(reference_split["y"], dtype=float).reshape(-1)
    distances = np.linalg.norm(
        _zscore_rows(X_i)[i_rows] - _zscore_rows(X_j)[j_rows],
        axis=1,
    ) / np.sqrt(X_i.shape[1])
    return pd.DataFrame(
        {
            "sample_i_id": target_index["sample_id"].to_numpy(dtype=int)[i_rows],
            "sample_j_id": reference_index["sample_id"].to_numpy(dtype=int)[j_rows],
            "x_i_start_date": target_index["x_start_date"].to_numpy()[i_rows],
            "x_i_end_date": target_index["x_end_date"].to_numpy()[i_rows],
            "target_i_date": target_index["target_date"].to_numpy()[i_rows],
            "x_j_start_date": reference_index["x_start_date"].to_numpy()[j_rows],
            "x_j_end_date": reference_index["x_end_date"].to_numpy()[j_rows],
            "target_j_date": reference_index["target_date"].to_numpy()[j_rows],
            "cpi_i": y_i[i_rows],
            "cpi_j": y_j[j_rows],
            "delta_cpi": y_i[i_rows] - y_j[j_rows],
            "window_distance": distances,
            "target_gap_months_actual": gaps[i_rows, j_rows],
        }
    )


def add_balanced_pair_weights(pairs: pd.DataFrame) -> pd.DataFrame:
    """Give every target total weight one and balance its available delta bins."""
    if "delta_bin" not in pairs:
        raise ValueError("pairs must contain delta_bin")
    result = pairs.copy()
    result["sample_weight"] = 0.0
    for _, target_rows in result.groupby("sample_i_id"):
        bins = list(target_rows.groupby("delta_bin").groups.values())
        bin_total = 1.0 / len(bins)
        for indices in bins:
            result.loc[indices, "sample_weight"] = bin_total / len(indices)
    totals = result.groupby("sample_i_id")["sample_weight"].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0):
        raise ValueError("target-balanced weights do not sum to one")
    return result


def mirror_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add reversed train-only pairs with opposite delta and equal total weight."""
    original = pairs.copy()
    original["sample_weight"] = original["sample_weight"] * 0.5
    mirrored = original.copy()
    swaps = (
        ("sample_i_id", "sample_j_id"),
        ("x_i_start_date", "x_j_start_date"),
        ("x_i_end_date", "x_j_end_date"),
        ("target_i_date", "target_j_date"),
        ("cpi_i", "cpi_j"),
    )
    for left, right in swaps:
        mirrored[left] = original[right].to_numpy()
        mirrored[right] = original[left].to_numpy()
    mirrored["delta_cpi"] = -original["delta_cpi"].to_numpy(dtype=float)
    mirrored["selection_method"] = "mirrored_antisymmetry_regularization"
    output = pd.concat([original, mirrored], ignore_index=True)
    if not np.isclose(output["delta_cpi"].sum(), 0.0, atol=1e-10):
        raise ValueError("mirrored pair deltas are not antisymmetric")
    return output


def _with_search_distance(
    pairs: pd.DataFrame,
    level_weight: float,
) -> pd.DataFrame:
    result = pairs.copy()
    result["search_distance"] = (
        (1.0 - level_weight)
        * result["shape_distance_normalized"].to_numpy(dtype=float)
        + level_weight
        * result["level_distance_normalized"].to_numpy(dtype=float)
    )
    return result


def prepare_training_pairs(
    candidates: pd.DataFrame,
    config: PairOptimizationConfig,
) -> pd.DataFrame:
    thresholds = _compute_delta_bin_thresholds(candidates)
    binned = _assign_delta_bins(candidates, thresholds)
    if config.training_mode == "sampled":
        selected = _sample_pairs_by_bin(
            binned,
            max_per_bin=MAX_PAIRS_PER_BIN,
        ).reset_index(drop=True)
        selected["sample_weight"] = 1.0
        selected["selection_method"] = "target_gap_delta_stratified_sample"
    elif config.training_mode in {"balanced", "antisymmetric"}:
        selected = add_balanced_pair_weights(binned).reset_index(drop=True)
        selected["selection_method"] = "all_target_gap_pairs_balanced"
        if config.training_mode == "antisymmetric":
            selected = mirror_pairs(selected)
    else:
        raise ValueError(f"unknown training mode: {config.training_mode}")
    return _with_search_distance(selected, config.level_weight)


def prepare_evaluation_pairs(
    target_split: dict[str, object],
    train_pool: dict[str, object],
    calibration: HybridDistanceCalibration,
    config: PairOptimizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = build_target_gap_candidates(
        target_split,
        train_pool,
        config.target_gap_months,
    )
    candidates = add_hybrid_distances(
        raw,
        _window_lookup(target_split, train_pool),
        calibration,
    )
    selected = select_search_references(
        candidates,
        config.k_references,
        config.level_weight,
    )
    return selected, candidates


def _fit_and_validate(
    train_pairs: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
    config: PairOptimizationConfig,
    alphas: Iterable[float],
) -> dict[str, object]:
    antisymmetric = config.training_mode == "antisymmetric"
    train_features = build_pair_features(train_pairs, state_lookup, "signed_diff")
    validation_features = build_pair_features(
        validation_pairs,
        state_lookup,
        "signed_diff",
    )
    scaler = StandardScaler(with_mean=not antisymmetric).fit(train_features)
    train_scaled = scaler.transform(train_features)
    validation_scaled = scaler.transform(validation_features)
    target_delta = train_pairs["delta_cpi"].to_numpy(dtype=float)
    sample_weight = train_pairs["sample_weight"].to_numpy(dtype=float)
    trials: list[dict[str, float]] = []
    fitted: dict[float, Ridge] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        for alpha in alphas:
            model = Ridge(
                alpha=float(alpha),
                fit_intercept=not antisymmetric,
            ).fit(
                train_scaled,
                target_delta,
                sample_weight=sample_weight,
            )
            pair_output = validation_pairs.copy()
            delta_prediction = model.predict(validation_scaled)
            pair_output["delta_cpi_predicted"] = delta_prediction
            pair_output["cpi_pred_pair"] = (
                pair_output["cpi_j"].to_numpy(dtype=float) + delta_prediction
            )
            predictions = aggregate_power_predictions(
                pair_output,
                config.distance_power,
            )
            metrics = regression_metrics(
                predictions["cpi_actual"],
                predictions["cpi_predicted"],
            )
            trials.append({"alpha": float(alpha), **metrics})
            fitted[float(alpha)] = model
    best = min(trials, key=lambda row: (row["rmse"], row["mae"], row["alpha"]))
    best_alpha = float(best["alpha"])
    model = fitted[best_alpha]
    pair_output = validation_pairs.copy()
    delta_prediction = model.predict(validation_scaled)
    pair_output["delta_cpi_predicted"] = delta_prediction
    pair_output["cpi_pred_pair"] = (
        pair_output["cpi_j"].to_numpy(dtype=float) + delta_prediction
    )
    pair_output["delta_error"] = (
        pair_output["delta_cpi_predicted"] - pair_output["delta_cpi"]
    )
    predictions = aggregate_power_predictions(
        pair_output,
        config.distance_power,
    )
    block_rmse = _validation_block_rmse(predictions)
    block_std = float(np.std(block_rmse))
    return {
        "config": config,
        "scaler": scaler,
        "model": model,
        "selected_alpha": best_alpha,
        "alpha_trials": trials,
        "validation_pair_predictions": pair_output,
        "validation_predictions": predictions,
        "val_mae": float(best["mae"]),
        "val_rmse": float(best["rmse"]),
        "validation_block_rmse": block_rmse,
        "validation_block_rmse_std": block_std,
        "fitness": float(best["rmse"] + STABILITY_PENALTY * block_std),
    }


def _predict(
    result: dict[str, object],
    pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
    aggregation_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = build_pair_features(pairs, state_lookup, "signed_diff")
    delta_prediction = result["model"].predict(result["scaler"].transform(features))
    pair_output = pairs.copy()
    pair_output["delta_cpi_predicted"] = delta_prediction
    pair_output["cpi_pred_pair"] = (
        pair_output["cpi_j"].to_numpy(dtype=float) + delta_prediction
    )
    pair_output["delta_error"] = (
        pair_output["delta_cpi_predicted"] - pair_output["delta_cpi"]
    )
    predictions = aggregate_pair_predictions(
        pair_output,
        aggregation_mode,
        result["config"].distance_power,
    )
    return pair_output, predictions


def aggregate_pair_predictions(
    pair_predictions: pd.DataFrame,
    mode: str,
    distance_power: float,
) -> pd.DataFrame:
    """Aggregate K reference estimates without using the unknown target label."""
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"unknown aggregation mode: {mode}")
    rows: list[dict[str, object]] = []
    for sample_i_id, group in pair_predictions.groupby("sample_i_id", sort=False):
        estimates = group["cpi_pred_pair"].to_numpy(dtype=float)
        if mode == "power":
            distances = group["search_distance"].to_numpy(dtype=float)
            weights = 1.0 / np.maximum(distances, 1e-6) ** float(distance_power)
            prediction = float(np.average(estimates, weights=weights))
        elif mode == "mean":
            prediction = float(np.mean(estimates))
        elif mode == "trimmed_mean":
            if len(estimates) < 3:
                prediction = float(np.mean(estimates))
            else:
                prediction = float(np.mean(np.sort(estimates)[1:-1]))
        else:
            prediction = float(np.median(estimates))
        rows.append(
            {
                "sample_i_id": int(sample_i_id),
                "target_date": str(group["target_i_date"].iloc[0]),
                "cpi_actual": float(group["cpi_i"].iloc[0]),
                "cpi_predicted": prediction,
                "num_references": int(len(group)),
                "reference_prediction_std": float(np.std(estimates)),
                "mean_search_distance": float(
                    group["search_distance"].mean()
                ),
                "aggregation_mode": mode,
            }
        )
    result = pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)
    result["error"] = result["cpi_predicted"] - result["cpi_actual"]
    result["absolute_error"] = result["error"].abs()
    return result


def _save_figures(
    validation_table: pd.DataFrame,
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9.5, 5.0))
    for mode, rows in validation_table.groupby("training_mode"):
        rows = rows.sort_values("target_gap_months")
        axis.plot(
            rows["target_gap_months"],
            rows["val_rmse"],
            marker="o",
            label=mode,
        )
    axis.set_xticks(TARGET_GAPS)
    axis.set_xlabel("Target-month gap")
    axis.set_ylabel("Validation RMSE")
    axis.set_title("MoM Siamese pair-structure validation comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_pair_structure_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    axis.plot(
        predictions["target_date"],
        predictions["cpi_actual"],
        color="black",
        linewidth=2,
        label="Actual MoM CPI",
    )
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_ordinary"],
        label="Ordinary optical reservoir",
    )
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_current_ssa"],
        label="Current SSA Siamese",
    )
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_pair_optimized"],
        label="Pair-optimized Siamese",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI MoM index (previous month=100)")
    axis.set_title("Exploratory MoM strict-closed50 test predictions")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_prediction_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    ordered = comparison.sort_values("test_rmse")
    x = np.arange(len(ordered))
    width = 0.36
    axis.bar(x - width / 2, ordered["test_mae"], width, label="MAE")
    axis.bar(x + width / 2, ordered["test_rmse"], width, label="RMSE")
    axis.set_xticks(x, ordered["model"], rotation=15, ha="right")
    axis.set_ylabel("Error")
    axis.set_title("Exploratory MoM optical-reservoir test metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_metric_comparison.png", dpi=180)
    plt.close(fig)


def run_pair_optimization(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_DIR,
    state_dir: Path = STATE_DIR,
    ordinary_dir: Path = ORDINARY_DIR,
    current_ssa_dir: Path = CURRENT_SSA_DIR,
    target_gaps: Iterable[int] = TARGET_GAPS,
    training_modes: Iterable[str] = TRAINING_MODES,
    alphas: Iterable[float] = ALPHAS,
) -> pd.DataFrame:
    audit = audit_profile(data_dir, state_dir)
    train_pool = build_closed_train_pool(data_dir)
    validation_split = _load_split(data_dir, "val")
    validation_states = load_validation_state_lookup(state_dir)
    train_ids = set(train_pool["index"]["sample_id"].astype(int))
    rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    cache: dict[tuple[int, str], dict[str, object]] = {}

    for target_gap in target_gaps:
        raw_train = build_target_gap_candidates(
            train_pool,
            train_pool,
            int(target_gap),
        )
        calibration = fit_hybrid_distance_calibration(train_pool, raw_train)
        train_candidates = add_hybrid_distances(
            raw_train,
            _window_lookup(train_pool),
            calibration,
        )
        for training_mode in training_modes:
            config = PairOptimizationConfig(
                target_gap_months=int(target_gap),
                training_mode=str(training_mode),
            )
            train_pairs = prepare_training_pairs(train_candidates, config)
            validation_pairs, validation_candidates = prepare_evaluation_pairs(
                validation_split,
                train_pool,
                calibration,
                config,
            )
            result = _fit_and_validate(
                train_pairs,
                validation_pairs,
                validation_states,
                config,
                alphas,
            )
            result.update(
                {
                    "train_pairs": train_pairs,
                    "train_candidates": train_candidates,
                    "validation_pairs": validation_pairs,
                    "validation_candidates": validation_candidates,
                    "calibration": calibration,
                }
            )
            rows.append(
                {
                    **asdict(config),
                    "candidate_train_pairs": len(train_candidates),
                    "selected_train_pairs": len(train_pairs),
                    "causal_train_pair_targets": train_candidates[
                        "sample_i_id"
                    ].nunique(),
                    "validation_pairs": len(validation_pairs),
                    "selected_alpha": result["selected_alpha"],
                    "val_mae": result["val_mae"],
                    "val_rmse": result["val_rmse"],
                    "validation_block_rmse_std": result[
                        "validation_block_rmse_std"
                    ],
                    "fitness": result["fitness"],
                }
            )
            alpha_rows.extend(
                {
                    "target_gap_months": target_gap,
                    "training_mode": training_mode,
                    **trial,
                }
                for trial in result["alpha_trials"]
            )
            cache[(int(target_gap), str(training_mode))] = result

    validation_table = pd.DataFrame(rows).sort_values(
        ["fitness", "val_rmse", "val_mae"]
    )
    winner = validation_table.iloc[0]
    selected_result = cache[
        (int(winner["target_gap_months"]), str(winner["training_mode"]))
    ]
    selected_config: PairOptimizationConfig = selected_result["config"]
    aggregation_rows: list[dict[str, object]] = []
    aggregation_predictions: dict[str, pd.DataFrame] = {}
    validation_pair_output = selected_result["validation_pair_predictions"]
    for aggregation_mode in AGGREGATION_MODES:
        predictions = aggregate_pair_predictions(
            validation_pair_output,
            aggregation_mode,
            selected_config.distance_power,
        )
        metrics = regression_metrics(
            predictions["cpi_actual"],
            predictions["cpi_predicted"],
        )
        block_rmse = _validation_block_rmse(predictions)
        block_std = float(np.std(block_rmse))
        aggregation_rows.append(
            {
                "aggregation_mode": aggregation_mode,
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "validation_block_rmse_std": block_std,
                "fitness": metrics["rmse"] + STABILITY_PENALTY * block_std,
            }
        )
        aggregation_predictions[aggregation_mode] = predictions
    aggregation_table = pd.DataFrame(aggregation_rows).sort_values(
        ["fitness", "val_rmse", "val_mae"]
    )
    selected_aggregation = str(
        aggregation_table.iloc[0]["aggregation_mode"]
    )
    selected_validation_predictions = aggregation_predictions[
        selected_aggregation
    ]
    selected_validation_metrics = regression_metrics(
        selected_validation_predictions["cpi_actual"],
        selected_validation_predictions["cpi_predicted"],
    )

    # The test arrays and states are first opened after the configuration freeze.
    test_split = _load_split(data_dir, "test")
    all_states = load_state_lookup(state_dir)
    test_pairs, test_candidates = prepare_evaluation_pairs(
        test_split,
        train_pool,
        selected_result["calibration"],
        selected_config,
    )
    test_pair_predictions, test_predictions = _predict(
        selected_result,
        test_pairs,
        all_states,
        selected_aggregation,
    )
    test_metrics = regression_metrics(
        test_predictions["cpi_actual"],
        test_predictions["cpi_predicted"],
    )
    selected_reference_ids = set(
        selected_result["validation_pairs"]["sample_j_id"].astype(int)
    ).union(test_pairs["sample_j_id"].astype(int))
    if not selected_reference_ids.issubset(train_ids):
        raise ValueError("validation/test reference escaped the closed train50 bank")

    ordinary_metrics = pd.read_csv(
        ordinary_dir / "tables" / "optical_reservoir_metrics.csv"
    ).set_index("split")
    current_metrics = pd.read_csv(
        current_ssa_dir / "tables" / "model_comparison.csv"
    ).set_index("model")
    comparison = pd.DataFrame(
        [
            {
                "model": "ordinary_optical_reservoir",
                "val_mae": ordinary_metrics.loc["val", "mae"],
                "val_rmse": ordinary_metrics.loc["val", "rmse"],
                "test_mae": ordinary_metrics.loc["test", "mae"],
                "test_rmse": ordinary_metrics.loc["test", "rmse"],
            },
            {
                "model": "current_ssa_siamese",
                "val_mae": current_metrics.loc[
                    "siamese_closed50_ssa", "val_mae"
                ],
                "val_rmse": current_metrics.loc[
                    "siamese_closed50_ssa", "val_rmse"
                ],
                "test_mae": current_metrics.loc[
                    "siamese_closed50_ssa", "test_mae"
                ],
                "test_rmse": current_metrics.loc[
                    "siamese_closed50_ssa", "test_rmse"
                ],
            },
            {
                "model": "pair_optimized_siamese",
                "val_mae": selected_validation_metrics["mae"],
                "val_rmse": selected_validation_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            },
        ]
    )
    ordinary_predictions = pd.read_csv(
        ordinary_dir / "tables" / "optical_reservoir_predictions_test.csv"
    )[["sample_id", "target_date", "cpi_actual", "cpi_predicted"]].rename(
        columns={
            "sample_id": "sample_i_id",
            "cpi_predicted": "cpi_predicted_ordinary",
        }
    )
    current_predictions = pd.read_csv(
        current_ssa_dir / "tables" / "ssa_selected_test_predictions.csv"
    )[["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]].rename(
        columns={"cpi_predicted": "cpi_predicted_current_ssa"}
    )
    optimized_predictions = test_predictions[
        ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(columns={"cpi_predicted": "cpi_predicted_pair_optimized"})
    unified = ordinary_predictions.merge(
        current_predictions,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    ).merge(
        optimized_predictions,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    )
    if len(unified) != 47:
        raise ValueError(f"test comparison has {len(unified)} rows, expected 47")

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    validation_table.to_csv(
        table_dir / "validation_pair_structure_comparison.csv",
        index=False,
    )
    pd.DataFrame(alpha_rows).to_csv(
        table_dir / "validation_alpha_trials.csv",
        index=False,
    )
    aggregation_table.to_csv(
        table_dir / "validation_aggregation_comparison.csv",
        index=False,
    )
    comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    unified.to_csv(table_dir / "test_prediction_comparison.csv", index=False)
    selected_result["train_candidates"].to_csv(
        table_dir / "selected_train_candidates.csv",
        index=False,
    )
    selected_result["train_pairs"].to_csv(
        table_dir / "selected_train_pairs.csv",
        index=False,
    )
    selected_result["validation_pairs"].to_csv(
        table_dir / "selected_validation_pairs.csv",
        index=False,
    )
    selected_validation_predictions.to_csv(
        table_dir / "selected_validation_predictions.csv",
        index=False,
    )
    selected_result["validation_pair_predictions"].to_csv(
        table_dir / "selected_validation_pair_predictions.csv",
        index=False,
    )
    test_candidates.to_csv(table_dir / "selected_test_candidates.csv", index=False)
    test_pairs.to_csv(table_dir / "selected_test_pairs.csv", index=False)
    test_pair_predictions.to_csv(
        table_dir / "selected_test_pair_predictions.csv",
        index=False,
    )
    test_predictions.to_csv(
        table_dir / "selected_test_predictions.csv",
        index=False,
    )
    save_model(
        ReadoutBundle(
            scaler=selected_result["scaler"],
            model=selected_result["model"],
            feature_mode="signed_diff",
            aggregation=selected_aggregation,
        ),
        model_dir / "pair_optimized_siamese_readout.npz",
    )
    _save_figures(validation_table, comparison, unified, figure_dir)

    ordinary_test_rmse = float(ordinary_metrics.loc["test", "rmse"])
    current_test_rmse = float(
        current_metrics.loc["siamese_closed50_ssa", "test_rmse"]
    )
    manifest = {
        "experiment": "MoM strict-closed50 Siamese pair-structure optimization",
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_status": (
            "exploratory: the repository's 47-month test result had already "
            "been observed before this structural experiment"
        ),
        "profile_audit": audit,
        "frozen_optical_reservoir": True,
        "reservoir_parameters_changed": False,
        "accessible_train_windows": 50,
        "searched_siamese_only_factors": {
            "target_gap_months": list(target_gaps),
            "training_modes": list(training_modes),
        },
        "fixed_parameters_from_previous_validation_search": {
            "k_references": K_REFERENCES,
            "level_weight": LEVEL_WEIGHT,
            "distance_power": DISTANCE_POWER,
            "feature_mode": "signed_diff_50d",
        },
        "selected_configuration": asdict(selected_config),
        "selected_alpha": selected_result["selected_alpha"],
        "selected_aggregation": selected_aggregation,
        "candidate_train_pairs": len(selected_result["train_candidates"]),
        "selected_train_rows": len(selected_result["train_pairs"]),
        "causal_train_pair_targets": int(
            selected_result["train_candidates"]["sample_i_id"].nunique()
        ),
        "validation_pairs": len(selected_result["validation_pairs"]),
        "test_pairs": len(test_pairs),
        "all_validation_test_references_inside_train50": True,
        "validation_or_test_labels_used_for_reference_selection": False,
        "test_loaded_after_configuration_freeze": True,
        "pair_optimized_validation_mae": selected_validation_metrics["mae"],
        "pair_optimized_validation_rmse": selected_validation_metrics["rmse"],
        "pair_optimized_test_mae": test_metrics["mae"],
        "pair_optimized_test_rmse": test_metrics["rmse"],
        "test_rmse_change_percent_vs_ordinary": (
            (test_metrics["rmse"] / ordinary_test_rmse - 1.0) * 100.0
        ),
        "test_rmse_change_percent_vs_current_ssa": (
            (test_metrics["rmse"] / current_test_rmse - 1.0) * 100.0
        ),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# 环比严格封闭50样本：孪生配对结构优化",
                "",
                "本实验只改变孪生模型的配对规则和训练权重，不改变光储备池、状态或50个月数据预算。",
                "配置只根据45个月验证集选择；测试集随后评估一次。由于旧测试结果已经看过，本结果属于探索性分析。",
                "",
                f"- 最终配置：target_gap={selected_config.target_gap_months}，"
                f"training_mode={selected_config.training_mode}",
                f"- alpha={selected_result['selected_alpha']}",
                f"- 参考聚合={selected_aggregation}",
                f"- 候选训练对={len(selected_result['train_candidates'])}，"
                f"训练行={len(selected_result['train_pairs'])}，"
                f"覆盖配对目标={selected_result['train_candidates']['sample_i_id'].nunique()}",
                f"- 验证MAE/RMSE={selected_validation_metrics['mae']:.6f}/"
                f"{selected_validation_metrics['rmse']:.6f}",
                f"- 测试MAE/RMSE={test_metrics['mae']:.6f}/"
                f"{test_metrics['rmse']:.6f}",
                f"- 测试RMSE相对单光变化="
                f"{(test_metrics['rmse'] / ordinary_test_rmse - 1.0) * 100.0:+.2f}%",
                f"- 测试RMSE相对当前SSA孪生变化="
                f"{(test_metrics['rmse'] / current_test_rmse - 1.0) * 100.0:+.2f}%",
                "",
                "详细验证消融见 `tables/validation_pair_structure_comparison.csv`。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_pair_optimization(output_dir=args.output_dir)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
