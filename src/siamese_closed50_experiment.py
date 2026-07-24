"""Strict closed-pool 50-window Siamese optical-reservoir experiment.

Only the last 50 windows of the original chronological training split may be
used as supervised targets or historical references. Validation and test use
the same fixed 50-window reference bank; no older train window, validation
label, or earlier test label may act as a reference.

This gives the ordinary and Siamese models the same total training-data budget.
The existing 12-month pair-isolation rule is kept unchanged, so some early
windows in the closed pool cannot be Siamese pair targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, RESULTS_DIR
from src.create_siamese_pairs import (
    MIN_GAP_MONTHS,
    _assign_delta_bins,
    _build_candidate_pairs,
    _compute_delta_bin_thresholds,
    _sample_pairs_by_bin,
)
from src.siamese_final_test import select_final_references
from src.siamese_recent50_experiment import (
    BASE_STATE_DIR,
    FROZEN_CONFIG_PATH,
    ORDINARY_EXPERIMENT_DIR,
    RECENT_STATE_DIR,
    RECENT_TRAIN_TARGETS,
    select_recent_training_targets,
    verify_recent_state_cache,
)
from src.siamese_reservoir_regression import (
    DEFAULT_ALPHAS,
    load_state_lookup,
    predict_pairs,
    regression_metrics,
    save_model,
    train_readout,
)
from src.siamese_validation_search import select_validation_references


OUTPUT_DIR = RESULTS_DIR / "siamese_optical_closed50_20260723"


def _load_split(data_dir: Path, split: str) -> dict[str, object]:
    sample_index = pd.read_csv(data_dir / "sample_index.csv")
    index = (
        sample_index.loc[sample_index["split"] == split]
        .copy()
        .reset_index(drop=True)
    )
    X = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy")
    if len(X) != len(y) or len(X) != len(index):
        raise ValueError(f"{split} split is misaligned")
    if not np.allclose(y, index["y"].to_numpy(dtype=float)):
        raise ValueError(f"{split} labels do not match sample_index.csv")
    return {"X": X, "y": y, "index": index}


def build_closed_train_pool(
    data_dir: Path = DATA_PROCESSED_DIR,
    count: int = RECENT_TRAIN_TARGETS,
) -> dict[str, object]:
    """Create a split-like dictionary containing only the latest train windows."""
    full_train = _load_split(data_dir, "train")
    recent_index = select_recent_training_targets(
        full_train["index"], count=count
    ).reset_index(drop=True)
    positions = {
        int(sample_id): row
        for row, sample_id in enumerate(
            full_train["index"]["sample_id"].to_numpy(dtype=int)
        )
    }
    rows = [positions[int(sample_id)] for sample_id in recent_index["sample_id"]]
    return {
        "X": np.asarray(full_train["X"])[rows],
        "y": np.asarray(full_train["y"])[rows],
        "index": recent_index,
    }


def build_closed_training_pairs(
    train_pool: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    """Build training pairs with both i and j restricted to the same 50 windows."""
    candidates = _build_candidate_pairs(
        train_pool,
        train_pool,
        min_gap_months=MIN_GAP_MONTHS,
    )
    if candidates.empty:
        raise ValueError("closed50 training candidate pool is empty")
    pool_ids = set(train_pool["index"]["sample_id"].astype(int))
    if not set(candidates["sample_i_id"].astype(int)).issubset(pool_ids):
        raise ValueError("closed50 training target escaped the 50-window pool")
    if not set(candidates["sample_j_id"].astype(int)).issubset(pool_ids):
        raise ValueError("closed50 training reference escaped the 50-window pool")

    thresholds = _compute_delta_bin_thresholds(candidates)
    binned = _assign_delta_bins(candidates, thresholds)
    selected = _sample_pairs_by_bin(binned).reset_index(drop=True)
    selected["selection_method"] = "closed50_delta_stratified_train"
    if selected.duplicated(["sample_i_id", "sample_j_id"]).any():
        raise ValueError("closed50 selected training pairs contain duplicates")
    return selected, candidates, thresholds


def build_fixed_bank_evaluation_pairs(
    train_pool: dict[str, object],
    evaluation_split: dict[str, object],
    split_name: str,
    strategy: str,
    k: int,
) -> pd.DataFrame:
    """Select validation/test references exclusively from the closed train bank."""
    candidates = _build_candidate_pairs(
        evaluation_split,
        train_pool,
        min_gap_months=MIN_GAP_MONTHS,
    )
    if candidates.empty:
        raise ValueError(f"closed50 {split_name} candidate pool is empty")
    if split_name == "val":
        selected = select_validation_references(candidates, strategy, k)
    elif split_name == "test":
        selected = select_final_references(candidates, strategy, k)
    else:
        raise ValueError("split_name must be val or test")
    selected = selected.copy()
    selected["selection_method"] = (
        f"{strategy}_closed50_fixed_train_reference_bank"
    )

    pool_ids = set(train_pool["index"]["sample_id"].astype(int))
    reference_ids = set(selected["sample_j_id"].astype(int))
    if not reference_ids.issubset(pool_ids):
        raise ValueError(f"{split_name} selected a reference outside closed50")
    return selected.reset_index(drop=True)


def _metric_row(
    split: str,
    pair_predictions: pd.DataFrame,
    target_predictions: pd.DataFrame,
) -> dict[str, object]:
    metrics = regression_metrics(
        target_predictions["cpi_actual"].to_numpy(dtype=float),
        target_predictions["cpi_predicted"].to_numpy(dtype=float),
    )
    return {
        "split": split,
        "num_pair_targets": int(len(target_predictions)),
        "num_pairs": int(len(pair_predictions)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }


def _save_figures(predictions: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(predictions["target_date"], predictions["cpi_actual"], label="Actual CPI")
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_ordinary_recent50"],
        label="Ordinary reservoir: closed 50",
    )
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_siamese_closed50"],
        label="Siamese reservoir: closed 50",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI")
    axis.set_title("Strict closed-50 optical reservoir CPI test predictions")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "closed50_test_prediction_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.plot(
        predictions["target_date"],
        predictions["residual_ordinary_recent50"],
        label="Ordinary reservoir: closed 50",
    )
    axis.plot(
        predictions["target_date"],
        predictions["residual_siamese_closed50"],
        label="Siamese reservoir: closed 50",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("Prediction - actual")
    axis.set_title("Strict closed-50 optical reservoir CPI test residuals")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "closed50_test_residual_comparison.png", dpi=180)
    plt.close(fig)


def run_closed50_experiment(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_PROCESSED_DIR,
    base_state_dir: Path = BASE_STATE_DIR,
    recent_state_dir: Path = RECENT_STATE_DIR,
    ordinary_experiment_dir: Path = ORDINARY_EXPERIMENT_DIR,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> pd.DataFrame:
    """Run strict closed50 Siamese and compare it with ordinary recent50."""
    with FROZEN_CONFIG_PATH.open("r", encoding="utf-8") as file:
        frozen_config = json.load(file)
    strategy = str(frozen_config["reference_strategy"])
    k = int(frozen_config["k"])
    feature_mode = str(frozen_config["feature_mode"])
    aggregation = str(frozen_config["aggregation"])

    train_pool = build_closed_train_pool(data_dir)
    val_split = _load_split(data_dir, "val")
    test_split = _load_split(data_dir, "test")
    train_pairs, train_candidates, thresholds = build_closed_training_pairs(train_pool)
    val_pairs = build_fixed_bank_evaluation_pairs(
        train_pool, val_split, "val", strategy, k
    )
    test_pairs = build_fixed_bank_evaluation_pairs(
        train_pool, test_split, "test", strategy, k
    )

    state_cache_check = verify_recent_state_cache(base_state_dir, recent_state_dir)
    state_lookup = load_state_lookup(recent_state_dir)
    bundle, alpha_trials = train_readout(
        {"train": train_pairs, "val": val_pairs},
        state_lookup,
        feature_mode=feature_mode,
        aggregation=aggregation,
        alphas=alphas,
    )

    pairs_by_split = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    pair_outputs: dict[str, pd.DataFrame] = {}
    target_outputs: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, object]] = []
    for split, pairs in pairs_by_split.items():
        pair_predictions, target_predictions = predict_pairs(bundle, pairs, state_lookup)
        pair_outputs[split] = pair_predictions
        target_outputs[split] = target_predictions
        metric_rows.append(_metric_row(split, pair_predictions, target_predictions))
    siamese_metrics = pd.DataFrame(metric_rows)

    ordinary_metrics = pd.read_csv(
        ordinary_experiment_dir / "tables" / "optical_reservoir_metrics.csv"
    ).set_index("split")
    ordinary_summary = json.loads(
        (
            ordinary_experiment_dir
            / "tables"
            / "optical_reservoir_run_summary.json"
        ).read_text(encoding="utf-8")
    )
    siamese_val = siamese_metrics.loc[siamese_metrics["split"] == "val"].iloc[0]
    siamese_test = siamese_metrics.loc[siamese_metrics["split"] == "test"].iloc[0]
    pair_target_count = int(train_pairs["sample_i_id"].nunique())
    comparison = pd.DataFrame(
        [
            {
                "model": "ordinary_optical_reservoir_closed50",
                "accessible_train_windows": 50,
                "direct_supervised_targets": 50,
                "siamese_pair_targets": 0,
                "num_train_pairs": 0,
                "alpha": float(ordinary_summary["selected_alpha"]),
                "val_mae": float(ordinary_metrics.loc["val", "mae"]),
                "val_rmse": float(ordinary_metrics.loc["val", "rmse"]),
                "test_mae": float(ordinary_metrics.loc["test", "mae"]),
                "test_rmse": float(ordinary_metrics.loc["test", "rmse"]),
            },
            {
                "model": "siamese_optical_reservoir_closed50",
                "accessible_train_windows": 50,
                "direct_supervised_targets": 0,
                "siamese_pair_targets": pair_target_count,
                "num_train_pairs": int(len(train_pairs)),
                "alpha": float(bundle.model.alpha),
                "val_mae": float(siamese_val["mae"]),
                "val_rmse": float(siamese_val["rmse"]),
                "test_mae": float(siamese_test["mae"]),
                "test_rmse": float(siamese_test["rmse"]),
            },
        ]
    )

    ordinary_predictions = pd.read_csv(
        ordinary_experiment_dir
        / "tables"
        / "optical_reservoir_predictions_test.csv"
    )
    siamese_predictions = target_outputs["test"].rename(
        columns={"cpi_predicted": "cpi_predicted_siamese_closed50"}
    )
    unified = ordinary_predictions[
        ["sample_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(
        columns={
            "sample_id": "sample_i_id",
            "cpi_predicted": "cpi_predicted_ordinary_recent50",
        }
    )
    unified = unified.merge(
        siamese_predictions[
            [
                "sample_i_id",
                "target_date",
                "cpi_actual",
                "cpi_predicted_siamese_closed50",
                "num_references",
                "reference_prediction_std",
            ]
        ],
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    )
    if len(unified) != 47:
        raise ValueError(f"closed50 test output has {len(unified)} targets, expected 47")
    unified["residual_ordinary_recent50"] = (
        unified["cpi_predicted_ordinary_recent50"] - unified["cpi_actual"]
    )
    unified["residual_siamese_closed50"] = (
        unified["cpi_predicted_siamese_closed50"] - unified["cpi_actual"]
    )
    unified["absolute_error_ordinary_recent50"] = unified[
        "residual_ordinary_recent50"
    ].abs()
    unified["absolute_error_siamese_closed50"] = unified[
        "residual_siamese_closed50"
    ].abs()

    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    model_dir = output_dir / "models"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_candidates.to_csv(
        table_dir / "closed50_all_legal_train_candidates.csv", index=False
    )
    train_pairs.to_csv(table_dir / "closed50_selected_train_pairs.csv", index=False)
    val_pairs.to_csv(table_dir / "closed50_validation_pairs.csv", index=False)
    test_pairs.to_csv(table_dir / "closed50_test_pairs.csv", index=False)
    siamese_metrics.to_csv(table_dir / "closed50_siamese_metrics.csv", index=False)
    pd.DataFrame(alpha_trials).to_csv(
        table_dir / "closed50_siamese_alpha_selection.csv", index=False
    )
    for split in ("train", "val", "test"):
        pair_outputs[split].to_csv(
            table_dir / f"closed50_pair_predictions_{split}.csv", index=False
        )
        target_outputs[split].to_csv(
            table_dir / f"closed50_predictions_{split}.csv", index=False
        )
    comparison.to_csv(table_dir / "closed50_model_comparison.csv", index=False)
    unified.to_csv(table_dir / "closed50_test_prediction_comparison.csv", index=False)
    save_model(bundle, model_dir / "siamese_closed50_readout.npz")
    _save_figures(unified, figure_dir)

    train_pool_ids = set(train_pool["index"]["sample_id"].astype(int))
    manifest = {
        "experiment": "strict closed 50-window training-data budget",
        "evaluation_status": "exploratory re-analysis; current test split was seen before",
        "accessible_train_windows": 50,
        "accessible_sample_id_min": int(min(train_pool_ids)),
        "accessible_sample_id_max": int(max(train_pool_ids)),
        "accessible_target_date_min": str(
            train_pool["index"]["target_date"].iloc[0]
        ),
        "accessible_target_date_max": str(
            train_pool["index"]["target_date"].iloc[-1]
        ),
        "training_reference_ids_all_inside_closed_pool": bool(
            set(train_pairs["sample_j_id"].astype(int)).issubset(train_pool_ids)
        ),
        "validation_reference_ids_all_inside_closed_pool": bool(
            set(val_pairs["sample_j_id"].astype(int)).issubset(train_pool_ids)
        ),
        "test_reference_ids_all_inside_closed_pool": bool(
            set(test_pairs["sample_j_id"].astype(int)).issubset(train_pool_ids)
        ),
        "older_train_references_used": False,
        "validation_or_test_labels_used_as_references": False,
        "minimum_pair_gap_months": int(MIN_GAP_MONTHS),
        "legal_training_candidates": int(len(train_candidates)),
        "legal_training_pair_targets": int(
            train_candidates["sample_i_id"].nunique()
        ),
        "selected_training_pairs": int(len(train_pairs)),
        "selected_training_pair_targets": pair_target_count,
        "closed_pool_windows_without_legal_earlier_reference": int(
            len(train_pool_ids) - pair_target_count
        ),
        "delta_bin_thresholds": {
            "q33": float(thresholds[0]),
            "q67": float(thresholds[1]),
        },
        "reference_strategy": strategy,
        "k_validation_test": k,
        "feature_mode": feature_mode,
        "aggregation": aggregation,
        "selected_alpha_on_validation": float(bundle.model.alpha),
        "optical_reservoir": "fixed; colleague recent50 state cache reused",
        "recent_state_cache_check": state_cache_check,
        "ordinary_comparison_directory": str(ordinary_experiment_dir),
    }
    with (output_dir / "experiment_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_closed50_experiment(output_dir=args.output_dir)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
