"""Select the Siamese optical-reservoir configuration on validation data only.

This module is intentionally isolated from the existing V0 runner. It loads
only train/validation states, keeps the frozen training pairs, rebuilds a full
leak-free validation reference pool, and compares a small predefined grid.
No test state, test pair table, or test metric is read or produced here.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, ROOT_DIR
from src.create_siamese_pairs import MIN_GAP_MONTHS, _build_candidate_pairs
from src.siamese_reservoir_regression import (
    DEFAULT_ALPHAS,
    predict_pairs,
    regression_metrics,
    train_readout,
)


DEFAULT_STATE_DIR = ROOT_DIR / "matlab" / "optical_reservoir_cpi" / "states"
REFERENCE_STRATEGIES = ("window_distance", "recent")
REFERENCE_COUNTS = (1, 3, 5, 10)
FEATURE_MODES = ("signed_diff", "signed_abs")
AGGREGATIONS = ("mean", "inverse_distance")
VALIDATION_STATE_SPLITS = ("train", "val")


def _load_split(data_dir: Path, split: str) -> dict[str, object]:
    """Load one frozen window split and its global sample metadata."""
    sample_index = pd.read_csv(data_dir / "sample_index.csv")
    index = (
        sample_index.loc[sample_index["split"] == split]
        .copy()
        .reset_index(drop=True)
    )
    X = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy")
    if len(X) != len(y) or len(X) != len(index):
        raise ValueError(
            f"Frozen {split} split is misaligned: X={len(X)}, y={len(y)}, "
            f"index={len(index)}"
        )
    if not np.allclose(y, index["y"].to_numpy(dtype=float)):
        raise ValueError(f"Frozen {split} labels do not match sample_index.csv")
    return {"X": X, "y": y, "index": index}


def build_validation_candidates(
    data_dir: Path = DATA_PROCESSED_DIR,
) -> pd.DataFrame:
    """Build every legal validation pair from train and earlier validation data."""
    train = _load_split(data_dir, "train")
    val = _load_split(data_dir, "val")
    candidates = pd.concat(
        [
            _build_candidate_pairs(val, train, min_gap_months=MIN_GAP_MONTHS),
            _build_candidate_pairs(val, val, min_gap_months=MIN_GAP_MONTHS),
        ],
        ignore_index=True,
    )
    if candidates.empty:
        raise ValueError("Validation candidate pool is empty")
    if candidates.duplicated(["sample_i_id", "sample_j_id"]).any():
        raise ValueError("Validation candidate pool contains duplicate pairs")

    target_i = pd.to_datetime(candidates["target_i_date"])
    target_j = pd.to_datetime(candidates["target_j_date"])
    start_i = pd.to_datetime(candidates["x_i_start_date"])
    end_j = pd.to_datetime(candidates["x_j_end_date"])
    gaps = (
        (start_i.dt.year - end_j.dt.year) * 12
        + start_i.dt.month
        - end_j.dt.month
    )
    if not (target_j < target_i).all() or not (gaps >= MIN_GAP_MONTHS).all():
        raise ValueError("Validation candidate pool violates the time boundary")
    return candidates


def select_validation_references(
    candidates: pd.DataFrame,
    strategy: str,
    k: int,
) -> pd.DataFrame:
    """Choose references using input-derived distance or known reference dates only.

    The selection keys deliberately exclude cpi_i, cpi_j, delta_cpi, and every
    prediction/error field. Labels remain in the returned table only because
    they are needed after selection to evaluate validation predictions.
    """
    if strategy not in REFERENCE_STRATEGIES:
        raise ValueError(f"Unknown reference strategy: {strategy}")
    if k <= 0:
        raise ValueError("k must be positive")

    if strategy == "window_distance":
        ordered = candidates.sort_values(
            ["sample_i_id", "window_distance", "target_j_date", "sample_j_id"],
            ascending=[True, True, True, True],
        )
    else:
        ordered = candidates.sort_values(
            ["sample_i_id", "target_j_date", "sample_j_id"],
            ascending=[True, False, False],
        )

    selected = (
        ordered.groupby("sample_i_id", as_index=False, group_keys=False)
        .head(k)
        .copy()
        .reset_index(drop=True)
    )
    selected["selection_method"] = f"{strategy}_validation_only"
    group_sizes = selected.groupby("sample_i_id").size()
    if len(group_sizes) != candidates["sample_i_id"].nunique():
        raise ValueError("Reference selection dropped one or more validation targets")
    if not (group_sizes == k).all():
        raise ValueError(f"At least one validation target has fewer than {k} references")
    return selected


def load_validation_state_lookup(state_dir: Path) -> dict[int, np.ndarray]:
    """Load only train and validation state caches; never open the test cache."""
    lookup: dict[int, np.ndarray] = {}
    state_width: int | None = None
    for split in VALIDATION_STATE_SPLITS:
        state_path = state_dir / f"states_{split}.mat"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing validation-search state file: {state_path}")
        data = loadmat(state_path)
        states = np.asarray(data["state_matrix"], dtype=float)
        sample_ids = np.asarray(data["sample_id"]).reshape(-1).astype(int)
        if states.ndim != 2 or len(states) != len(sample_ids):
            raise ValueError(f"Invalid state shapes in {state_path}")
        if not np.isfinite(states).all():
            raise ValueError(f"Non-finite states in {state_path}")
        if state_width is None:
            state_width = states.shape[1]
        elif states.shape[1] != state_width:
            raise ValueError("Train and validation states use different widths")
        for sample_id, state in zip(sample_ids, states):
            if int(sample_id) in lookup:
                raise ValueError(f"Duplicate sample_id in validation states: {sample_id}")
            lookup[int(sample_id)] = state
    return lookup


def _configuration_key(row: dict[str, object]) -> tuple[object, ...]:
    """Rank by validation evidence, then prefer the simpler tied configuration."""
    feature_rank = {"signed_diff": 0, "signed_abs": 1}
    aggregation_rank = {"mean": 0, "inverse_distance": 1}
    strategy_rank = {"window_distance": 0, "recent": 1}
    return (
        float(row["val_rmse"]),
        float(row["val_mae"]),
        feature_rank[str(row["feature_mode"])],
        aggregation_rank[str(row["aggregation"])],
        int(row["k"]),
        strategy_rank[str(row["reference_strategy"])],
        float(row["alpha"]),
    )


def run_validation_search(
    state_dir: Path = DEFAULT_STATE_DIR,
    data_dir: Path = DATA_PROCESSED_DIR,
    output_dir: Path = RESULTS_DIR,
    reference_strategies: Iterable[str] = REFERENCE_STRATEGIES,
    reference_counts: Iterable[int] = REFERENCE_COUNTS,
    feature_modes: Iterable[str] = FEATURE_MODES,
    aggregations: Iterable[str] = AGGREGATIONS,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the predefined validation-only grid and save its frozen winner."""
    train_pairs = pd.read_csv(data_dir / "pair_indices_train.csv")
    candidates = build_validation_candidates(data_dir)
    state_lookup = load_validation_state_lookup(state_dir)

    rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    selected_cache: dict[tuple[str, int], pd.DataFrame] = {}
    prediction_cache: dict[tuple[str, int, str, str], pd.DataFrame] = {}

    grid = itertools.product(
        reference_strategies, reference_counts, feature_modes, aggregations
    )
    for strategy, k, feature_mode, aggregation in grid:
        pair_key = (str(strategy), int(k))
        if pair_key not in selected_cache:
            selected_cache[pair_key] = select_validation_references(
                candidates, str(strategy), int(k)
            )
        val_pairs = selected_cache[pair_key]
        bundle, alpha_trials = train_readout(
            {"train": train_pairs, "val": val_pairs},
            state_lookup,
            feature_mode=str(feature_mode),
            aggregation=str(aggregation),
            alphas=alphas,
        )
        _, target_predictions = predict_pairs(bundle, val_pairs, state_lookup)
        metrics = regression_metrics(
            target_predictions["cpi_actual"],
            target_predictions["cpi_predicted"],
        )
        group_sizes = val_pairs.groupby("sample_i_id").size()
        row = {
            "reference_strategy": str(strategy),
            "k": int(k),
            "feature_mode": str(feature_mode),
            "aggregation": str(aggregation),
            "alpha": float(bundle.model.alpha),
            "num_train_pairs": int(len(train_pairs)),
            "num_val_pairs": int(len(val_pairs)),
            "num_val_targets": int(len(target_predictions)),
            "references_per_target_min": int(group_sizes.min()),
            "references_per_target_max": int(group_sizes.max()),
            "val_mae": metrics["mae"],
            "val_rmse": metrics["rmse"],
        }
        rows.append(row)
        prediction_cache[(str(strategy), int(k), str(feature_mode), str(aggregation))] = (
            target_predictions
        )
        for trial in alpha_trials:
            alpha_rows.append(
                {
                    "reference_strategy": str(strategy),
                    "k": int(k),
                    "feature_mode": str(feature_mode),
                    "aggregation": str(aggregation),
                    "alpha": float(trial["alpha"]),
                    "val_mae": float(trial["mae"]),
                    "val_rmse": float(trial["rmse"]),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["val_rmse", "val_mae", "feature_mode", "aggregation", "k"],
        kind="stable",
    ).reset_index(drop=True)
    best = min(rows, key=_configuration_key)
    best_key = (
        str(best["reference_strategy"]),
        int(best["k"]),
        str(best["feature_mode"]),
        str(best["aggregation"]),
    )
    best_pairs = selected_cache[(best_key[0], best_key[1])]
    best_predictions = prediction_cache[best_key]

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        table_dir / "siamese_validation_configuration_search.csv", index=False
    )
    pd.DataFrame(alpha_rows).to_csv(
        table_dir / "siamese_validation_alpha_trials.csv", index=False
    )
    best_pairs.to_csv(
        table_dir / "siamese_validation_selected_pairs.csv", index=False
    )
    best_predictions.to_csv(
        table_dir / "siamese_validation_selected_predictions.csv", index=False
    )

    selected_configuration: dict[str, object] = {
        "status": "validation_selected_not_tested",
        "reference_strategy": best["reference_strategy"],
        "k": best["k"],
        "feature_mode": best["feature_mode"],
        "aggregation": best["aggregation"],
        "alpha": best["alpha"],
        "validation_mae": best["val_mae"],
        "validation_rmse": best["val_rmse"],
        "num_train_pairs": best["num_train_pairs"],
        "num_validation_pairs": best["num_val_pairs"],
        "num_validation_targets": best["num_val_targets"],
        "selection_primary_metric": "validation target-level RMSE",
        "selection_tie_break": "MAE, simpler feature/aggregation, smaller K",
        "reference_information_boundary": (
            "input-window distance or already-published reference date only"
        ),
        "test_state_loaded": False,
        "test_pairs_loaded": False,
        "test_evaluated": False,
    }
    with (table_dir / "siamese_validation_selected_configuration.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(selected_configuration, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return results, selected_configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, selected = run_validation_search(
        state_dir=args.state_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(results.to_string(index=False))
    print("\nSelected validation configuration:")
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
