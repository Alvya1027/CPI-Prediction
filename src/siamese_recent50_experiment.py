"""Run the Siamese optical-reservoir experiment with 50 recent train targets.

The supervised target set is the last 50 targets from the original chronological
training split. Older training windows remain available only as already-known
historical references, which is the intended reference-based prediction setting.
Validation/test targets, optical states, reference strategy, feature mode, and
aggregation remain unchanged. Ridge alpha is reselected on validation data.

Outputs are isolated under results/siamese_optical_recent50_20260723 and never
overwrite the frozen stage-3 results.
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
from scipy.io import loadmat

from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, ROOT_DIR
from src.siamese_reservoir_regression import (
    DEFAULT_ALPHAS,
    load_state_lookup,
    predict_pairs,
    regression_metrics,
    save_model,
    train_readout,
)


RECENT_TRAIN_TARGETS = 50
BASE_STATE_DIR = ROOT_DIR / "matlab" / "optical_reservoir_cpi" / "states"
ORDINARY_EXPERIMENT_DIR = RESULTS_DIR / "optical_reservoir_recent50_20260723"
RECENT_STATE_DIR = ORDINARY_EXPERIMENT_DIR / "input_states_recent50"
OUTPUT_DIR = RESULTS_DIR / "siamese_optical_recent50_20260723"
FROZEN_CONFIG_PATH = (
    RESULTS_DIR / "tables" / "siamese_validation_selected_configuration.json"
)
VALIDATION_PAIRS_PATH = (
    RESULTS_DIR / "tables" / "siamese_validation_selected_pairs.csv"
)
TEST_PAIRS_PATH = RESULTS_DIR / "tables" / "final_siamese_test_selected_pairs.csv"
TRAIN_PAIRS_PATH = DATA_PROCESSED_DIR / "pair_indices_train.csv"


def select_recent_training_targets(
    sample_index: pd.DataFrame,
    count: int = RECENT_TRAIN_TARGETS,
) -> pd.DataFrame:
    """Return the last ``count`` targets from the chronological train split."""
    required = {"sample_id", "split", "target_date"}
    missing = required.difference(sample_index.columns)
    if missing:
        raise ValueError(f"sample index is missing columns: {sorted(missing)}")
    if count <= 0:
        raise ValueError("count must be positive")

    train = sample_index.loc[sample_index["split"] == "train"].copy()
    train["_target_period"] = pd.PeriodIndex(train["target_date"], freq="M")
    train = train.sort_values(["_target_period", "sample_id"]).reset_index(drop=True)
    if len(train) < count:
        raise ValueError(f"train split contains only {len(train)} targets, need {count}")

    recent = train.tail(count).copy().reset_index(drop=True)
    periods = recent["_target_period"].astype("int64").to_numpy()
    if len(periods) > 1 and not np.all(np.diff(periods) == 1):
        raise ValueError("recent training targets are not consecutive months")
    return recent.drop(columns="_target_period")


def filter_training_pairs(
    train_pairs: pd.DataFrame,
    target_ids: Iterable[int],
) -> pd.DataFrame:
    """Keep pairs whose supervised target is one of the recent target IDs.

    References are deliberately not restricted to the 50 target IDs. A reference
    supplies an already-known historical CPI anchor; it is not an additional
    supervised target for the readout.
    """
    required = {"sample_i_id", "sample_j_id", "delta_cpi", "cpi_j"}
    missing = required.difference(train_pairs.columns)
    if missing:
        raise ValueError(f"training pair table is missing: {sorted(missing)}")

    target_set = {int(value) for value in target_ids}
    selected = (
        train_pairs.loc[train_pairs["sample_i_id"].astype(int).isin(target_set)]
        .copy()
        .reset_index(drop=True)
    )
    covered = set(selected["sample_i_id"].astype(int))
    if covered != target_set:
        missing_targets = sorted(target_set.difference(covered))
        raise ValueError(
            f"recent targets without frozen training pairs: {missing_targets[:10]}"
        )
    if selected.duplicated(["sample_i_id", "sample_j_id"]).any():
        raise ValueError("recent50 training pairs contain duplicates")
    return selected


def verify_recent_state_cache(
    base_state_dir: Path,
    recent_state_dir: Path,
) -> dict[str, dict[str, object]]:
    """Verify the colleague's isolated states are exact subsets of base states."""
    summary: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        base = loadmat(base_state_dir / f"states_{split}.mat")
        recent = loadmat(recent_state_dir / f"states_{split}.mat")
        base_states = np.asarray(base["state_matrix"], dtype=float)
        base_ids = np.asarray(base["sample_id"]).reshape(-1).astype(int)
        recent_states = np.asarray(recent["state_matrix"], dtype=float)
        recent_ids = np.asarray(recent["sample_id"]).reshape(-1).astype(int)
        positions = {int(sample_id): row for row, sample_id in enumerate(base_ids)}
        missing = sorted(set(recent_ids).difference(positions))
        if missing:
            raise ValueError(f"{split} recent state IDs are absent from base: {missing}")
        expected = np.vstack([base_states[positions[int(value)]] for value in recent_ids])
        maximum_difference = float(np.max(np.abs(expected - recent_states)))
        if maximum_difference != 0.0:
            raise ValueError(
                f"{split} recent state cache differs from base states: "
                f"max abs diff={maximum_difference}"
            )
        summary[split] = {
            "num_targets": int(len(recent_ids)),
            "sample_id_min": int(recent_ids.min()),
            "sample_id_max": int(recent_ids.max()),
            "state_width": int(recent_states.shape[1]),
            "max_abs_difference_from_base": maximum_difference,
        }
    return summary


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return regression_metrics(
        actual.to_numpy(dtype=float),
        predicted.to_numpy(dtype=float),
    )


def _save_comparison_figures(predictions: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(predictions["target_date"], predictions["cpi_actual"], label="Actual CPI")
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_ordinary_recent50"],
        label="Ordinary reservoir: recent 50",
    )
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_siamese_recent50"],
        label="Siamese reservoir: recent 50 targets",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI")
    axis.set_title("Recent-50 optical reservoir CPI test predictions")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "recent50_test_prediction_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.plot(
        predictions["target_date"],
        predictions["residual_ordinary_recent50"],
        label="Ordinary reservoir: recent 50",
    )
    axis.plot(
        predictions["target_date"],
        predictions["residual_siamese_recent50"],
        label="Siamese reservoir: recent 50 targets",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("Prediction - actual")
    axis.set_title("Recent-50 optical reservoir CPI test residuals")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "recent50_test_residual_comparison.png", dpi=180)
    plt.close(fig)


def run_recent50_experiment(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_PROCESSED_DIR,
    base_state_dir: Path = BASE_STATE_DIR,
    recent_state_dir: Path = RECENT_STATE_DIR,
    ordinary_experiment_dir: Path = ORDINARY_EXPERIMENT_DIR,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> pd.DataFrame:
    """Train/evaluate recent-50 Siamese and compare with colleague's ordinary run."""
    with FROZEN_CONFIG_PATH.open("r", encoding="utf-8") as file:
        frozen_config = json.load(file)
    feature_mode = str(frozen_config["feature_mode"])
    aggregation = str(frozen_config["aggregation"])

    sample_index = pd.read_csv(data_dir / "sample_index.csv")
    recent_targets = select_recent_training_targets(sample_index)
    target_ids = recent_targets["sample_id"].astype(int).tolist()

    all_train_pairs = pd.read_csv(TRAIN_PAIRS_PATH)
    train_pairs = filter_training_pairs(all_train_pairs, target_ids)
    val_pairs = pd.read_csv(VALIDATION_PAIRS_PATH)
    test_pairs = pd.read_csv(TEST_PAIRS_PATH)
    state_cache_check = verify_recent_state_cache(base_state_dir, recent_state_dir)
    state_lookup = load_state_lookup(base_state_dir)

    bundle, alpha_trials = train_readout(
        {"train": train_pairs, "val": val_pairs},
        state_lookup,
        feature_mode=feature_mode,
        aggregation=aggregation,
        alphas=alphas,
    )

    pair_outputs: dict[str, pd.DataFrame] = {}
    target_outputs: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, object]] = []
    for split, pairs in (
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ):
        pair_predictions, target_predictions = predict_pairs(bundle, pairs, state_lookup)
        metrics = _metrics(
            target_predictions["cpi_actual"],
            target_predictions["cpi_predicted"],
        )
        metric_rows.append(
            {
                "split": split,
                "num_supervised_targets": int(len(target_predictions)),
                "num_pairs": int(len(pair_predictions)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        )
        pair_outputs[split] = pair_predictions
        target_outputs[split] = target_predictions

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
    comparison = pd.DataFrame(
        [
            {
                "model": "ordinary_optical_reservoir_recent50",
                "num_supervised_train_targets": 50,
                "num_train_pairs": 0,
                "alpha": float(ordinary_summary["selected_alpha"]),
                "val_mae": float(ordinary_metrics.loc["val", "mae"]),
                "val_rmse": float(ordinary_metrics.loc["val", "rmse"]),
                "test_mae": float(ordinary_metrics.loc["test", "mae"]),
                "test_rmse": float(ordinary_metrics.loc["test", "rmse"]),
            },
            {
                "model": "siamese_optical_reservoir_recent50_targets",
                "num_supervised_train_targets": 50,
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
        columns={"cpi_predicted": "cpi_predicted_siamese_recent50"}
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
                "cpi_predicted_siamese_recent50",
                "num_references",
                "reference_prediction_std",
            ]
        ],
        on=["sample_i_id", "target_date", "cpi_actual"],
        how="inner",
        validate="one_to_one",
    )
    if len(unified) != 47:
        raise ValueError(f"test comparison has {len(unified)} targets, expected 47")
    for name in ("ordinary_recent50", "siamese_recent50"):
        unified[f"residual_{name}"] = (
            unified[f"cpi_predicted_{name}"] - unified["cpi_actual"]
        )
        unified[f"absolute_error_{name}"] = unified[f"residual_{name}"].abs()

    recent_target_set = set(target_ids)
    reference_ids = set(train_pairs["sample_j_id"].astype(int))
    inside_reference_pairs = int(
        train_pairs["sample_j_id"].astype(int).isin(recent_target_set).sum()
    )
    reference_summary = {
        "unique_historical_reference_windows": int(len(reference_ids)),
        "reference_pairs_inside_recent50": inside_reference_pairs,
        "reference_pairs_before_recent50": int(len(train_pairs) - inside_reference_pairs),
        "oldest_reference_sample_id": int(min(reference_ids)),
        "newest_reference_sample_id": int(max(reference_ids)),
    }

    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    model_dir = output_dir / "models"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_pairs.to_csv(table_dir / "siamese_recent50_train_pairs.csv", index=False)
    siamese_metrics.to_csv(table_dir / "siamese_recent50_metrics.csv", index=False)
    pd.DataFrame(alpha_trials).to_csv(
        table_dir / "siamese_recent50_alpha_selection.csv", index=False
    )
    for split in ("train", "val", "test"):
        pair_outputs[split].to_csv(
            table_dir / f"siamese_recent50_pair_predictions_{split}.csv",
            index=False,
        )
        target_outputs[split].to_csv(
            table_dir / f"siamese_recent50_predictions_{split}.csv",
            index=False,
        )
    comparison.to_csv(table_dir / "recent50_model_comparison.csv", index=False)
    unified.to_csv(table_dir / "recent50_test_prediction_comparison.csv", index=False)
    save_model(bundle, model_dir / "siamese_recent50_readout.npz")
    _save_comparison_figures(unified, figure_dir)

    manifest = {
        "experiment": "siamese optical reservoir with last 50 supervised train targets",
        "evaluation_status": "exploratory re-analysis; current test split was seen before",
        "train_selection": "last 50 target windows of original chronological train split",
        "train_target_count": int(len(recent_targets)),
        "train_target_sample_id_min": int(recent_targets["sample_id"].min()),
        "train_target_sample_id_max": int(recent_targets["sample_id"].max()),
        "train_target_date_min": str(recent_targets["target_date"].iloc[0]),
        "train_target_date_max": str(recent_targets["target_date"].iloc[-1]),
        "validation_targets": 45,
        "test_targets": 47,
        "num_train_pairs": int(len(train_pairs)),
        "num_train_pair_targets": int(train_pairs["sample_i_id"].nunique()),
        "historical_reference_policy": (
            "older original-train windows may serve as already-known CPI anchors; "
            "only the recent 50 sample_i targets supervise the readout"
        ),
        "reference_summary": reference_summary,
        "feature_mode": feature_mode,
        "aggregation": aggregation,
        "reference_strategy": frozen_config["reference_strategy"],
        "validation_references_per_target": int(frozen_config["k"]),
        "test_references_per_target": int(frozen_config["k"]),
        "selected_alpha_on_validation": float(bundle.model.alpha),
        "optical_reservoir": "fixed; existing Simulink states reused",
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
    comparison = run_recent50_experiment(output_dir=args.output_dir)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
