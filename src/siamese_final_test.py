"""Run the frozen Siamese configuration on the test split exactly once.

The selected Siamese configuration is loaded from the validation-stage JSON
and is never changed using test results. The script also trains two ordinary
optical-reservoir controls: all 212 training targets and the 189 targets that
appear as Siamese training targets. A final summary file acts as a rerun guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, ROOT_DIR
from src.create_siamese_pairs import MIN_GAP_MONTHS, _build_candidate_pairs
from src.optical_reservoir_regression import load_state_splits
from src.siamese_reservoir_regression import (
    DEFAULT_ALPHAS,
    ReadoutBundle,
    build_pair_features,
    load_state_lookup,
    predict_pairs,
    regression_metrics,
)


DEFAULT_STATE_DIR = ROOT_DIR / "matlab" / "optical_reservoir_cpi" / "states"
DEFAULT_CONFIG_PATH = (
    RESULTS_DIR / "tables" / "siamese_validation_selected_configuration.json"
)
FINAL_SUMMARY_NAME = "final_test_run_summary.json"


@dataclass
class OrdinaryBundle:
    scaler: StandardScaler
    model: Ridge


def load_frozen_configuration(config_path: Path) -> dict[str, object]:
    """Load and validate the configuration selected before final testing."""
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    required = {
        "status",
        "reference_strategy",
        "k",
        "feature_mode",
        "aggregation",
        "alpha",
        "validation_mae",
        "validation_rmse",
        "test_evaluated",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Frozen configuration is missing: {sorted(missing)}")
    if config["status"] != "validation_selected_not_tested":
        raise ValueError("Configuration was not frozen at the validation-only stage")
    if bool(config["test_evaluated"]):
        raise ValueError("Frozen configuration already indicates test evaluation")
    return config


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
        raise ValueError(f"Frozen {split} split is misaligned")
    if not np.allclose(y, index["y"].to_numpy(dtype=float)):
        raise ValueError(f"Frozen {split} labels do not match sample_index.csv")
    return {"X": X, "y": y, "index": index}


def build_test_candidates(data_dir: Path = DATA_PROCESSED_DIR) -> pd.DataFrame:
    """Build all temporally legal references for each frozen test target."""
    train = _load_split(data_dir, "train")
    val = _load_split(data_dir, "val")
    test = _load_split(data_dir, "test")
    candidates = pd.concat(
        [
            _build_candidate_pairs(test, train, min_gap_months=MIN_GAP_MONTHS),
            _build_candidate_pairs(test, val, min_gap_months=MIN_GAP_MONTHS),
            _build_candidate_pairs(test, test, min_gap_months=MIN_GAP_MONTHS),
        ],
        ignore_index=True,
    )
    if candidates.empty:
        raise ValueError("Final test candidate pool is empty")
    if candidates.duplicated(["sample_i_id", "sample_j_id"]).any():
        raise ValueError("Final test candidate pool contains duplicate pairs")
    return candidates


def select_final_references(
    candidates: pd.DataFrame,
    strategy: str,
    k: int,
) -> pd.DataFrame:
    """Apply the frozen label-independent reference rule to final test targets."""
    if strategy == "window_distance":
        ordered = candidates.sort_values(
            ["sample_i_id", "window_distance", "target_j_date", "sample_j_id"],
            ascending=[True, True, True, True],
        )
    elif strategy == "recent":
        ordered = candidates.sort_values(
            ["sample_i_id", "target_j_date", "sample_j_id"],
            ascending=[True, False, False],
        )
    else:
        raise ValueError(f"Unknown frozen reference strategy: {strategy}")

    selected = (
        ordered.groupby("sample_i_id", as_index=False, group_keys=False)
        .head(int(k))
        .copy()
        .reset_index(drop=True)
    )
    selected["selection_method"] = f"{strategy}_frozen_final_test"
    sizes = selected.groupby("sample_i_id").size()
    if len(sizes) != candidates["sample_i_id"].nunique():
        raise ValueError("Final reference selection dropped a test target")
    if not (sizes == int(k)).all():
        raise ValueError(f"At least one test target has fewer than {k} references")
    return selected


def _fit_frozen_siamese(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
    config: dict[str, object],
) -> tuple[ReadoutBundle, pd.DataFrame, pd.DataFrame]:
    train_features = build_pair_features(
        train_pairs, state_lookup, str(config["feature_mode"])
    )
    scaler = StandardScaler().fit(train_features)
    model = Ridge(alpha=float(config["alpha"])).fit(
        scaler.transform(train_features),
        train_pairs["delta_cpi"].to_numpy(dtype=float),
    )
    bundle = ReadoutBundle(
        scaler=scaler,
        model=model,
        feature_mode=str(config["feature_mode"]),
        aggregation=str(config["aggregation"]),
    )
    pair_predictions, target_predictions = predict_pairs(
        bundle, test_pairs, state_lookup
    )
    return bundle, pair_predictions, target_predictions


def _state_matrix(table: pd.DataFrame) -> np.ndarray:
    return np.vstack(table["state"].to_numpy())


def _fit_ordinary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> tuple[OrdinaryBundle, list[dict[str, float]], dict[str, float]]:
    train_states = _state_matrix(train)
    train_targets = train["y"].to_numpy(dtype=float)
    val_states = _state_matrix(val)
    val_targets = val["y"].to_numpy(dtype=float)
    scaler = StandardScaler().fit(train_states)
    train_scaled = scaler.transform(train_states)
    val_scaled = scaler.transform(val_states)
    trials: list[dict[str, float]] = []
    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(train_scaled, train_targets)
        metrics = regression_metrics(val_targets, model.predict(val_scaled))
        trials.append({"alpha": float(alpha), **metrics})
    best = min(trials, key=lambda row: (row["rmse"], row["mae"], row["alpha"]))
    model = Ridge(alpha=float(best["alpha"])).fit(train_scaled, train_targets)
    return OrdinaryBundle(scaler, model), trials, {
        "mae": float(best["mae"]),
        "rmse": float(best["rmse"]),
    }


def _predict_ordinary(bundle: OrdinaryBundle, table: pd.DataFrame) -> pd.DataFrame:
    predicted = bundle.model.predict(bundle.scaler.transform(_state_matrix(table)))
    output = table[["sample_id", "target_date", "y"]].rename(
        columns={"y": "cpi_actual"}
    )
    output["cpi_predicted"] = predicted
    output["error"] = output["cpi_predicted"] - output["cpi_actual"]
    output["absolute_error"] = output["error"].abs()
    return output.reset_index(drop=True)


def _save_readout(
    output_path: Path,
    scaler: StandardScaler,
    model: Ridge,
    metadata: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        coefficient=model.coef_,
        intercept=np.asarray([model.intercept_]),
        alpha=np.asarray([model.alpha]),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        metadata=np.asarray([json.dumps(metadata, ensure_ascii=False)]),
    )


def _save_figures(predictions: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "cpi_predicted_siamese": "Siamese frozen",
        "cpi_predicted_ordinary_full": "Ordinary 212",
        "cpi_predicted_ordinary_matched": "Ordinary matched 189",
    }
    fig, axis = plt.subplots(figsize=(12, 5.2))
    axis.plot(predictions["target_date"], predictions["cpi_actual"], label="Actual CPI")
    for column, label in labels.items():
        axis.plot(predictions["target_date"], predictions[column], label=label)
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI")
    axis.set_title("Frozen final test: optical reservoir CPI predictions")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "final_optical_test_predictions.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5.2))
    axis.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    for column, label in labels.items():
        residual = predictions[column] - predictions["cpi_actual"]
        axis.plot(predictions["target_date"], residual, label=label)
    axis.set_xlabel("Target month")
    axis.set_ylabel("Prediction residual")
    axis.set_title("Frozen final test: residuals")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "final_optical_test_residuals.png", dpi=180)
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _leakage_audit(
    test_pairs: pd.DataFrame,
    sample_index: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    target_i = pd.to_datetime(test_pairs["target_i_date"])
    target_j = pd.to_datetime(test_pairs["target_j_date"])
    start_i = pd.to_datetime(test_pairs["x_i_start_date"])
    end_i = pd.to_datetime(test_pairs["x_i_end_date"])
    end_j = pd.to_datetime(test_pairs["x_j_end_date"])
    gaps = (
        (start_i.dt.year - end_j.dt.year) * 12
        + start_i.dt.month
        - end_j.dt.month
    )
    split_lookup = sample_index.set_index("sample_id")["split"]
    delta_error = np.max(
        np.abs(
            test_pairs["delta_cpi"].to_numpy(dtype=float)
            - (
                test_pairs["cpi_i"].to_numpy(dtype=float)
                - test_pairs["cpi_j"].to_numpy(dtype=float)
            )
        )
    )
    sizes = test_pairs.groupby("sample_i_id").size()
    return {
        "evaluation_protocol": "monthly rolling one-step-ahead",
        "reference_strategy": config["reference_strategy"],
        "k": int(config["k"]),
        "selection_method_values": sorted(
            test_pairs["selection_method"].unique().tolist()
        ),
        "num_test_pairs": int(len(test_pairs)),
        "num_test_targets": int(test_pairs["sample_i_id"].nunique()),
        "references_per_target_min": int(sizes.min()),
        "references_per_target_max": int(sizes.max()),
        "target_j_strictly_earlier": bool((target_j < target_i).all()),
        "reference_known_by_target_input_end": bool((target_j <= end_i).all()),
        "minimum_non_overlap_gap_months": int(gaps.min()),
        "duplicate_pairs": int(
            test_pairs.duplicated(["sample_i_id", "sample_j_id"]).sum()
        ),
        "delta_formula_max_error": float(delta_error),
        "target_split_counts": {
            str(key): int(value)
            for key, value in test_pairs["sample_i_id"]
            .map(split_lookup)
            .value_counts()
            .items()
        },
        "reference_split_counts": {
            str(key): int(value)
            for key, value in test_pairs["sample_j_id"]
            .map(split_lookup)
            .value_counts()
            .items()
        },
        "reference_selection_uses_target_label": False,
        "configuration_changed_after_validation": False,
    }


def run_final_test(
    state_dir: Path = DEFAULT_STATE_DIR,
    data_dir: Path = DATA_PROCESSED_DIR,
    output_dir: Path = RESULTS_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Execute and save the one-time final evaluation."""
    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    summary_path = table_dir / FINAL_SUMMARY_NAME
    if summary_path.exists():
        raise FileExistsError(
            f"Final test is locked because {summary_path} already exists. "
            "Do not rerun or tune on the final test."
        )

    config = load_frozen_configuration(config_path)
    train_pairs = pd.read_csv(data_dir / "pair_indices_train.csv")
    candidates = build_test_candidates(data_dir)
    test_pairs = select_final_references(
        candidates,
        strategy=str(config["reference_strategy"]),
        k=int(config["k"]),
    )
    state_lookup = load_state_lookup(state_dir)
    siamese_bundle, siamese_pair_predictions, siamese_predictions = (
        _fit_frozen_siamese(train_pairs, test_pairs, state_lookup, config)
    )

    state_splits = load_state_splits(state_dir)
    train_full = state_splits["train"]
    val = state_splits["val"]
    test = state_splits["test"]
    matched_ids = set(train_pairs["sample_i_id"].astype(int))
    train_matched = train_full.loc[train_full["sample_id"].isin(matched_ids)].copy()
    if len(train_matched) != len(matched_ids):
        raise ValueError("Matched ordinary baseline is missing Siamese target IDs")

    ordinary_full, full_trials, full_val_metrics = _fit_ordinary(train_full, val)
    ordinary_matched, matched_trials, matched_val_metrics = _fit_ordinary(
        train_matched, val
    )
    full_predictions = _predict_ordinary(ordinary_full, test)
    matched_predictions = _predict_ordinary(ordinary_matched, test)

    siamese_test_metrics = regression_metrics(
        siamese_predictions["cpi_actual"], siamese_predictions["cpi_predicted"]
    )
    full_test_metrics = regression_metrics(
        full_predictions["cpi_actual"], full_predictions["cpi_predicted"]
    )
    matched_test_metrics = regression_metrics(
        matched_predictions["cpi_actual"], matched_predictions["cpi_predicted"]
    )
    comparison = pd.DataFrame(
        [
            {
                "model": "siamese_frozen",
                "num_train_targets": int(train_pairs["sample_i_id"].nunique()),
                "num_train_pairs": int(len(train_pairs)),
                "alpha": float(config["alpha"]),
                "val_mae": float(config["validation_mae"]),
                "val_rmse": float(config["validation_rmse"]),
                "test_mae": siamese_test_metrics["mae"],
                "test_rmse": siamese_test_metrics["rmse"],
            },
            {
                "model": "ordinary_full_212",
                "num_train_targets": int(len(train_full)),
                "num_train_pairs": 0,
                "alpha": float(ordinary_full.model.alpha),
                "val_mae": full_val_metrics["mae"],
                "val_rmse": full_val_metrics["rmse"],
                "test_mae": full_test_metrics["mae"],
                "test_rmse": full_test_metrics["rmse"],
            },
            {
                "model": "ordinary_matched_189",
                "num_train_targets": int(len(train_matched)),
                "num_train_pairs": 0,
                "alpha": float(ordinary_matched.model.alpha),
                "val_mae": matched_val_metrics["mae"],
                "val_rmse": matched_val_metrics["rmse"],
                "test_mae": matched_test_metrics["mae"],
                "test_rmse": matched_test_metrics["rmse"],
            },
        ]
    )

    unified = siamese_predictions[
        ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(
        columns={
            "sample_i_id": "sample_id",
            "cpi_predicted": "cpi_predicted_siamese",
        }
    )
    unified = unified.merge(
        full_predictions[["sample_id", "target_date", "cpi_predicted"]].rename(
            columns={"cpi_predicted": "cpi_predicted_ordinary_full"}
        ),
        on=["sample_id", "target_date"],
        validate="one_to_one",
    )
    unified = unified.merge(
        matched_predictions[["sample_id", "target_date", "cpi_predicted"]].rename(
            columns={"cpi_predicted": "cpi_predicted_ordinary_matched"}
        ),
        on=["sample_id", "target_date"],
        validate="one_to_one",
    )
    for name in ("siamese", "ordinary_full", "ordinary_matched"):
        prediction_column = f"cpi_predicted_{name}"
        unified[f"residual_{name}"] = unified[prediction_column] - unified["cpi_actual"]
        unified[f"absolute_error_{name}"] = unified[f"residual_{name}"].abs()

    residual_rows: list[dict[str, object]] = []
    for name in ("siamese", "ordinary_full", "ordinary_matched"):
        residual = unified[f"residual_{name}"].to_numpy(dtype=float)
        residual_rows.append(
            {
                "model": name,
                "mean_residual": float(np.mean(residual)),
                "residual_std": float(np.std(residual)),
                "median_absolute_error": float(np.median(np.abs(residual))),
                "max_absolute_error": float(np.max(np.abs(residual))),
            }
        )
    residual_summary = pd.DataFrame(residual_rows)

    parameter_table = pd.DataFrame(
        [
            {
                "model": "siamese_frozen",
                "state_width": 50,
                "reference_strategy": config["reference_strategy"],
                "k": config["k"],
                "feature_mode": config["feature_mode"],
                "aggregation": config["aggregation"],
                "alpha": config["alpha"],
                "selection_source": "validation only",
            },
            {
                "model": "ordinary_full_212",
                "state_width": 50,
                "reference_strategy": "not_applicable",
                "k": 0,
                "feature_mode": "single_state",
                "aggregation": "not_applicable",
                "alpha": ordinary_full.model.alpha,
                "selection_source": "validation only",
            },
            {
                "model": "ordinary_matched_189",
                "state_width": 50,
                "reference_strategy": "not_applicable",
                "k": 0,
                "feature_mode": "single_state",
                "aggregation": "not_applicable",
                "alpha": ordinary_matched.model.alpha,
                "selection_source": "validation only",
            },
        ]
    )

    sample_index = pd.read_csv(data_dir / "sample_index.csv")
    audit = _leakage_audit(test_pairs, sample_index, config)
    audit["frozen_configuration_file"] = str(config_path)
    audit["frozen_configuration_sha256"] = _sha256(config_path)
    audit["test_state_loaded"] = True
    audit["test_evaluated_once_by_this_runner"] = True

    table_dir.mkdir(parents=True, exist_ok=True)
    siamese_pair_predictions.to_csv(
        table_dir / "final_siamese_test_pair_predictions.csv", index=False
    )
    test_pairs.to_csv(table_dir / "final_siamese_test_selected_pairs.csv", index=False)
    unified.to_csv(table_dir / "final_test_predictions.csv", index=False)
    comparison.to_csv(table_dir / "final_model_comparison.csv", index=False)
    residual_summary.to_csv(table_dir / "final_residual_summary.csv", index=False)
    parameter_table.to_csv(table_dir / "final_model_parameters.csv", index=False)
    pd.concat(
        [
            pd.DataFrame(full_trials).assign(model="ordinary_full_212"),
            pd.DataFrame(matched_trials).assign(model="ordinary_matched_189"),
        ],
        ignore_index=True,
    ).to_csv(table_dir / "final_ordinary_alpha_selection.csv", index=False)
    with (table_dir / "final_test_leakage_audit.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
        file.write("\n")

    _save_readout(
        model_dir / "final_siamese_readout.npz",
        siamese_bundle.scaler,
        siamese_bundle.model,
        {"model": "siamese_frozen", **config},
    )
    _save_readout(
        model_dir / "final_ordinary_full_readout.npz",
        ordinary_full.scaler,
        ordinary_full.model,
        {"model": "ordinary_full_212"},
    )
    _save_readout(
        model_dir / "final_ordinary_matched_readout.npz",
        ordinary_matched.scaler,
        ordinary_matched.model,
        {"model": "ordinary_matched_189"},
    )
    _save_figures(unified, figure_dir)

    summary = {
        "status": "final_test_completed_locked",
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "frozen_configuration_sha256": _sha256(config_path),
        "frozen_siamese_configuration": config,
        "test_targets": int(len(unified)),
        "test_run_policy": "one-time final evaluation; do not tune or rerun",
        "comparison_file": str(table_dir / "final_model_comparison.csv"),
        "leakage_audit_file": str(table_dir / "final_test_leakage_audit.json"),
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_final_test(
        state_dir=args.state_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    print(comparison.to_string(index=False))
    print("\nFinal test is now locked by final_test_run_summary.json")


if __name__ == "__main__":
    main()
