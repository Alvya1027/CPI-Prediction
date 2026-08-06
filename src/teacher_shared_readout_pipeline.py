"""Teacher-specified shared-readout experiment on fixed optical states.

The optical reservoir, mask and physical parameters stay frozen.  Target and
reference branches use the same linear output function ``b + w.T @ z``.  The
validation stage deliberately opens only the physically isolated train and
validation files; the test stage requires a previously written frozen config.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.config import RESULTS_DIR, ROOT_DIR
from src.create_siamese_pairs import _build_candidate_pairs
from src.siamese_reservoir_regression import regression_metrics
from src.siamese_split_isolation import (
    build_isolated_closed_train_pool,
    load_isolated_split,
)
from src.teacher_shared_readout import (
    SharedReadoutModel,
    aggregate_pair_predictions,
    fit_absolute_ridge,
    fit_joint_shared_readout,
)


PROFILE_ROOT = ROOT_DIR / "matlab" / "optical_reservoir_cpi_mom_recent50_20260730"
DATA_DIR = PROFILE_ROOT / "data"
SERIAL_STATE_DIR = PROFILE_ROOT / "states"
OUTPUT_DIR = RESULTS_DIR / "siamese_optical_mom_teacher_shared_readout_20260802"

EXPECTED_SPLITS = {
    "train": (50, "2014-07", "2018-08"),
    "val": (45, "2018-09", "2022-05"),
    "test": (47, "2022-06", "2026-04"),
}
# The core objective uses mean squared error.  Values through 2.0 are the
# exact normalized equivalents of the legacy sklearn Ridge grid through 100
# for n_train=50; 10 and 100 retain stronger-shrinkage teacher candidates.
DEFAULT_ALPHAS = (0.0, 2e-8, 2e-6, 2e-4, 0.002, 0.02, 0.2, 2.0, 10.0, 100.0)
DEFAULT_PAIR_WEIGHTS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_K_VALUES = (1, 3, 5, 10)
DEFAULT_AGGREGATIONS = ("mean", "inverse_distance")
MIN_GAP_MONTHS = 1
STATE_PROTOCOL = "continuous_serial_shared_fixed_reservoir"

StateLoader = Callable[
    [Path, dict[str, dict[str, object]]],
    tuple[dict[int, np.ndarray], dict[str, object]],
]


@dataclass(frozen=True)
class FrozenTeacherConfig:
    """Configuration selected without opening test data or test states."""

    alpha: float
    pair_weight: float
    k_references: int
    aggregation: str
    min_gap_months: int
    absolute_alpha: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_matrix(
    sample_ids: Sequence[int] | np.ndarray,
    state_lookup: dict[int, np.ndarray],
) -> np.ndarray:
    ids = np.asarray(sample_ids, dtype=int).reshape(-1)
    missing = set(ids.tolist()).difference(state_lookup)
    if missing:
        raise ValueError(f"missing optical states for IDs: {sorted(missing)[:10]}")
    return np.vstack([state_lookup[int(sample_id)] for sample_id in ids]).astype(
        np.float64
    )


def attach_cache_provenance(
    pairs: pd.DataFrame,
    state_audit: dict[str, object],
) -> pd.DataFrame:
    """Attach both cache records without inventing a pair simulation run.

    Legacy state loaders do not expose row-level cache records and therefore
    return the pair table unchanged.  The explicit Twin loader supplies the
    records, allowing every semantic relation to point back to the two actual
    MATLAB generation runs that produced ``h_i`` and ``h_j``.
    """
    records = state_audit.get("cache_records")
    if records is None:
        return pairs
    if not records:
        raise ValueError("explicit state audit contains an empty cache_records table")
    if state_audit.get("cache_reuse_declared") is not True:
        raise ValueError("cache provenance requires cache_reuse_declared=true")
    if state_audit.get("semantic_pairs_simulated_simultaneously") is not False:
        raise ValueError(
            "cached semantic pairs must not claim simultaneous MATLAB simulation"
        )
    provenance = pd.DataFrame(records)
    required = (
        "split",
        "sample_id",
        "cache_record_id",
        "generation_run_id",
        "branch_id",
        "partner_sample_id",
        "state_file_sha256",
    )
    missing = set(required).difference(provenance.columns)
    if missing:
        raise ValueError(f"state cache provenance lacks: {sorted(missing)}")
    if provenance["sample_id"].duplicated().any():
        raise ValueError("state cache provenance contains duplicate sample IDs")
    forbidden_run_columns = {
        "simultaneous_run_id",
        "pair_simulation_run_id",
    }
    if forbidden_run_columns.intersection(pairs.columns) or forbidden_run_columns.intersection(
        provenance.columns
    ):
        raise ValueError(
            "cached semantic pairs must not carry a fabricated simultaneous run ID"
        )
    reserved_columns = {
        "semantic_pair_id",
        "state_resolution",
        "derived_from_cache",
        "semantic_pair_simulated_simultaneously",
    }
    for suffix in ("i", "j"):
        reserved_columns.update(
            {
                f"{suffix}_cache_split",
                f"{suffix}_cache_record_id",
                f"{suffix}_generation_run_id",
                f"{suffix}_generation_branch",
                f"{suffix}_generation_partner_sample_id",
                f"{suffix}_state_file_sha256",
            }
        )
    collision = reserved_columns.intersection(pairs.columns)
    if collision:
        raise ValueError(
            "pair table already contains reserved cache provenance columns: "
            f"{sorted(collision)}"
        )
    result = pairs.copy()
    for suffix in ("i", "j"):
        renamed = provenance[list(required)].rename(
            columns={
                "split": f"{suffix}_cache_split",
                "sample_id": f"sample_{suffix}_id",
                "cache_record_id": f"{suffix}_cache_record_id",
                "generation_run_id": f"{suffix}_generation_run_id",
                "branch_id": f"{suffix}_generation_branch",
                "partner_sample_id": f"{suffix}_generation_partner_sample_id",
                "state_file_sha256": f"{suffix}_state_file_sha256",
            }
        )
        result = result.merge(
            renamed,
            on=f"sample_{suffix}_id",
            how="left",
            validate="many_to_one",
        )
        if result[f"{suffix}_cache_record_id"].isna().any():
            raise ValueError(f"pair table lacks {suffix}-side cache provenance")
    result.insert(
        0,
        "semantic_pair_id",
        [
            f"{i_split}:{int(i_id)}__{j_split}:{int(j_id)}"
            for i_split, i_id, j_split, j_id in zip(
                result["i_cache_split"],
                result["sample_i_id"],
                result["j_cache_split"],
                result["sample_j_id"],
            )
        ],
    )
    if result["semantic_pair_id"].duplicated().any():
        raise ValueError("semantic pair table contains duplicate target/reference IDs")
    result["state_resolution"] = "unique_window_cache_lookup"
    result["derived_from_cache"] = True
    result["semantic_pair_simulated_simultaneously"] = False
    return result


def load_state_splits(
    state_dir: Path,
    split_payloads: dict[str, dict[str, object]],
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    """Load exactly the requested state splits and audit them against data."""
    lookup: dict[int, np.ndarray] = {}
    masks: list[np.ndarray] = []
    width: int | None = None
    rows: dict[str, object] = {}
    for split, data in split_payloads.items():
        if split not in EXPECTED_SPLITS:
            raise ValueError(f"unexpected split: {split}")
        expected_count, first_date, last_date = EXPECTED_SPLITS[split]
        state_path = state_dir / f"states_{split}.mat"
        if not state_path.exists():
            raise FileNotFoundError(f"missing state cache: {state_path}")
        payload = loadmat(state_path)
        required = {"state_matrix", "sample_id", "target", "mask"}
        missing_fields = required.difference(payload)
        if missing_fields:
            raise ValueError(f"{state_path} lacks fields: {sorted(missing_fields)}")
        states = np.asarray(payload["state_matrix"], dtype=np.float64)
        sample_ids = np.asarray(payload["sample_id"]).reshape(-1).astype(int)
        targets = np.asarray(payload["target"], dtype=np.float64).reshape(-1)
        index = data["index"].reset_index(drop=True)
        expected_ids = index["sample_id"].to_numpy(dtype=int)
        expected_targets = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        if len(index) != expected_count:
            raise ValueError(f"{split} has {len(index)} rows, expected {expected_count}")
        if (
            str(index["target_date"].iloc[0]) != first_date
            or str(index["target_date"].iloc[-1]) != last_date
        ):
            raise ValueError(f"{split} target dates do not match the frozen boundary")
        if states.ndim != 2 or states.shape[0] != expected_count:
            raise ValueError(f"invalid {split} state shape: {states.shape}")
        if width is None:
            width = int(states.shape[1])
        elif states.shape[1] != width:
            raise ValueError("state splits use different virtual-node widths")
        if not np.array_equal(sample_ids, expected_ids):
            raise ValueError(f"{split} state IDs disagree with isolated data")
        if not np.allclose(targets, expected_targets):
            raise ValueError(f"{split} state targets disagree with isolated data")
        if not np.isfinite(states).all() or float(np.std(states)) == 0.0:
            raise ValueError(f"{split} states are non-finite or constant")
        for sample_id, state in zip(sample_ids, states):
            if int(sample_id) in lookup:
                raise ValueError(f"duplicate state ID: {sample_id}")
            lookup[int(sample_id)] = state
        masks.append(np.asarray(payload["mask"]))
        rows[split] = {
            "state_file": str(state_path),
            "sha256": _sha256(state_path),
            "shape": [int(value) for value in states.shape],
            "first_target_date": first_date,
            "last_target_date": last_date,
        }
    if not all(np.array_equal(masks[0], mask) for mask in masks[1:]):
        raise ValueError("requested state splits do not share one fixed mask")
    return lookup, {
        "status": "passed",
        "state_protocol": STATE_PROTOCOL,
        "state_width": int(width or 0),
        "shared_fixed_mask": True,
        "loaded_splits": list(split_payloads),
        "splits": rows,
    }


def build_train_pairs(
    train_pool: dict[str, object],
    min_gap_months: int = MIN_GAP_MONTHS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build legal relations from the same closed train50 target pool."""
    pairs = _build_candidate_pairs(
        train_pool, train_pool, min_gap_months=min_gap_months
    ).reset_index(drop=True)
    if pairs.empty:
        raise ValueError("closed train pair table is empty")
    train_ids = train_pool["index"]["sample_id"].to_numpy(dtype=int)
    position = {int(sample_id): row for row, sample_id in enumerate(train_ids)}
    try:
        pair_i = np.asarray(
            [position[int(value)] for value in pairs["sample_i_id"]], dtype=int
        )
        pair_j = np.asarray(
            [position[int(value)] for value in pairs["sample_j_id"]], dtype=int
        )
    except KeyError as exc:
        raise ValueError("a training pair escaped the closed train50 pool") from exc
    target_dates = pd.to_datetime(pairs["target_i_date"])
    reference_dates = pd.to_datetime(pairs["target_j_date"])
    if not (reference_dates < target_dates).all():
        raise ValueError("training pairs are not chronological")
    return pairs, pair_i, pair_j


def build_evaluation_candidates(
    target_split: dict[str, object],
    train_pool: dict[str, object],
    min_gap_months: int = MIN_GAP_MONTHS,
) -> pd.DataFrame:
    candidates = _build_candidate_pairs(
        target_split, train_pool, min_gap_months=min_gap_months
    ).reset_index(drop=True)
    if candidates.empty:
        raise ValueError("evaluation candidate table is empty")
    train_ids = set(train_pool["index"]["sample_id"].astype(int))
    if not set(candidates["sample_j_id"].astype(int)).issubset(train_ids):
        raise ValueError("an evaluation reference escaped train50")
    return candidates


def select_references(candidates: pd.DataFrame, k: int) -> pd.DataFrame:
    """Select by input-window distance only; CPI labels never rank rows."""
    if int(k) <= 0:
        raise ValueError("k must be positive")
    required = {"sample_i_id", "sample_j_id", "window_distance", "target_j_date"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"candidate fields missing: {sorted(missing)}")
    ordered = candidates.sort_values(
        ["sample_i_id", "window_distance", "target_j_date", "sample_j_id"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    selected = (
        ordered.groupby("sample_i_id", sort=False, group_keys=False)
        .head(int(k))
        .copy()
        .reset_index(drop=True)
    )
    sizes = selected.groupby("sample_i_id").size()
    if len(sizes) != candidates["sample_i_id"].nunique() or not (sizes == int(k)).all():
        raise ValueError(f"at least one target has fewer than K={int(k)} references")
    selected["selection_method"] = f"input_window_distance_k{int(k)}"
    return selected


def predict_direct_split(
    model: SharedReadoutModel,
    target_split: dict[str, object],
    state_lookup: dict[int, np.ndarray],
) -> pd.DataFrame:
    index = target_split["index"].reset_index(drop=True)
    states = _state_matrix(index["sample_id"].to_numpy(dtype=int), state_lookup)
    actual = np.asarray(target_split["y"], dtype=np.float64).reshape(-1)
    predicted = model.predict_direct(states)
    result = pd.DataFrame(
        {
            "sample_i_id": index["sample_id"].to_numpy(dtype=int),
            "target_date": index["target_date"].astype(str),
            "cpi_actual": actual,
            "cpi_predicted": predicted,
        }
    )
    result["error"] = result["cpi_predicted"] - result["cpi_actual"]
    result["absolute_error"] = result["error"].abs()
    return result


def predict_reference_split(
    model: SharedReadoutModel,
    selected_pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
    aggregation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states_i = _state_matrix(selected_pairs["sample_i_id"], state_lookup)
    states_j = _state_matrix(selected_pairs["sample_j_id"], state_lookup)
    pair_output = selected_pairs.copy()
    pair_output["delta_cpi_predicted"] = model.predict_delta(states_i, states_j)
    pair_output["cpi_pred_pair"] = model.predict_from_references(
        states_i,
        states_j,
        pair_output["cpi_j"].to_numpy(dtype=float),
    )
    pair_output["delta_error"] = (
        pair_output["delta_cpi_predicted"]
        - pair_output["delta_cpi"].to_numpy(dtype=float)
    )
    grouped = aggregate_pair_predictions(
        pair_output["sample_i_id"].to_numpy(dtype=int),
        pair_output["cpi_pred_pair"].to_numpy(dtype=float),
        method=aggregation,
        distances=pair_output["window_distance"].to_numpy(dtype=float),
    )
    pair_output["aggregation_weight"] = grouped.pair_weights
    metadata = (
        pair_output.groupby("sample_i_id", sort=False)
        .agg(
            target_date=("target_i_date", "first"),
            cpi_actual=("cpi_i", "first"),
            reference_prediction_std=("cpi_pred_pair", "std"),
        )
        .reset_index()
    )
    predictions = pd.DataFrame(
        {
            "sample_i_id": grouped.target_ids.astype(int),
            "cpi_predicted": grouped.predictions,
            "num_references": grouped.num_references,
        }
    ).merge(metadata, on="sample_i_id", validate="one_to_one")
    predictions["reference_prediction_std"] = predictions[
        "reference_prediction_std"
    ].fillna(0.0)
    predictions = predictions.sort_values("target_date").reset_index(drop=True)
    predictions["error"] = predictions["cpi_predicted"] - predictions["cpi_actual"]
    predictions["absolute_error"] = predictions["error"].abs()
    return pair_output, predictions


def _metrics(predictions: pd.DataFrame) -> dict[str, float]:
    return regression_metrics(
        predictions["cpi_actual"].to_numpy(dtype=float),
        predictions["cpi_predicted"].to_numpy(dtype=float),
    )


def _save_validation_figures(
    baseline: pd.DataFrame,
    teacher: pd.DataFrame,
    metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    axis.plot(baseline["target_date"], baseline["cpi_actual"], color="black", linewidth=2, label="Actual")
    axis.plot(baseline["target_date"], baseline["cpi_predicted"], label="Absolute-only shared reservoir")
    axis.plot(teacher["target_date"], teacher["cpi_predicted"], label="Teacher shared-readout pair")
    axis.set_xlabel("Validation target month")
    axis.set_ylabel("CPI MoM index (previous month = 100)")
    axis.set_title("Validation predictions (test remains unopened)")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_prediction_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(metrics))
    width = 0.36
    axis.bar(x - width / 2, metrics["val_mae"], width, label="MAE")
    axis.bar(x + width / 2, metrics["val_rmse"], width, label="RMSE")
    axis.set_xticks(x, metrics["model"], rotation=15, ha="right")
    axis.set_ylabel("Validation error")
    axis.set_title("Validation-only model comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_metric_comparison.png", dpi=180)
    plt.close(fig)


def run_validation_search(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_DIR,
    state_dir: Path = SERIAL_STATE_DIR,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    pair_weights: Iterable[float] = DEFAULT_PAIR_WEIGHTS,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    aggregations: Iterable[str] = DEFAULT_AGGREGATIONS,
    state_loader: StateLoader = load_state_splits,
    state_protocol: str = STATE_PROTOCOL,
    experiment_name: str = (
        "teacher shared linear readout on a fixed optical reservoir"
    ),
) -> FrozenTeacherConfig:
    """Select all readout/reference settings using train and validation only."""
    train_pool = build_isolated_closed_train_pool(data_dir, count=50)
    validation = load_isolated_split(data_dir, "val")
    state_lookup, state_audit = state_loader(
        state_dir, {"train": train_pool, "val": validation}
    )
    if state_audit.get("state_protocol") != state_protocol:
        raise ValueError(
            "state loader returned an unexpected protocol: "
            f"{state_audit.get('state_protocol')!r}"
        )
    train_ids = train_pool["index"]["sample_id"].to_numpy(dtype=int)
    train_states = _state_matrix(train_ids, state_lookup)
    train_targets = np.asarray(train_pool["y"], dtype=np.float64).reshape(-1)
    train_pairs, pair_i, pair_j = build_train_pairs(train_pool)
    train_pairs = attach_cache_provenance(train_pairs, state_audit)
    candidates = build_evaluation_candidates(validation, train_pool)

    alpha_values = tuple(float(value) for value in alphas)
    pair_weight_values = tuple(float(value) for value in pair_weights)
    k_options = tuple(int(value) for value in k_values)
    aggregation_options = tuple(str(value) for value in aggregations)
    if not alpha_values or not pair_weight_values or not k_options or not aggregation_options:
        raise ValueError("validation grids must be non-empty")

    absolute_rows: list[dict[str, object]] = []
    absolute_cache: dict[float, tuple[SharedReadoutModel, pd.DataFrame]] = {}
    for alpha in alpha_values:
        model = fit_absolute_ridge(train_states, train_targets, alpha=alpha)
        predictions = predict_direct_split(model, validation, state_lookup)
        metrics = _metrics(predictions)
        absolute_rows.append({"alpha": alpha, **metrics})
        absolute_cache[alpha] = (model, predictions)
    absolute_best = min(
        absolute_rows, key=lambda row: (row["rmse"], row["mae"], row["alpha"])
    )
    absolute_model, absolute_predictions = absolute_cache[
        float(absolute_best["alpha"])
    ]

    selected_cache = {
        k: attach_cache_provenance(select_references(candidates, k), state_audit)
        for k in k_options
    }
    rows: list[dict[str, object]] = []
    prediction_cache: dict[tuple[float, float, int, str], tuple[pd.DataFrame, pd.DataFrame, SharedReadoutModel]] = {}
    for alpha, pair_weight in itertools.product(alpha_values, pair_weight_values):
        model = fit_joint_shared_readout(
            train_states,
            train_targets,
            pair_i,
            pair_j,
            alpha=alpha,
            pair_weight=pair_weight,
            absolute_weight=1.0,
        )
        direct_predictions = predict_direct_split(model, validation, state_lookup)
        direct_metrics = _metrics(direct_predictions)
        for k, aggregation in itertools.product(k_options, aggregation_options):
            pair_output, predictions = predict_reference_split(
                model, selected_cache[k], state_lookup, aggregation
            )
            metrics = _metrics(predictions)
            key = (alpha, pair_weight, k, aggregation)
            rows.append(
                {
                    "alpha": alpha,
                    "pair_weight": pair_weight,
                    "k_references": k,
                    "aggregation": aggregation,
                    "val_mae": metrics["mae"],
                    "val_rmse": metrics["rmse"],
                    "joint_direct_val_mae": direct_metrics["mae"],
                    "joint_direct_val_rmse": direct_metrics["rmse"],
                    "num_train_targets": len(train_pool["index"]),
                    "num_train_pairs": len(train_pairs),
                    "num_train_pair_targets": train_pairs["sample_i_id"].nunique(),
                    "num_validation_targets": len(predictions),
                    "num_validation_pairs": len(pair_output),
                }
            )
            prediction_cache[key] = (pair_output, predictions, model)
    search = pd.DataFrame(rows).sort_values(
        ["val_rmse", "val_mae", "pair_weight", "k_references", "aggregation", "alpha"],
        kind="stable",
    ).reset_index(drop=True)
    positive_pair_search = search.loc[search["pair_weight"] > 0].reset_index(drop=True)
    if positive_pair_search.empty:
        raise ValueError(
            "the teacher joint model requires at least one positive pair weight"
        )
    # lambda=0 is retained as a diagnostic reference-calibration ablation.  The
    # teacher model itself must actually train on target/reference relations.
    best_row = positive_pair_search.iloc[0]
    zero_pair_search = search.loc[search["pair_weight"] == 0].reset_index(drop=True)
    zero_pair_best = zero_pair_search.iloc[0] if not zero_pair_search.empty else None
    config = FrozenTeacherConfig(
        alpha=float(best_row["alpha"]),
        pair_weight=float(best_row["pair_weight"]),
        k_references=int(best_row["k_references"]),
        aggregation=str(best_row["aggregation"]),
        min_gap_months=MIN_GAP_MONTHS,
        absolute_alpha=float(absolute_best["alpha"]),
    )
    best_key = (
        config.alpha,
        config.pair_weight,
        config.k_references,
        config.aggregation,
    )
    best_pair_output, best_predictions, best_model = prediction_cache[best_key]

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    search.to_csv(table_dir / "validation_configuration_search.csv", index=False)
    pd.DataFrame(absolute_rows).to_csv(
        table_dir / "absolute_alpha_validation_search.csv", index=False
    )
    train_pairs.to_csv(table_dir / "train_pair_relations.csv", index=False)
    selected_cache[config.k_references].to_csv(
        table_dir / "selected_validation_references.csv", index=False
    )
    best_pair_output.to_csv(
        table_dir / "selected_validation_pair_predictions.csv", index=False
    )
    best_predictions.to_csv(
        table_dir / "selected_validation_predictions.csv", index=False
    )
    absolute_predictions.to_csv(
        table_dir / "absolute_validation_predictions.csv", index=False
    )
    np.savez_compressed(
        model_dir / "teacher_shared_readout_validation_frozen.npz",
        **best_model.to_npz_dict(),
    )
    np.savez_compressed(
        model_dir / "absolute_readout_validation_frozen.npz",
        **absolute_model.to_npz_dict(),
    )
    comparison_rows: list[dict[str, object]] = [
            {
                "model": "absolute_only_same_states",
                "val_mae": float(absolute_best["mae"]),
                "val_rmse": float(absolute_best["rmse"]),
            },
            {
                "model": "teacher_shared_readout_pair",
                "val_mae": float(best_row["val_mae"]),
                "val_rmse": float(best_row["val_rmse"]),
            },
        ]
    if zero_pair_best is not None:
        comparison_rows.insert(
            1,
            {
                "model": "reference_calibration_without_pair_loss_ablation",
                "val_mae": float(zero_pair_best["val_mae"]),
                "val_rmse": float(zero_pair_best["val_rmse"]),
            },
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(table_dir / "validation_model_comparison.csv", index=False)
    _save_validation_figures(
        absolute_predictions, best_predictions, comparison, figure_dir
    )

    frozen = {
        "reporting_status": (
            "teacher_final_explicit_twin_validation_frozen"
            if state_protocol
            == "explicit_twin_audited_unique_window_cache_v1"
            else "legacy_continuous_state_not_teacher_final_explicit_twin"
        ),
        "status": "validation_frozen_not_tested",
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": experiment_name,
        "state_protocol": state_protocol,
        "configuration": asdict(config),
        "selection_metric": "validation target-level RMSE, then MAE",
        "ridge_objective": "mean_squared_error + alpha * squared_l2_output_weight",
        "alpha_grid_scale": (
            "includes exact n_train=50 normalized equivalents of the legacy "
            "sklearn Ridge alpha grid"
        ),
        "protocol_revision": (
            "An initial local test artifact was invalidated before reporting because "
            "the normalized alpha grid omitted alpha=2, equivalent to the legacy "
            "single-network alpha=100. Validation freeze and test were rerun from "
            "scratch after correcting this implementation-level fairness issue."
        ),
        "accessible_original_train_months": 50,
        "derived_train_pair_relations": int(len(train_pairs)),
        "derived_train_pair_target_months": int(train_pairs["sample_i_id"].nunique()),
        "validation_targets": 45,
        "test_data_loaded": False,
        "test_state_loaded": False,
        "test_evaluated": False,
        "reservoir_parameters_trained": False,
        "only_output_weights_trained": True,
        "train_validation_state_audit": state_audit,
        "data_manifest_sha256": _sha256(data_dir / "isolated_split_manifest.json"),
        "validation_metrics": {
            "absolute_only": {
                "mae": float(absolute_best["mae"]),
                "rmse": float(absolute_best["rmse"]),
            },
            "teacher_pair": {
                "mae": float(best_row["val_mae"]),
                "rmse": float(best_row["val_rmse"]),
            },
            "reference_calibration_without_pair_loss_ablation": (
                None
                if zero_pair_best is None
                else {
                    "mae": float(zero_pair_best["val_mae"]),
                    "rmse": float(zero_pair_best["val_rmse"]),
                }
            ),
        },
    }
    config_path = table_dir / "selected_configuration.json"
    config_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config


def _save_test_figures(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    axis.plot(predictions["target_date"], predictions["cpi_actual"], color="black", linewidth=2, label="Actual")
    axis.plot(predictions["target_date"], predictions["cpi_predicted_absolute"], label="Absolute-only")
    axis.plot(predictions["target_date"], predictions["cpi_predicted_teacher_pair"], label="Teacher shared-readout pair")
    axis.set_xlabel("Test target month")
    axis.set_ylabel("CPI MoM index (previous month = 100)")
    axis.set_title("Frozen-config test predictions")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_prediction_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(metrics))
    width = 0.36
    axis.bar(x - width / 2, metrics["test_mae"], width, label="MAE")
    axis.bar(x + width / 2, metrics["test_rmse"], width, label="RMSE")
    axis.set_xticks(x, metrics["model"], rotation=15, ha="right")
    axis.set_ylabel("Test error")
    axis.set_title("Same-state fair test comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_metric_comparison.png", dpi=180)
    plt.close(fig)


def run_frozen_test(
    frozen_config_path: Path,
    data_dir: Path = DATA_DIR,
    state_dir: Path = SERIAL_STATE_DIR,
    state_loader: StateLoader = load_state_splits,
    expected_state_protocol: str = STATE_PROTOCOL,
) -> pd.DataFrame:
    """Evaluate test once from a validation-frozen JSON configuration."""
    frozen_config_path = frozen_config_path.resolve()
    frozen = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "validation_frozen_not_tested":
        raise ValueError("the supplied configuration is not validation-frozen")
    if frozen.get("test_data_loaded") or frozen.get("test_state_loaded"):
        raise ValueError("the supplied configuration already records test access")
    if frozen.get("state_protocol") != expected_state_protocol:
        raise ValueError(
            "the frozen configuration uses a different state protocol"
        )
    output_dir = frozen_config_path.parent.parent
    completion_path = output_dir / "test_evaluation_manifest.json"
    if completion_path.exists():
        raise FileExistsError(
            f"test was already evaluated for this output directory: {completion_path}"
        )
    config = FrozenTeacherConfig(**frozen["configuration"])
    if config.min_gap_months != MIN_GAP_MONTHS:
        raise ValueError("frozen gap does not match this implementation")
    if frozen.get("data_manifest_sha256") != _sha256(
        data_dir / "isolated_split_manifest.json"
    ):
        raise ValueError("the isolated data manifest changed after validation freeze")

    train_pool = build_isolated_closed_train_pool(data_dir, count=50)
    test = load_isolated_split(data_dir, "test")
    state_lookup, state_audit = state_loader(
        state_dir, {"train": train_pool, "test": test}
    )
    if state_audit.get("state_protocol") != expected_state_protocol:
        raise ValueError("test states use a different state protocol")
    expected_train_hash = frozen["train_validation_state_audit"]["splits"]["train"][
        "sha256"
    ]
    if state_audit["splits"]["train"]["sha256"] != expected_train_hash:
        raise ValueError("training states changed after validation selection")
    expected_protocol_identity = frozen["train_validation_state_audit"].get(
        "protocol_identity_sha256"
    )
    if expected_protocol_identity is not None and state_audit.get(
        "protocol_identity_sha256"
    ) != expected_protocol_identity:
        raise ValueError(
            "the Twin model, mask, simulation protocol, or equivalence audit "
            "changed after validation selection"
        )

    train_ids = train_pool["index"]["sample_id"].to_numpy(dtype=int)
    train_states = _state_matrix(train_ids, state_lookup)
    train_targets = np.asarray(train_pool["y"], dtype=np.float64).reshape(-1)
    train_pairs, pair_i, pair_j = build_train_pairs(
        train_pool, min_gap_months=config.min_gap_months
    )
    train_pairs = attach_cache_provenance(train_pairs, state_audit)
    absolute_model = fit_absolute_ridge(
        train_states, train_targets, alpha=config.absolute_alpha
    )
    teacher_model = fit_joint_shared_readout(
        train_states,
        train_targets,
        pair_i,
        pair_j,
        alpha=config.alpha,
        pair_weight=config.pair_weight,
        absolute_weight=1.0,
    )
    absolute_predictions = predict_direct_split(absolute_model, test, state_lookup)
    joint_direct_predictions = predict_direct_split(teacher_model, test, state_lookup)
    candidates = build_evaluation_candidates(
        test, train_pool, min_gap_months=config.min_gap_months
    )
    selected = attach_cache_provenance(
        select_references(candidates, config.k_references), state_audit
    )
    pair_predictions, teacher_predictions = predict_reference_split(
        teacher_model, selected, state_lookup, config.aggregation
    )
    if len(absolute_predictions) != 47 or len(teacher_predictions) != 47:
        raise ValueError("test prediction tables must each contain 47 targets")
    absolute_metrics = _metrics(absolute_predictions)
    direct_metrics = _metrics(joint_direct_predictions)
    teacher_metrics = _metrics(teacher_predictions)
    comparison = pd.DataFrame(
        [
            {
                "model": "absolute_only_same_states",
                "test_mae": absolute_metrics["mae"],
                "test_rmse": absolute_metrics["rmse"],
            },
            {
                "model": "joint_shared_readout_direct_diagnostic",
                "test_mae": direct_metrics["mae"],
                "test_rmse": direct_metrics["rmse"],
            },
            {
                "model": "teacher_shared_readout_pair",
                "test_mae": teacher_metrics["mae"],
                "test_rmse": teacher_metrics["rmse"],
            },
        ]
    )
    unified = absolute_predictions[
        ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(columns={"cpi_predicted": "cpi_predicted_absolute"})
    unified = unified.merge(
        joint_direct_predictions[
            ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
        ].rename(columns={"cpi_predicted": "cpi_predicted_joint_direct"}),
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    ).merge(
        teacher_predictions[
            ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
        ].rename(columns={"cpi_predicted": "cpi_predicted_teacher_pair"}),
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    )
    for suffix in ("absolute", "joint_direct", "teacher_pair"):
        unified[f"residual_{suffix}"] = (
            unified[f"cpi_predicted_{suffix}"] - unified["cpi_actual"]
        )
        unified[f"absolute_error_{suffix}"] = unified[f"residual_{suffix}"].abs()

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(table_dir / "test_model_comparison.csv", index=False)
    unified.to_csv(table_dir / "test_prediction_comparison.csv", index=False)
    selected.to_csv(table_dir / "selected_test_references.csv", index=False)
    pair_predictions.to_csv(table_dir / "selected_test_pair_predictions.csv", index=False)
    np.savez_compressed(
        model_dir / "teacher_shared_readout_final.npz", **teacher_model.to_npz_dict()
    )
    np.savez_compressed(
        model_dir / "absolute_readout_final.npz", **absolute_model.to_npz_dict()
    )
    _save_test_figures(unified, comparison, figure_dir)

    absolute_rmse = absolute_metrics["rmse"]
    manifest = {
        **frozen,
        "reporting_status": (
            "teacher_final_explicit_twin_test_evaluated_once"
            if expected_state_protocol
            == "explicit_twin_audited_unique_window_cache_v1"
            else "legacy_continuous_state_not_teacher_final_explicit_twin"
        ),
        "status": "validation_frozen_then_test_evaluated_once",
        "test_evaluated_at": datetime.now().astimezone().isoformat(),
        "test_data_loaded": True,
        "test_state_loaded": True,
        "test_evaluated": True,
        "test_targets": 47,
        "test_state_audit": state_audit,
        "test_metrics": {
            "absolute_only": absolute_metrics,
            "joint_direct_diagnostic": direct_metrics,
            "teacher_pair": teacher_metrics,
        },
        "teacher_pair_rmse_change_percent_vs_absolute": float(
            (teacher_metrics["rmse"] / absolute_rmse - 1.0) * 100.0
        ),
        "all_test_references_from_train50": True,
        "test_labels_used_for_reference_selection": False,
    }
    completion_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation_absolute = frozen["validation_metrics"]["absolute_only"]
    validation_teacher = frozen["validation_metrics"]["teacher_pair"]
    validation_rmse_change = (
        validation_teacher["rmse"] / validation_absolute["rmse"] - 1.0
    ) * 100.0
    test_mae_change = (
        teacher_metrics["mae"] / absolute_metrics["mae"] - 1.0
    ) * 100.0
    is_explicit_twin = (
        expected_state_protocol
        == "explicit_twin_audited_unique_window_cache_v1"
    )
    method_document = (
        "docs/teacher_explicit_twin_matlab_protocol.md"
        if is_explicit_twin
        else "docs/teacher_shared_optical_reservoir_protocol.md"
    )
    readme_title = (
        "老师最终显式 Twin 方案：共享光储备池 + 共享线性输出权重"
        if is_explicit_twin
        else "历史连续状态方案：固定共享光储备池 + 共享线性输出权重"
    )
    legacy_notice = (
        ""
        if is_explicit_twin
        else (
            "> **LEGACY：本目录不是老师最终显式 MATLAB 双分支结果。** "
            "数值只保留为连续状态历史对照。\n\n"
        )
    )
    readme = f"""# {readme_title}

{legacy_notice}本实验严格使用 50/45/47 个环比目标，储备池内部参数、掩码和状态全部固定，只闭式训练同一组输出权重 `(b,w)`。孪生训练目标为绝对 CPI 损失与关系差值损失的联合；741 对仅由 50 个训练月份内部组合，不是新增样本。

冻结配置：`alpha={config.alpha}`、`lambda_pair={config.pair_weight}`、`K={config.k_references}`、聚合=`{config.aggregation}`、`gap={config.min_gap_months}`。验证选参期间没有读取测试数据或测试状态，之后按冻结配置评价测试集一次。

| 模型 | 验证 MAE | 验证 RMSE | 测试 MAE | 测试 RMSE |
| --- | ---: | ---: | ---: | ---: |
| 同状态单光读出 | {validation_absolute['mae']:.6f} | {validation_absolute['rmse']:.6f} | {absolute_metrics['mae']:.6f} | {absolute_metrics['rmse']:.6f} |
| 老师方案孪生参考还原 | {validation_teacher['mae']:.6f} | {validation_teacher['rmse']:.6f} | {teacher_metrics['mae']:.6f} | {teacher_metrics['rmse']:.6f} |

验证 RMSE 相对单光变化 `{validation_rmse_change:+.2f}%`，但测试 MAE/RMSE 分别变化 `{test_mae_change:+.2f}%` / `{manifest['teacher_pair_rmse_change_percent_vs_absolute']:+.2f}%`。因此该配置在验证区间表现更好，却没有迁移到测试区间，不能得出孪生方案优于单光的结论。验证中 `lambda_pair=0` 的参考校准消融与正关系损失几乎相同，说明验证增益主要来自历史参考校准，而非差值监督本身。

关键文件：

- `tables/validation_model_comparison.csv`
- `tables/test_model_comparison.csv`
- `tables/test_prediction_comparison.csv`
- `tables/selected_configuration.json`
- `figures/validation_prediction_comparison.png`
- `figures/test_prediction_comparison.png`
- `experiment_manifest.json`

方法与边界详见 `{method_document}`。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return comparison
