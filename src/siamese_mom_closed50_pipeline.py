"""Run the leakage-safe MoM strict-closed50 Siamese reservoir experiment.

The optical reservoir and its cached states stay frozen. Only parameters that
exist in the Siamese branch are searched. The test split is not loaded until
the validation-only SSA search has selected and frozen one configuration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.config import RESULTS_DIR, ROOT_DIR
from src.siamese_closed50_experiment import _load_split, build_closed_train_pool
from src.siamese_closed50_ssa_optimization import (
    BASELINE_POSITION,
    DEFAULT_ITERATIONS,
    DEFAULT_POPULATION,
    DEFAULT_SEEDS,
    CandidateEvaluator,
    SiameseSearchConfig,
    _build_test_pairs,
    decode_position,
    predict_with_power_distance,
    sparrow_search,
)
from src.siamese_reservoir_regression import (
    ReadoutBundle,
    load_state_lookup,
    regression_metrics,
    save_model,
)
from src.siamese_validation_search import load_validation_state_lookup


PROFILE_ROOT = (
    ROOT_DIR / "matlab" / "optical_reservoir_cpi_mom_recent50_20260730"
)
DATA_DIR = PROFILE_ROOT / "data"
STATE_DIR = PROFILE_ROOT / "states"
ORDINARY_DIR = RESULTS_DIR / "optical_reservoir_mom_recent50_20260730"
OUTPUT_DIR = RESULTS_DIR / "siamese_optical_mom_closed50_ssa_20260730"
EXPECTED_SPLITS = {
    "train": (50, "2014-07", "2018-08"),
    "val": (45, "2018-09", "2022-05"),
    "test": (47, "2022-06", "2026-04"),
}


def audit_profile(
    data_dir: Path = DATA_DIR,
    state_dir: Path = STATE_DIR,
) -> dict[str, object]:
    """Verify the isolated MoM data and MATLAB state caches end to end."""
    index = pd.read_csv(data_dir / "sample_index.csv")
    raw = pd.read_csv(
        ROOT_DIR / "data_processed" / "cpi_data_lastmonth=100.csv"
    )
    raw["date"] = pd.to_datetime(
        dict(year=raw["year"], month=raw["month"], day=1)
    )
    raw = raw.sort_values("date").reset_index(drop=True)
    expected_months = pd.date_range(raw["date"].iloc[0], raw["date"].iloc[-1], freq="MS")
    if not np.array_equal(raw["date"].to_numpy(), expected_months.to_numpy()):
        raise ValueError("the source actual sequence contains a missing or duplicate month")
    actual_by_month = raw.set_index("date")["actual"].astype(float)
    if index["sample_id"].duplicated().any():
        raise ValueError("sample_index.csv contains duplicate sample_id values")
    if index["target_date"].duplicated().any():
        raise ValueError("sample_index.csv contains duplicate target months")

    payload = loadmat(data_dir / "cpi_windows.mat")
    masks: list[np.ndarray] = []
    state_width: int | None = None
    split_rows: dict[str, dict[str, object]] = {}
    for split, (expected_count, first_date, last_date) in EXPECTED_SPLITS.items():
        rows = index.loc[index["split"].eq(split)].reset_index(drop=True)
        if (
            len(rows) != expected_count
            or str(rows["target_date"].iloc[0]) != first_date
            or str(rows["target_date"].iloc[-1]) != last_date
        ):
            raise ValueError(f"{split} does not match the frozen 50/45/47 split")

        X = np.asarray(payload[f"X_{split}"], dtype=float)
        y = np.asarray(payload[f"y_{split}"], dtype=float).reshape(-1)
        ids = np.asarray(payload[f"sample_id_{split}"]).reshape(-1).astype(int)
        if X.shape != (expected_count, 12):
            raise ValueError(f"{split} windows have shape {X.shape}, expected {(expected_count, 12)}")
        if not np.array_equal(ids, rows["sample_id"].to_numpy(dtype=int)):
            raise ValueError(f"{split} MAT sample IDs do not match sample_index.csv")
        if not np.allclose(y, rows["y"].to_numpy(dtype=float)):
            raise ValueError(f"{split} MAT targets do not match sample_index.csv")
        target_months = pd.to_datetime(rows["target_date"] + "-01")
        expected_targets = actual_by_month.loc[target_months].to_numpy(dtype=float)
        expected_windows = np.vstack(
            [
                actual_by_month.loc[
                    pd.date_range(
                        target_month - pd.DateOffset(months=12),
                        target_month - pd.DateOffset(months=1),
                        freq="MS",
                    )
                ].to_numpy(dtype=float)
                for target_month in target_months
            ]
        )
        if not np.allclose(y, expected_targets):
            raise ValueError(f"{split} targets do not match the source actual sequence")
        if not np.allclose(X, expected_windows):
            raise ValueError(f"{split} windows are not the continuous previous 12 actual values")

        state_payload = loadmat(state_dir / f"states_{split}.mat")
        states = np.asarray(state_payload["state_matrix"], dtype=float)
        state_ids = np.asarray(state_payload["sample_id"]).reshape(-1).astype(int)
        state_targets = np.asarray(state_payload["target"]).reshape(-1).astype(float)
        if states.shape[0] != expected_count or states.ndim != 2:
            raise ValueError(f"{split} state matrix has invalid shape {states.shape}")
        if state_width is None:
            state_width = states.shape[1]
        elif states.shape[1] != state_width:
            raise ValueError("state files use different virtual-node counts")
        if not np.array_equal(state_ids, ids):
            raise ValueError(f"{split} state IDs do not match the MAT data")
        if not np.allclose(state_targets, y):
            raise ValueError(f"{split} state targets do not match the MAT data")
        if not np.isfinite(states).all() or float(np.std(states)) == 0.0:
            raise ValueError(f"{split} states are non-finite or constant")
        masks.append(np.asarray(state_payload["mask"]))
        split_rows[split] = {
            "num_targets": expected_count,
            "first_target_date": first_date,
            "last_target_date": last_date,
            "state_shape": list(states.shape),
        }

    if not all(np.array_equal(masks[0], mask) for mask in masks[1:]):
        raise ValueError("train/validation/test states do not share one mask")
    train_ids = set(index.loc[index["split"].eq("train"), "sample_id"].astype(int))
    return {
        "status": "passed",
        "target_scale": "CPI MoM index (previous month=100)",
        "window_size": 12,
        "source_sequence_verified": True,
        "state_width": int(state_width or 0),
        "shared_mask": True,
        "splits": split_rows,
        "closed_train_sample_ids": [min(train_ids), max(train_ids)],
    }


def _evaluate_on_test(
    result: dict[str, object],
    evaluator: CandidateEvaluator,
    train_pool: dict[str, object],
    test_split: dict[str, object],
    all_states: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    config = result["config"]
    pairs, candidates = _build_test_pairs(config, evaluator, train_pool, test_split)
    pair_predictions, predictions = predict_with_power_distance(
        result["scaler"],
        result["model"],
        pairs,
        all_states,
        config.distance_power,
    )
    metrics = regression_metrics(
        predictions["cpi_actual"].to_numpy(dtype=float),
        predictions["cpi_predicted"].to_numpy(dtype=float),
    )
    return candidates, pair_predictions, predictions, metrics


def _ordinary_outputs(
    ordinary_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(
        ordinary_dir / "tables" / "optical_reservoir_metrics.csv"
    ).set_index("split")
    predictions = pd.read_csv(
        ordinary_dir / "tables" / "optical_reservoir_predictions_test.csv"
    )[["sample_id", "target_date", "cpi_actual", "cpi_predicted"]].rename(
        columns={
            "sample_id": "sample_i_id",
            "cpi_predicted": "cpi_predicted_ordinary",
        }
    )
    return metrics, predictions


def _save_figures(
    history: pd.DataFrame,
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for seed, rows in history.groupby("seed"):
        rows = rows.sort_values("iteration")
        axis.plot(rows["iteration"], rows["best_fitness"], marker="o", label=f"seed={int(seed)}")
    axis.set_xlabel("SSA iteration")
    axis.set_ylabel("Validation fitness")
    axis.set_title("MoM strict-closed50 SSA validation convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "ssa_validation_convergence.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    axis.plot(predictions["target_date"], predictions["cpi_actual"], color="black", linewidth=2, label="Actual MoM CPI")
    axis.plot(predictions["target_date"], predictions["cpi_predicted_ordinary"], label="Ordinary optical reservoir")
    axis.plot(predictions["target_date"], predictions["cpi_predicted_siamese_baseline"], label="Siamese baseline")
    axis.plot(predictions["target_date"], predictions["cpi_predicted_ssa"], label="SSA Siamese")
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI MoM index (previous month=100)")
    axis.set_title("MoM strict-closed50 test predictions")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_prediction_comparison.png", dpi=180)
    plt.close(fig)

    test_rows = comparison.sort_values("test_rmse")
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(test_rows))
    width = 0.36
    axis.bar(x - width / 2, test_rows["test_mae"], width, label="MAE")
    axis.bar(x + width / 2, test_rows["test_rmse"], width, label="RMSE")
    axis.set_xticks(x, test_rows["model"], rotation=15, ha="right")
    axis.set_ylabel("Error")
    axis.set_title("MoM strict-closed50 test error comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_metric_comparison.png", dpi=180)
    plt.close(fig)


def run_pipeline(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_DIR,
    state_dir: Path = STATE_DIR,
    ordinary_dir: Path = ORDINARY_DIR,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    population_size: int = DEFAULT_POPULATION,
    iterations: int = DEFAULT_ITERATIONS,
) -> pd.DataFrame:
    """Run baseline and SSA Siamese models using validation only for selection."""
    audit = audit_profile(data_dir, state_dir)
    seeds = tuple(int(seed) for seed in seeds)
    train_pool = build_closed_train_pool(data_dir)
    if len(train_pool["index"]) != 50:
        raise ValueError("strict closed experiment must expose exactly 50 train windows")

    # Test arrays and test states are deliberately not opened before selection.
    validation_split = _load_split(data_dir, "val")
    validation_states = load_validation_state_lookup(state_dir)
    evaluator = CandidateEvaluator(train_pool, validation_split, validation_states)

    baseline_config = decode_position(BASELINE_POSITION)
    baseline_result = evaluator.evaluate(baseline_config)
    histories: list[pd.DataFrame] = []
    run_rows: list[dict[str, object]] = []
    best_results: list[dict[str, object]] = []
    for seed in seeds:
        best_position, best_fitness, history = sparrow_search(
            lambda position: evaluator.evaluate(position)["fitness"],
            seed=seed,
            population_size=population_size,
            iterations=iterations,
        )
        best_result = evaluator.evaluate(decode_position(best_position))
        best_results.append(best_result)
        histories.append(pd.DataFrame(history))
        run_rows.append(
            {
                "seed": seed,
                **asdict(best_result["config"]),
                "shape_weight": best_result["config"].shape_weight,
                "fitness": best_fitness,
                "val_mae": best_result["val_mae"],
                "val_rmse": best_result["val_rmse"],
                "validation_block_rmse_std": best_result["validation_block_rmse_std"],
                "selected_alpha": best_result["selected_alpha"],
            }
        )
    selected_result = min(
        best_results,
        key=lambda row: (row["fitness"], row["val_rmse"], row["val_mae"]),
    )
    selected_config: SiameseSearchConfig = selected_result["config"]

    # Configuration is frozen here. Only now may the test arrays/states be read.
    test_split = _load_split(data_dir, "test")
    all_states = load_state_lookup(state_dir)
    baseline_candidates, baseline_pair_predictions, baseline_predictions, baseline_test_metrics = _evaluate_on_test(
        baseline_result, evaluator, train_pool, test_split, all_states
    )
    test_candidates, test_pair_predictions, test_predictions, test_metrics = _evaluate_on_test(
        selected_result, evaluator, train_pool, test_split, all_states
    )
    ordinary_metrics, ordinary_predictions = _ordinary_outputs(ordinary_dir)

    comparison = pd.DataFrame(
        [
            {
                "model": "ordinary_optical_reservoir",
                "optimized_by_ssa": False,
                "val_mae": float(ordinary_metrics.loc["val", "mae"]),
                "val_rmse": float(ordinary_metrics.loc["val", "rmse"]),
                "test_mae": float(ordinary_metrics.loc["test", "mae"]),
                "test_rmse": float(ordinary_metrics.loc["test", "rmse"]),
            },
            {
                "model": "siamese_closed50_baseline",
                "optimized_by_ssa": False,
                "val_mae": float(baseline_result["val_mae"]),
                "val_rmse": float(baseline_result["val_rmse"]),
                "test_mae": baseline_test_metrics["mae"],
                "test_rmse": baseline_test_metrics["rmse"],
            },
            {
                "model": "siamese_closed50_ssa",
                "optimized_by_ssa": True,
                "val_mae": float(selected_result["val_mae"]),
                "val_rmse": float(selected_result["val_rmse"]),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            },
        ]
    )
    baseline_test = baseline_predictions[
        ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(columns={"cpi_predicted": "cpi_predicted_siamese_baseline"})
    selected_test = test_predictions[
        ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(columns={"cpi_predicted": "cpi_predicted_ssa"})
    unified = ordinary_predictions.merge(
        baseline_test,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    ).merge(
        selected_test,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    )
    if len(unified) != 47:
        raise ValueError(f"test comparison has {len(unified)} rows, expected 47")
    for name in ("ordinary", "siamese_baseline", "ssa"):
        unified[f"residual_{name}"] = unified[f"cpi_predicted_{name}"] - unified["cpi_actual"]
        unified[f"absolute_error_{name}"] = unified[f"residual_{name}"].abs()

    train_ids = set(train_pool["index"]["sample_id"].astype(int))
    selected_reference_ids = set(selected_result["train_pairs"]["sample_j_id"].astype(int))
    selected_reference_ids.update(selected_result["validation_pairs"]["sample_j_id"].astype(int))
    selected_reference_ids.update(test_pair_predictions["sample_j_id"].astype(int))
    if not selected_reference_ids.issubset(train_ids):
        raise ValueError("a selected reference escaped the closed 50-window pool")

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    evaluations = pd.DataFrame(evaluator.evaluation_rows).sort_values(
        ["fitness", "val_rmse", "val_mae"]
    )
    history_table = pd.concat(histories, ignore_index=True)
    pd.DataFrame(run_rows).to_csv(table_dir / "ssa_run_summary.csv", index=False)
    evaluations.to_csv(table_dir / "ssa_all_unique_evaluations.csv", index=False)
    history_table.to_csv(table_dir / "ssa_iteration_history.csv", index=False)
    comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    unified.to_csv(table_dir / "test_prediction_comparison.csv", index=False)
    baseline_result["validation_predictions"].to_csv(table_dir / "baseline_validation_predictions.csv", index=False)
    baseline_predictions.to_csv(table_dir / "baseline_test_predictions.csv", index=False)
    baseline_candidates.to_csv(table_dir / "baseline_test_candidates.csv", index=False)
    baseline_pair_predictions.to_csv(table_dir / "baseline_test_pair_predictions.csv", index=False)
    selected_result["train_pairs"].to_csv(table_dir / "ssa_selected_train_pairs.csv", index=False)
    selected_result["validation_pairs"].to_csv(table_dir / "ssa_selected_validation_pairs.csv", index=False)
    selected_result["validation_predictions"].to_csv(table_dir / "ssa_selected_validation_predictions.csv", index=False)
    test_candidates.to_csv(table_dir / "ssa_selected_test_candidates.csv", index=False)
    test_pair_predictions.to_csv(table_dir / "ssa_selected_test_pair_predictions.csv", index=False)
    test_predictions.to_csv(table_dir / "ssa_selected_test_predictions.csv", index=False)
    pd.DataFrame(selected_result["alpha_trials"]).to_csv(table_dir / "ssa_selected_alpha_trials.csv", index=False)
    save_model(
        ReadoutBundle(
            scaler=selected_result["scaler"],
            model=selected_result["model"],
            feature_mode="signed_diff",
            aggregation="ssa_inverse_power_search_distance",
        ),
        model_dir / "ssa_siamese_readout.npz",
    )
    _save_figures(history_table, comparison, unified, figure_dir)

    ordinary_test_rmse = float(ordinary_metrics.loc["test", "rmse"])
    manifest = {
        "experiment": "MoM strict-closed50 Siamese optical reservoir with SSA",
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_status": "fresh MoM evaluation; validation selected, test evaluated once",
        "profile_audit": audit,
        "frozen_optical_reservoir": True,
        "reservoir_parameters_changed": False,
        "state_directory": str(state_dir),
        "data_directory": str(data_dir),
        "accessible_train_windows": 50,
        "validation_targets": 45,
        "test_targets": 47,
        "baseline_configuration": asdict(baseline_config),
        "selected_configuration": asdict(selected_config),
        "selected_shape_weight": selected_config.shape_weight,
        "selected_alpha": selected_result["selected_alpha"],
        "optimized_parameters": [
            "gap_months",
            "k_references",
            "level_weight",
            "distance_power",
            "max_pairs_per_bin",
        ],
        "feature_mode": "signed_diff_50d_fixed",
        "fitness": "validation RMSE + 0.10 * chronological-block RMSE std",
        "ssa": {
            "seeds": list(seeds),
            "population_size": population_size,
            "iterations": iterations,
            "unique_evaluations": len(evaluations),
        },
        "all_references_inside_closed_train50": True,
        "num_selected_train_pairs": int(len(selected_result["train_pairs"])),
        "num_selected_train_pair_targets": int(
            selected_result["train_pairs"]["sample_i_id"].nunique()
        ),
        "num_selected_validation_pairs": int(
            len(selected_result["validation_pairs"])
        ),
        "num_selected_test_pairs": int(len(test_pair_predictions)),
        "validation_or_test_labels_used_for_reference_selection": False,
        "test_loaded_after_configuration_freeze": True,
        "ordinary_test_rmse": ordinary_test_rmse,
        "ssa_test_rmse": test_metrics["rmse"],
        "ssa_test_rmse_change_percent_vs_ordinary": (
            (test_metrics["rmse"] / ordinary_test_rmse - 1.0) * 100.0
        ),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# 环比严格封闭50样本孪生光储备池与SSA",
                "",
                "本实验使用单光储备池已经生成的同一份50维状态，不改变光储备池参数。",
                "孪生模型只能访问2014-07至2018-08的50个训练窗口，验证和测试参考均来自该固定参考池。",
                "",
                f"- SSA配置：gap={selected_config.gap_months}，K={selected_config.k_references}，"
                f"β={selected_config.level_weight}，p={selected_config.distance_power}，"
                f"M={selected_config.max_pairs_per_bin}，alpha={selected_result['selected_alpha']}",
                f"- 训练配对：{len(selected_result['train_pairs'])}对，"
                f"覆盖{selected_result['train_pairs']['sample_i_id'].nunique()}个配对目标；"
                "其余训练窗口仍可作为参考窗口。",
                f"- 验证集：MAE={selected_result['val_mae']:.6f}，"
                f"RMSE={selected_result['val_rmse']:.6f}",
                f"- 测试集：MAE={test_metrics['mae']:.6f}，RMSE={test_metrics['rmse']:.6f}",
                f"- 相对单光测试RMSE变化："
                f"{(test_metrics['rmse'] / ordinary_test_rmse - 1.0) * 100.0:+.2f}%",
                "",
                "所有参数只根据验证集选择；测试集在配置冻结后读取并评估一次。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_pipeline(
        output_dir=args.output_dir,
        seeds=args.seeds,
        population_size=args.population_size,
        iterations=args.iterations,
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
