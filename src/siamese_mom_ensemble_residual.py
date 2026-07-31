"""Stability ensemble and constrained residual correction for MoM closed50.

This experiment reuses the three validation-selected winners from independent
SSA runs.  The optical reservoir states remain frozen.  An equal-weight
ensemble is formed first, then a single convex residual-correction strength is
selected on validation data:

    y_final = y_ordinary + lambda * (y_siamese_ensemble - y_ordinary)

With ``lambda`` restricted to ``[0, 1]``, the final prediction cannot extrapolate
beyond the ordinary and Siamese predictions.  Test data are loaded only after
the member configurations and ``lambda`` have been frozen.
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

from src.config import RESULTS_DIR
from src.siamese_closed50_experiment import _load_split, build_closed_train_pool
from src.siamese_closed50_ssa_optimization import (
    CandidateEvaluator,
    SiameseSearchConfig,
    _build_test_pairs,
    predict_with_power_distance,
)
from src.siamese_mom_closed50_pipeline import (
    DATA_DIR,
    ORDINARY_DIR,
    STATE_DIR,
    audit_profile,
)
from src.siamese_reservoir_regression import (
    ReadoutBundle,
    load_state_lookup,
    regression_metrics,
    save_model,
)
from src.siamese_validation_search import load_validation_state_lookup


SSA_RESULT_DIR = RESULTS_DIR / "siamese_optical_mom_closed50_ssa_20260730"
OUTPUT_DIR = RESULTS_DIR / "siamese_optical_mom_ssa_ensemble_residual_20260731"
DEFAULT_STRENGTHS = tuple(np.linspace(0.0, 1.0, 21))
STABILITY_PENALTY = 0.10
KEY_COLUMNS = ["sample_i_id", "target_date", "cpi_actual"]


def load_seed_winner_configs(
    summary_path: Path,
) -> list[tuple[int, SiameseSearchConfig]]:
    """Load one validation-selected configuration per independent SSA seed."""
    table = pd.read_csv(summary_path).sort_values("seed")
    required = {
        "seed",
        "gap_months",
        "k_references",
        "level_weight",
        "distance_power",
        "max_pairs_per_bin",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"SSA run summary is missing columns: {sorted(missing)}")
    if table["seed"].duplicated().any():
        raise ValueError("SSA run summary contains duplicate seeds")

    winners: list[tuple[int, SiameseSearchConfig]] = []
    seen: set[SiameseSearchConfig] = set()
    for row in table.itertuples(index=False):
        config = SiameseSearchConfig(
            gap_months=int(row.gap_months),
            k_references=int(row.k_references),
            level_weight=float(row.level_weight),
            distance_power=float(row.distance_power),
            max_pairs_per_bin=int(row.max_pairs_per_bin),
        )
        if config in seen:
            continue
        winners.append((int(row.seed), config))
        seen.add(config)
    if len(winners) < 2:
        raise ValueError("at least two distinct SSA winners are required")
    return winners


def build_equal_weight_ensemble(
    member_predictions: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Align member predictions and calculate their mean and disagreement."""
    if not member_predictions:
        raise ValueError("member_predictions cannot be empty")

    merged: pd.DataFrame | None = None
    prediction_columns: list[str] = []
    for name, predictions in member_predictions:
        column = f"cpi_predicted_{name}"
        member = predictions[KEY_COLUMNS + ["cpi_predicted"]].rename(
            columns={"cpi_predicted": column}
        )
        if member["sample_i_id"].duplicated().any():
            raise ValueError(f"member {name} contains duplicate target IDs")
        merged = (
            member
            if merged is None
            else merged.merge(member, on=KEY_COLUMNS, validate="one_to_one")
        )
        prediction_columns.append(column)

    assert merged is not None
    values = merged[prediction_columns].to_numpy(dtype=float)
    merged["cpi_predicted_ensemble"] = np.mean(values, axis=1)
    merged["member_prediction_std"] = np.std(values, axis=1)
    merged["error_ensemble"] = (
        merged["cpi_predicted_ensemble"] - merged["cpi_actual"]
    )
    merged["absolute_error_ensemble"] = merged["error_ensemble"].abs()
    return merged.sort_values("target_date").reset_index(drop=True)


def chronological_block_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    blocks: int = 3,
) -> list[float]:
    """Return RMSE over consecutive, nearly equal-sized time blocks."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) != len(predicted) or len(actual) < blocks:
        raise ValueError("invalid arrays for chronological block RMSE")
    return [
        regression_metrics(actual[index], predicted[index])["rmse"]
        for index in np.array_split(np.arange(len(actual)), blocks)
    ]


def score_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate ordinary errors plus the validation stability objective."""
    metrics = regression_metrics(actual, predicted)
    block_rmse = chronological_block_rmse(actual, predicted)
    block_std = float(np.std(block_rmse))
    return {
        **metrics,
        "block_rmse_1": block_rmse[0],
        "block_rmse_2": block_rmse[1],
        "block_rmse_3": block_rmse[2],
        "block_rmse_std": block_std,
        "fitness": float(metrics["rmse"] + STABILITY_PENALTY * block_std),
    }


def select_residual_strength(
    actual: np.ndarray,
    ordinary_prediction: np.ndarray,
    siamese_prediction: np.ndarray,
    strengths: Iterable[float] = DEFAULT_STRENGTHS,
) -> tuple[float, pd.DataFrame]:
    """Select a convex Siamese residual-correction strength on validation."""
    actual = np.asarray(actual, dtype=float)
    ordinary_prediction = np.asarray(ordinary_prediction, dtype=float)
    siamese_prediction = np.asarray(siamese_prediction, dtype=float)
    if not (
        len(actual) == len(ordinary_prediction) == len(siamese_prediction)
    ):
        raise ValueError("residual correction arrays have different lengths")

    rows: list[dict[str, float]] = []
    for strength in strengths:
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("residual strength must stay inside [0, 1]")
        predicted = ordinary_prediction + strength * (
            siamese_prediction - ordinary_prediction
        )
        rows.append({"residual_strength": strength, **score_predictions(actual, predicted)})
    trials = pd.DataFrame(rows).sort_values(
        ["fitness", "rmse", "mae", "residual_strength"]
    ).reset_index(drop=True)
    return float(trials.loc[0, "residual_strength"]), trials


def apply_residual_correction(
    ordinary_prediction: np.ndarray,
    siamese_prediction: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply the frozen convex correction rule."""
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("residual strength must stay inside [0, 1]")
    ordinary_prediction = np.asarray(ordinary_prediction, dtype=float)
    siamese_prediction = np.asarray(siamese_prediction, dtype=float)
    return ordinary_prediction + float(strength) * (
        siamese_prediction - ordinary_prediction
    )


def _ordinary_predictions(ordinary_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(
        ordinary_dir / "tables" / f"optical_reservoir_predictions_{split}.csv"
    )[["sample_id", "target_date", "cpi_actual", "cpi_predicted"]].rename(
        columns={
            "sample_id": "sample_i_id",
            "cpi_predicted": "cpi_predicted_ordinary",
        }
    )


def _add_residual_prediction(
    ensemble: pd.DataFrame,
    ordinary: pd.DataFrame,
    strength: float,
) -> pd.DataFrame:
    result = ordinary.merge(ensemble, on=KEY_COLUMNS, validate="one_to_one")
    result["cpi_predicted_residual"] = apply_residual_correction(
        result["cpi_predicted_ordinary"],
        result["cpi_predicted_ensemble"],
        strength,
    )
    for suffix in ("ordinary", "ensemble", "residual"):
        result[f"error_{suffix}"] = (
            result[f"cpi_predicted_{suffix}"] - result["cpi_actual"]
        )
        result[f"absolute_error_{suffix}"] = result[f"error_{suffix}"].abs()
    return result.sort_values("target_date").reset_index(drop=True)


def _comparison_rows(
    split: str,
    predictions: pd.DataFrame,
    best_single_prediction: np.ndarray,
) -> list[dict[str, object]]:
    actual = predictions["cpi_actual"].to_numpy(dtype=float)
    models = {
        "ordinary_optical_reservoir": predictions[
            "cpi_predicted_ordinary"
        ].to_numpy(dtype=float),
        "ssa_best_single": np.asarray(best_single_prediction, dtype=float),
        "ssa_seed_winner_ensemble": predictions[
            "cpi_predicted_ensemble"
        ].to_numpy(dtype=float),
        "ordinary_plus_siamese_residual": predictions[
            "cpi_predicted_residual"
        ].to_numpy(dtype=float),
    }
    return [
        {"model": model, "split": split, **score_predictions(actual, predicted)}
        for model, predicted in models.items()
    ]


def _save_figures(
    strength_trials: pd.DataFrame,
    comparison: pd.DataFrame,
    test_predictions: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    ordered = strength_trials.sort_values("residual_strength")
    axis.plot(
        ordered["residual_strength"],
        ordered["rmse"],
        marker="o",
        label="Validation RMSE",
    )
    axis.plot(
        ordered["residual_strength"],
        ordered["fitness"],
        marker="s",
        label="Validation stability fitness",
    )
    axis.set_xlabel("Siamese residual strength")
    axis.set_ylabel("Error")
    axis.set_title("Validation-only residual-strength selection")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_residual_strength.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    axis.plot(
        test_predictions["target_date"],
        test_predictions["cpi_actual"],
        color="black",
        linewidth=2,
        label="Actual MoM CPI",
    )
    axis.plot(
        test_predictions["target_date"],
        test_predictions["cpi_predicted_ordinary"],
        label="Ordinary optical reservoir",
    )
    axis.plot(
        test_predictions["target_date"],
        test_predictions["cpi_predicted_ensemble"],
        label="SSA seed-winner ensemble",
    )
    axis.plot(
        test_predictions["target_date"],
        test_predictions["cpi_predicted_residual"],
        label="Residual-corrected prediction",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI MoM index (previous month=100)")
    axis.set_title("MoM strict-closed50 exploratory test comparison")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_prediction_comparison.png", dpi=180)
    plt.close(fig)

    test = comparison.loc[comparison["split"].eq("test")].sort_values("rmse")
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(test))
    width = 0.36
    axis.bar(x - width / 2, test["mae"], width, label="MAE")
    axis.bar(x + width / 2, test["rmse"], width, label="RMSE")
    axis.set_xticks(x, test["model"], rotation=18, ha="right")
    axis.set_ylabel("Error")
    axis.set_title("Exploratory test error comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_metric_comparison.png", dpi=180)
    plt.close(fig)


def _markdown_table(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def run_experiment(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_DIR,
    state_dir: Path = STATE_DIR,
    ordinary_dir: Path = ORDINARY_DIR,
    ssa_result_dir: Path = SSA_RESULT_DIR,
    strengths: Iterable[float] = DEFAULT_STRENGTHS,
) -> pd.DataFrame:
    """Run validation-selected SSA ensemble and residual correction."""
    train_pool = build_closed_train_pool(data_dir)
    if len(train_pool["index"]) != 50:
        raise ValueError("strict closed experiment must expose exactly 50 train windows")

    winners = load_seed_winner_configs(
        ssa_result_dir / "tables" / "ssa_run_summary.csv"
    )
    validation_split = _load_split(data_dir, "val")
    validation_states = load_validation_state_lookup(state_dir)
    evaluator = CandidateEvaluator(train_pool, validation_split, validation_states)

    member_results: list[tuple[int, dict[str, object]]] = [
        (seed, evaluator.evaluate(config)) for seed, config in winners
    ]
    validation_members = [
        (f"seed_{seed}", result["validation_predictions"])
        for seed, result in member_results
    ]
    validation_ensemble = build_equal_weight_ensemble(validation_members)
    ordinary_validation = _ordinary_predictions(ordinary_dir, "val")
    validation_joined = ordinary_validation.merge(
        validation_ensemble, on=KEY_COLUMNS, validate="one_to_one"
    )
    selected_strength, strength_trials = select_residual_strength(
        validation_joined["cpi_actual"],
        validation_joined["cpi_predicted_ordinary"],
        validation_joined["cpi_predicted_ensemble"],
        strengths,
    )
    validation_predictions = _add_residual_prediction(
        validation_ensemble,
        ordinary_validation,
        selected_strength,
    )
    best_seed, best_result = min(
        member_results,
        key=lambda item: (
            item[1]["fitness"],
            item[1]["val_rmse"],
            item[1]["val_mae"],
        ),
    )
    best_validation = (
        best_result["validation_predictions"]
        .sort_values("target_date")["cpi_predicted"]
        .to_numpy(dtype=float)
    )
    validation_rows = _comparison_rows(
        "val", validation_predictions, best_validation
    )

    validation_candidates = pd.DataFrame(validation_rows)
    deployable = validation_candidates.loc[
        validation_candidates["model"].isin(
            [
                "ssa_best_single",
                "ssa_seed_winner_ensemble",
                "ordinary_plus_siamese_residual",
            ]
        )
    ]
    selected_model = str(
        deployable.sort_values(["fitness", "rmse", "mae"]).iloc[0]["model"]
    )

    # Everything affecting the model is frozen above. Test is first loaded here.
    audit = audit_profile(data_dir, state_dir)
    test_split = _load_split(data_dir, "test")
    all_states = load_state_lookup(state_dir)
    test_members: list[tuple[str, pd.DataFrame]] = []
    test_pair_tables: list[pd.DataFrame] = []
    for seed, result in member_results:
        config: SiameseSearchConfig = result["config"]
        pairs, _ = _build_test_pairs(
            config, evaluator, train_pool, test_split
        )
        pair_predictions, predictions = predict_with_power_distance(
            result["scaler"],
            result["model"],
            pairs,
            all_states,
            config.distance_power,
        )
        pair_predictions.insert(0, "member_seed", seed)
        test_pair_tables.append(pair_predictions)
        test_members.append((f"seed_{seed}", predictions))

    test_ensemble = build_equal_weight_ensemble(test_members)
    ordinary_test = _ordinary_predictions(ordinary_dir, "test")
    test_predictions = _add_residual_prediction(
        test_ensemble, ordinary_test, selected_strength
    )
    best_test_column = f"cpi_predicted_seed_{best_seed}"
    test_rows = _comparison_rows(
        "test",
        test_predictions,
        test_predictions[best_test_column].to_numpy(dtype=float),
    )
    comparison = pd.DataFrame([*validation_rows, *test_rows])

    member_rows = []
    for seed, result in member_results:
        config: SiameseSearchConfig = result["config"]
        member_rows.append(
            {
                "seed": seed,
                **asdict(config),
                "shape_weight": config.shape_weight,
                "selected_alpha": result["selected_alpha"],
                "val_mae": result["val_mae"],
                "val_rmse": result["val_rmse"],
                "validation_fitness": result["fitness"],
            }
        )
    member_table = pd.DataFrame(member_rows)

    train_ids = set(train_pool["index"]["sample_id"].astype(int))
    reference_ids: set[int] = set()
    for _, result in member_results:
        reference_ids.update(result["train_pairs"]["sample_j_id"].astype(int))
        reference_ids.update(result["validation_pairs"]["sample_j_id"].astype(int))
    for table in test_pair_tables:
        reference_ids.update(table["sample_j_id"].astype(int))
    if not reference_ids.issubset(train_ids):
        raise ValueError("an ensemble reference escaped the closed train50 pool")

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    member_table.to_csv(table_dir / "ensemble_member_configurations.csv", index=False)
    strength_trials.to_csv(table_dir / "residual_strength_selection.csv", index=False)
    validation_predictions.to_csv(table_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(table_dir / "test_predictions.csv", index=False)
    pd.concat(test_pair_tables, ignore_index=True).to_csv(
        table_dir / "member_test_pair_predictions.csv", index=False
    )
    comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    for seed, result in member_results:
        save_model(
            ReadoutBundle(
                scaler=result["scaler"],
                model=result["model"],
                feature_mode="signed_diff",
                aggregation="ssa_inverse_power_search_distance",
            ),
            model_dir / f"ssa_member_seed_{seed}.npz",
        )
    _save_figures(strength_trials, comparison, test_predictions, figure_dir)

    test_comparison = comparison.loc[comparison["split"].eq("test")].set_index(
        "model"
    )
    ordinary_rmse = float(
        test_comparison.loc["ordinary_optical_reservoir", "rmse"]
    )
    residual_rmse = float(
        test_comparison.loc["ordinary_plus_siamese_residual", "rmse"]
    )
    ensemble_rmse = float(
        test_comparison.loc["ssa_seed_winner_ensemble", "rmse"]
    )
    selected_test_rmse = float(test_comparison.loc[selected_model, "rmse"])
    manifest = {
        "experiment": "MoM strict-closed50 SSA ensemble and residual correction",
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_status": (
            "exploratory re-analysis; this historical test interval had already "
            "been observed in earlier experiments"
        ),
        "profile_audit": audit,
        "frozen_optical_reservoir": True,
        "reservoir_parameters_changed": False,
        "accessible_train_windows": 50,
        "validation_targets": 45,
        "test_targets": 47,
        "ensemble_rule": "equal mean of one winner from each independent SSA seed",
        "member_seeds": [seed for seed, _ in member_results],
        "member_configurations": [
            {"seed": seed, **asdict(result["config"])}
            for seed, result in member_results
        ],
        "residual_rule": (
            "ordinary + lambda * (SSA ensemble - ordinary), lambda constrained "
            "to [0, 1]"
        ),
        "residual_strength_grid": [float(value) for value in strengths],
        "selected_residual_strength": selected_strength,
        "selection_fitness": (
            "validation RMSE + 0.10 * std(RMSE over three chronological blocks)"
        ),
        "validation_selected_model": selected_model,
        "all_references_inside_closed_train50": True,
        "validation_or_test_labels_used_for_reference_selection": False,
        "test_loaded_after_member_and_strength_freeze": True,
        "test_rmse": {
            "ordinary": ordinary_rmse,
            "ensemble": ensemble_rmse,
            "residual": residual_rmse,
            "validation_selected_model": selected_test_rmse,
        },
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_table = comparison[
        ["model", "split", "mae", "rmse", "fitness"]
    ].copy()
    for column in ("mae", "rmse", "fitness"):
        report_table[column] = report_table[column].map(lambda value: f"{value:.6f}")
    residual_change = (residual_rmse / ordinary_rmse - 1.0) * 100.0
    ensemble_change = (ensemble_rmse / ordinary_rmse - 1.0) * 100.0
    selected_change = (selected_test_rmse / ordinary_rmse - 1.0) * 100.0
    conclusion = (
        "验证集选择的方案在这段探索性测试上也优于单光储备池。"
        if selected_change < 0
        else "验证集选择的方案在这段探索性测试上未优于单光储备池，因此暂不替换当前主结果。"
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# 环比严格封闭50：SSA稳定集成与残差式孪生修正",
                "",
                "本实验冻结光储备池及其50维状态，只优化孪生分支的组合方式。",
                "三个独立SSA种子的验证集最优配置分别重训后等权平均，再把孪生集成与单光预测之差作为受约束修正量。",
                "",
                "## 防泄漏边界",
                "",
                "- 模型只访问固定的50个训练窗口；验证集45个目标、测试集47个目标。",
                "- 验证和测试所用的历史参考全部来自训练50窗口。",
                "- SSA成员、集成规则和残差强度只依据训练集与验证集确定。",
                "- 测试集在成员及残差强度冻结后才进入本流程；但该历史测试区间此前已被项目查看，因此结果属于探索性复分析。",
                "",
                "## 方法",
                "",
                f"- SSA成员种子：{', '.join(str(seed) for seed, _ in member_results)}。",
                "- 集成：三个成员等权平均，不根据测试表现分配权重。",
                "- 残差修正：`最终值 = 单光预测 + λ × (孪生集成预测 - 单光预测)`。",
                f"- 验证集选择得到 `λ = {selected_strength:.2f}`；λ被限制在[0, 1]，最终值不会越过两种基础预测。",
                f"- 按验证稳定性目标最终选中的候选：`{selected_model}`。",
                "",
                "## 结果",
                "",
                _markdown_table(report_table),
                "",
                f"- SSA等权集成相对单光测试RMSE变化：{ensemble_change:+.2f}%。",
                f"- 残差式修正相对单光测试RMSE变化：{residual_change:+.2f}%。",
                f"- 验证集最终选中方案相对单光测试RMSE变化：{selected_change:+.2f}%。",
                f"- 结论：{conclusion}",
                "",
                "这里不能因为反复查看同一测试区间而宣称获得新的无偏提升。",
                "正式结论应冻结当前方案，等待2026年5月之后的新月份，或另做滚动回测。",
                "",
                "## 关键文件",
                "",
                "- `tables/ensemble_member_configurations.csv`：三个SSA成员及验证表现。",
                "- `tables/residual_strength_selection.csv`：λ的验证集选择过程。",
                "- `tables/validation_predictions.csv`：45个验证目标的逐月结果。",
                "- `tables/test_predictions.csv`：47个测试目标的逐月探索性结果。",
                "- `tables/model_comparison.csv`：单光、最佳单个SSA、SSA集成和残差修正的统一比较。",
                "- `figures/validation_residual_strength.png`：残差强度选择曲线。",
                "- `figures/test_prediction_comparison.png`：测试预测曲线。",
                "- `figures/test_metric_comparison.png`：测试MAE/RMSE柱状图。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_experiment(output_dir=args.output_dir)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
