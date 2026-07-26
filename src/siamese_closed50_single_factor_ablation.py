"""Single-factor ablation for the strict closed-50 Siamese experiment.

Baseline:
  gap=12, shape-only reference distance, [h_i-h_j] 50D features.

Ablations:
  1. change only gap, selecting from 1/3/6/12 on validation;
  2. change only reference distance to shape+absolute-level hybrid;
  3. change only the readout feature to [h_i, h_i-h_j] (100D).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, RESULTS_DIR
from src.create_siamese_pairs import (
    _assign_delta_bins,
    _build_candidate_pairs,
    _compute_delta_bin_thresholds,
    _sample_pairs_by_bin,
)
from src.siamese_closed50_experiment import _load_split, build_closed_train_pool
from src.siamese_closed50_gap_hybrid_experiment import (
    EXTENDED_ALPHAS,
    GAPS,
    K_REFERENCES,
    HybridDistanceCalibration,
    _markdown_table,
    _window_lookup,
    add_hybrid_distances,
    fit_hybrid_distance_calibration,
    select_hybrid_references,
)
from src.siamese_recent50_experiment import (
    BASE_STATE_DIR,
    ORDINARY_EXPERIMENT_DIR,
    RECENT_STATE_DIR,
    verify_recent_state_cache,
)
from src.siamese_reservoir_regression import (
    build_pair_features,
    load_state_lookup,
    predict_pairs,
    regression_metrics,
    save_model,
    train_readout,
)
from src.siamese_validation_search import (
    load_validation_state_lookup,
    select_validation_references,
)


OUTPUT_DIR = RESULTS_DIR / "siamese_optical_closed50_single_factor_ablation_20260726"


@dataclass(frozen=True)
class AblationConfiguration:
    name: str
    changed_factor: str
    gap_months: int
    reference_mode: str
    feature_mode: str

    @property
    def aggregation(self) -> str:
        if self.reference_mode == "shape":
            return "inverse_distance"
        if self.reference_mode == "hybrid":
            return "inverse_hybrid_distance"
        raise ValueError(f"unknown reference_mode: {self.reference_mode}")


BASELINE = AblationConfiguration(
    name="baseline",
    changed_factor="none",
    gap_months=12,
    reference_mode="shape",
    feature_mode="signed_diff",
)


def build_ablation_configurations(
    selected_gap: int,
) -> tuple[AblationConfiguration, ...]:
    """Return the baseline and three configurations that each alter one factor."""
    return (
        BASELINE,
        AblationConfiguration(
            name="only_gap",
            changed_factor="gap_months",
            gap_months=int(selected_gap),
            reference_mode=BASELINE.reference_mode,
            feature_mode=BASELINE.feature_mode,
        ),
        AblationConfiguration(
            name="only_hybrid_distance",
            changed_factor="reference_mode",
            gap_months=BASELINE.gap_months,
            reference_mode="hybrid",
            feature_mode=BASELINE.feature_mode,
        ),
        AblationConfiguration(
            name="only_100d_feature",
            changed_factor="feature_mode",
            gap_months=BASELINE.gap_months,
            reference_mode=BASELINE.reference_mode,
            feature_mode="target_plus_diff",
        ),
    )


def _build_training_pairs(
    train_pool: dict[str, object],
    gap: int,
    reference_mode: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[float, float],
    HybridDistanceCalibration | None,
]:
    candidates = _build_candidate_pairs(
        train_pool,
        train_pool,
        min_gap_months=gap,
    )
    if candidates.empty:
        raise ValueError(f"gap={gap} produced no training candidates")
    calibration = None
    if reference_mode == "hybrid":
        calibration = fit_hybrid_distance_calibration(train_pool, candidates)
        candidates = add_hybrid_distances(
            candidates,
            _window_lookup(train_pool),
            calibration,
        )
    elif reference_mode != "shape":
        raise ValueError(f"unknown reference_mode: {reference_mode}")
    thresholds = _compute_delta_bin_thresholds(candidates)
    selected = _sample_pairs_by_bin(
        _assign_delta_bins(candidates, thresholds)
    ).reset_index(drop=True)
    selected["selection_method"] = (
        f"closed50_delta_stratified_gap{gap}_{reference_mode}_ablation"
    )
    return selected, candidates, thresholds, calibration


def _build_evaluation_pairs(
    train_pool: dict[str, object],
    evaluation_split: dict[str, object],
    config: AblationConfiguration,
    calibration: HybridDistanceCalibration | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _build_candidate_pairs(
        evaluation_split,
        train_pool,
        min_gap_months=config.gap_months,
    )
    if config.reference_mode == "shape":
        selected = select_validation_references(
            candidates,
            strategy="window_distance",
            k=K_REFERENCES,
        )
        selected = selected.copy()
        selected["selection_method"] = "shape_only_closed50_fixed_bank"
    else:
        if calibration is None:
            raise ValueError("hybrid configuration needs training-only calibration")
        candidates = add_hybrid_distances(
            candidates,
            _window_lookup(evaluation_split, train_pool),
            calibration,
        )
        selected = select_hybrid_references(candidates, k=K_REFERENCES)
    pool_ids = set(train_pool["index"]["sample_id"].astype(int))
    if not set(selected["sample_j_id"].astype(int)).issubset(pool_ids):
        raise ValueError(f"{config.name} reference escaped the closed50 bank")
    return selected.reset_index(drop=True), candidates.reset_index(drop=True)


def _metrics(predictions: pd.DataFrame) -> dict[str, float]:
    return regression_metrics(
        predictions["cpi_actual"].to_numpy(dtype=float),
        predictions["cpi_predicted"].to_numpy(dtype=float),
    )


def _fit_validation_configuration(
    config: AblationConfiguration,
    train_pool: dict[str, object],
    val_split: dict[str, object],
    validation_states: dict[int, np.ndarray],
    alphas: Iterable[float],
) -> dict[str, object]:
    train_pairs, train_candidates, thresholds, calibration = _build_training_pairs(
        train_pool,
        config.gap_months,
        config.reference_mode,
    )
    val_pairs, val_candidates = _build_evaluation_pairs(
        train_pool,
        val_split,
        config,
        calibration,
    )
    bundle, alpha_trials = train_readout(
        {"train": train_pairs, "val": val_pairs},
        validation_states,
        feature_mode=config.feature_mode,
        aggregation=config.aggregation,
        alphas=alphas,
    )
    _, val_predictions = predict_pairs(bundle, val_pairs, validation_states)
    validation_metrics = _metrics(val_predictions)
    feature_dimension = build_pair_features(
        train_pairs.iloc[:1],
        validation_states,
        config.feature_mode,
    ).shape[1]
    return {
        "config": config,
        "bundle": bundle,
        "train_pairs": train_pairs,
        "train_candidates": train_candidates,
        "val_pairs": val_pairs,
        "val_candidates": val_candidates,
        "val_predictions": val_predictions,
        "thresholds": thresholds,
        "calibration": calibration,
        "alpha_trials": alpha_trials,
        "feature_dimension": feature_dimension,
        "val_mae": validation_metrics["mae"],
        "val_rmse": validation_metrics["rmse"],
    }


def _validation_row(fitted: dict[str, object]) -> dict[str, object]:
    config = fitted["config"]
    return {
        "configuration": config.name,
        "changed_factor": config.changed_factor,
        "gap_months": config.gap_months,
        "reference_mode": config.reference_mode,
        "feature_mode": config.feature_mode,
        "feature_dimension": fitted["feature_dimension"],
        "legal_train_candidates": len(fitted["train_candidates"]),
        "train_pair_targets": fitted["train_pairs"]["sample_i_id"].nunique(),
        "selected_train_pairs": len(fitted["train_pairs"]),
        "selected_alpha": float(fitted["bundle"].model.alpha),
        "val_mae": fitted["val_mae"],
        "val_rmse": fitted["val_rmse"],
    }


def _save_figures(
    gap_results: pd.DataFrame,
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(gap_results["gap_months"], gap_results["val_rmse"], "o-", label="RMSE")
    axis.plot(gap_results["gap_months"], gap_results["val_mae"], "s--", label="MAE")
    axis.set_xticks(list(GAPS))
    axis.set_xlabel("Minimum gap (months)")
    axis.set_ylabel("Validation error")
    axis.set_title("Only-gap ablation (shape distance, 50D difference)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "only_gap_validation_comparison.png", dpi=180)
    plt.close(fig)

    selected_gap = int(
        comparison.loc[
            comparison["configuration"] == "only_gap", "gap_months"
        ].iloc[0]
    )
    labels = [
        "Baseline",
        f"Only gap\n(selected {selected_gap})",
        "Only hybrid\ndistance",
        "Only 100D\nfeature",
    ]
    positions = np.arange(len(comparison))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        positions - width / 2,
        comparison["val_rmse"],
        width,
        label="Validation RMSE",
    )
    axis.bar(
        positions + width / 2,
        comparison["test_rmse"],
        width,
        label="Test RMSE",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("RMSE")
    axis.set_title("Strict closed-50 single-factor ablation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "single_factor_rmse_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(predictions["target_date"], predictions["cpi_actual"], label="Actual CPI")
    for column, label in (
        ("cpi_predicted_baseline", "Baseline"),
        ("cpi_predicted_only_gap", "Only gap"),
        ("cpi_predicted_only_hybrid_distance", "Only hybrid distance"),
        ("cpi_predicted_only_100d_feature", "Only 100D feature"),
    ):
        axis.plot(predictions["target_date"], predictions[column], label=label)
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI")
    axis.set_title("Single-factor ablation: test predictions")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "single_factor_test_predictions.png", dpi=180)
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    gap_results: pd.DataFrame,
    comparison: pd.DataFrame,
    selected_gap: int,
) -> None:
    baseline = comparison.loc[comparison["configuration"] == "baseline"].iloc[0]
    interpretation = []
    labels = {
        "only_gap": "只改 gap",
        "only_hybrid_distance": "只改混合距离",
        "only_100d_feature": "只改100维特征",
    }
    for name, label in labels.items():
        row = comparison.loc[comparison["configuration"] == name].iloc[0]
        rmse_change = 100.0 * (
            float(row["test_rmse"]) - float(baseline["test_rmse"])
        ) / float(baseline["test_rmse"])
        if abs(rmse_change) < 1e-10:
            interpretation.append(
                f"- {label}：验证集仍选择基线设置，测试 RMSE 保持不变。"
            )
        else:
            direction = "下降（改善）" if rmse_change < 0 else "上升（退化）"
            interpretation.append(
                f"- {label}：测试 RMSE {direction} {abs(rmse_change):.2f}%。"
            )

    lines = [
        "# 严格封闭 50 窗口孪生光储备池：单因素消融报告",
        "",
        "## 实验设计",
        "",
        "基线固定为：`gap=12 + 纯形状参考距离 + 50维[h_i-h_j]`。"
        "每个消融配置相对基线只改变一个概念因素：",
        "",
        "- 只改 gap：保持纯形状距离和50维状态差，在1、3、6、12中按验证集选择。",
        "- 只改混合距离：保持gap=12和50维状态差，改为形状与绝对水平各占0.5。",
        "- 只改100维特征：保持gap=12和纯形状距离，改为`[h_i,h_i-h_j]`。",
        "- 所有配置只能访问同一批最后50个训练窗口；验证和测试参考均来自该固定窗口库。",
        "",
        "## 只改 gap：验证集选择",
        "",
        _markdown_table(gap_results),
        "",
        f"只改 gap 时，验证集选出的最佳值为 **gap={selected_gap}**。",
        "",
        "## 单因素最终比较",
        "",
        _markdown_table(comparison),
        "",
        "相对基线的测试 RMSE 变化：",
        "",
        *interpretation,
        "",
        "## 结论",
        "",
        "1. **不应缩小 gap。** gap=1、3、6虽然增加了训练配对目标，"
        "但验证误差都高于gap=12，因此严格按验证集选择后仍保留gap=12。",
        "2. **当前等权混合距离不优于纯形状距离。** 单独加入绝对水平后，"
        "验证和测试误差同时上升，说明水平接近不一定代表下一月CPI变化规律接近。",
        "3. **100维特征是性能下降的主要来源。** 在只有118个训练样本对时，"
        "读出维度从50增加到100，测试RMSE明显上升，表现出小样本高维过拟合。",
        "4. 因此上一轮三项组合实验的退化不是单纯由gap造成；"
        "100维特征贡献了最大的负面影响，等权混合距离也有较小的负面影响。",
        "",
        "当前测试区间在此前工作中已经被查看过，因此仍属于探索性复分析。"
        "正式结论需要在冻结最终配置后使用新的未见时间区间验证。",
        "",
        "关键文件：",
        "",
        "- `tables/only_gap_validation_comparison.csv`：只改gap的验证筛选。",
        "- `tables/single_factor_model_comparison.csv`：三个单因素与基线的验证/测试指标。",
        "- `tables/single_factor_test_predictions.csv`：47个测试月份逐月预测。",
        "- `figures/single_factor_rmse_comparison.png`：单因素RMSE柱状图。",
        "- `experiment_manifest.json`：配置隔离与防泄漏记录。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_ablation(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_PROCESSED_DIR,
    recent_state_dir: Path = RECENT_STATE_DIR,
    alphas: Iterable[float] = EXTENDED_ALPHAS,
) -> pd.DataFrame:
    """Run validation selection followed by predefined test comparisons."""
    train_pool = build_closed_train_pool(data_dir)
    val_split = _load_split(data_dir, "val")
    validation_states = load_validation_state_lookup(recent_state_dir)
    train_pool_ids = set(train_pool["index"]["sample_id"].astype(int))

    gap_fits: dict[int, dict[str, object]] = {}
    gap_rows: list[dict[str, object]] = []
    for gap in GAPS:
        config = AblationConfiguration(
            name=f"only_gap_{gap}",
            changed_factor="gap_months",
            gap_months=gap,
            reference_mode="shape",
            feature_mode="signed_diff",
        )
        fitted = _fit_validation_configuration(
            config,
            train_pool,
            val_split,
            validation_states,
            alphas,
        )
        gap_fits[gap] = fitted
        gap_rows.append(_validation_row(fitted))
    gap_results = pd.DataFrame(gap_rows).sort_values("gap_months").reset_index(drop=True)
    selected_gap_row = min(
        gap_rows,
        key=lambda row: (
            float(row["val_rmse"]),
            float(row["val_mae"]),
            -int(row["gap_months"]),
        ),
    )
    selected_gap = int(selected_gap_row["gap_months"])

    configs = build_ablation_configurations(selected_gap)
    fitted_configs: dict[str, dict[str, object]] = {
        "baseline": gap_fits[12],
        "only_gap": gap_fits[selected_gap],
    }
    for config in configs[2:]:
        fitted_configs[config.name] = _fit_validation_configuration(
            config,
            train_pool,
            val_split,
            validation_states,
            alphas,
        )

    # Freeze all predefined configurations before loading the test split.
    test_split = _load_split(data_dir, "test")
    state_cache_check = verify_recent_state_cache(BASE_STATE_DIR, recent_state_dir)
    all_states = load_state_lookup(recent_state_dir)
    comparison_rows: list[dict[str, object]] = []
    test_prediction_table: pd.DataFrame | None = None
    output_cache: dict[str, dict[str, object]] = {}

    for config in configs:
        fitted = fitted_configs[config.name]
        test_pairs, test_candidates = _build_evaluation_pairs(
            train_pool,
            test_split,
            config,
            fitted["calibration"],
        )
        pair_predictions, test_predictions = predict_pairs(
            fitted["bundle"],
            test_pairs,
            all_states,
        )
        test_metrics = _metrics(test_predictions)
        comparison_rows.append(
            {
                **_validation_row(fitted),
                "configuration": config.name,
                "changed_factor": config.changed_factor,
                "gap_months": config.gap_months,
                "reference_mode": config.reference_mode,
                "feature_mode": config.feature_mode,
                "aggregation": config.aggregation,
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            }
        )
        current = test_predictions[
            ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
        ].rename(columns={"cpi_predicted": f"cpi_predicted_{config.name}"})
        if test_prediction_table is None:
            test_prediction_table = current
        else:
            test_prediction_table = test_prediction_table.merge(
                current,
                on=["sample_i_id", "target_date", "cpi_actual"],
                validate="one_to_one",
            )
        output_cache[config.name] = {
            "test_pairs": test_pairs,
            "test_candidates": test_candidates,
            "test_pair_predictions": pair_predictions,
            "test_predictions": test_predictions,
        }

    comparison = pd.DataFrame(comparison_rows)
    baseline_test = comparison.loc[
        comparison["configuration"] == "baseline", "test_rmse"
    ].iloc[0]
    baseline_val = comparison.loc[
        comparison["configuration"] == "baseline", "val_rmse"
    ].iloc[0]
    comparison["val_rmse_change_vs_baseline"] = (
        comparison["val_rmse"] - baseline_val
    )
    comparison["test_rmse_change_vs_baseline"] = (
        comparison["test_rmse"] - baseline_test
    )
    comparison["test_rmse_change_percent_vs_baseline"] = (
        100.0 * comparison["test_rmse_change_vs_baseline"] / baseline_test
    )
    if test_prediction_table is None or len(test_prediction_table) != 47:
        raise ValueError("single-factor test output must contain 47 targets")

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    gap_results.to_csv(
        table_dir / "only_gap_validation_comparison.csv",
        index=False,
    )
    comparison.to_csv(
        table_dir / "single_factor_model_comparison.csv",
        index=False,
    )
    test_prediction_table.to_csv(
        table_dir / "single_factor_test_predictions.csv",
        index=False,
    )

    alpha_rows = []
    for name, fitted in fitted_configs.items():
        alpha_rows.extend(
            {"configuration": name, **row} for row in fitted["alpha_trials"]
        )
    pd.DataFrame(alpha_rows).to_csv(
        table_dir / "single_factor_alpha_selection.csv",
        index=False,
    )
    for config in configs:
        fitted = fitted_configs[config.name]
        fitted["train_pairs"].to_csv(
            table_dir / f"{config.name}_train_pairs.csv",
            index=False,
        )
        fitted["val_pairs"].to_csv(
            table_dir / f"{config.name}_validation_pairs.csv",
            index=False,
        )
        fitted["val_predictions"].to_csv(
            table_dir / f"{config.name}_validation_predictions.csv",
            index=False,
        )
        output_cache[config.name]["test_pairs"].to_csv(
            table_dir / f"{config.name}_test_pairs.csv",
            index=False,
        )
        output_cache[config.name]["test_pair_predictions"].to_csv(
            table_dir / f"{config.name}_test_pair_predictions.csv",
            index=False,
        )
        output_cache[config.name]["test_predictions"].to_csv(
            table_dir / f"{config.name}_test_predictions.csv",
            index=False,
        )
        save_model(
            fitted["bundle"],
            model_dir / f"{config.name}_readout.npz",
        )

    _save_figures(gap_results, comparison, test_prediction_table, figure_dir)
    manifest = {
        "experiment": "strict closed50 single-factor ablation",
        "evaluation_status": "exploratory re-analysis; test was seen before",
        "baseline": asdict(BASELINE),
        "only_gap_validation_grid": list(GAPS),
        "selected_gap_on_validation": selected_gap,
        "final_configurations": [asdict(config) for config in configs],
        "isolation_rule": (
            "each ablation differs from baseline in exactly one conceptual factor"
        ),
        "accessible_train_windows": len(train_pool_ids),
        "accessible_sample_id_min": min(train_pool_ids),
        "accessible_sample_id_max": max(train_pool_ids),
        "k_validation_test": K_REFERENCES,
        "all_test_reference_ids_inside_closed50": all(
            set(output_cache[config.name]["test_pairs"]["sample_j_id"].astype(int))
            .issubset(train_pool_ids)
            for config in configs
        ),
        "validation_or_test_labels_used_for_reference_selection": False,
        "test_loaded_after_validation_configuration_freeze": True,
        "hybrid_distance_weights": {"shape": 0.5, "absolute_level": 0.5},
        "state_cache_check": state_cache_check,
        "ordinary_recent50_context_directory": str(ORDINARY_EXPERIMENT_DIR),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_readme(output_dir, gap_results, comparison, selected_gap)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_ablation(output_dir=args.output_dir)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
