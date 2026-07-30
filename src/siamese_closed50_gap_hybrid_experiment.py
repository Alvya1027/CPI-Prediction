"""Gap and hybrid-reference experiment for the strict closed-50 setting.

The experiment keeps the accessible data budget fixed at the same latest
50 chronological training windows. It compares minimum window gaps of
1, 3, 6, and 12 months on validation data, selects one gap by validation
RMSE, and evaluates only that selected configuration on the test split.

Reference ranking combines:
  1. z-scored 12-month window shape distance; and
  2. absolute CPI level distance based on the window mean and latest value.

The readout feature is [h_i, h_i - h_j], which is 100-dimensional for the
current 50-node optical-reservoir state cache.
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
from src.siamese_closed50_experiment import (
    _load_split,
    build_closed_train_pool,
)
from src.siamese_recent50_experiment import (
    BASE_STATE_DIR,
    ORDINARY_EXPERIMENT_DIR,
    RECENT_STATE_DIR,
    verify_recent_state_cache,
)
from src.siamese_reservoir_regression import (
    DEFAULT_ALPHAS,
    build_pair_features,
    load_state_lookup,
    predict_pairs,
    regression_metrics,
    save_model,
    train_readout,
)
from src.siamese_validation_search import load_validation_state_lookup


GAPS = (1, 3, 6, 12)
K_REFERENCES = 10
FEATURE_MODE = "target_plus_diff"
AGGREGATION = "inverse_hybrid_distance"
EXTENDED_ALPHAS = (*DEFAULT_ALPHAS, 1_000.0, 10_000.0, 100_000.0)
OUTPUT_DIR = RESULTS_DIR / "siamese_optical_closed50_gap_hybrid_20260726"


@dataclass(frozen=True)
class HybridDistanceCalibration:
    """Training-only scales used by the hybrid reference distance."""

    mean_level_std: float
    last_level_std: float
    shape_distance_median: float
    level_distance_median: float
    shape_weight: float = 0.5
    level_weight: float = 0.5


def _safe_scale(value: float) -> float:
    return max(float(value), 1e-8)


def _window_lookup(*splits: dict[str, object]) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    for split in splits:
        index = split["index"]
        for row, sample_id in enumerate(index["sample_id"].to_numpy(dtype=int)):
            if int(sample_id) in lookup:
                raise ValueError(f"duplicate sample_id in window lookup: {sample_id}")
            lookup[int(sample_id)] = np.asarray(split["X"][row], dtype=float)
    return lookup


def _raw_level_distance(
    pairs: pd.DataFrame,
    windows: dict[int, np.ndarray],
    mean_level_std: float,
    last_level_std: float,
) -> np.ndarray:
    x_i = np.vstack([windows[int(value)] for value in pairs["sample_i_id"]])
    x_j = np.vstack([windows[int(value)] for value in pairs["sample_j_id"]])
    mean_difference = (x_i.mean(axis=1) - x_j.mean(axis=1)) / _safe_scale(
        mean_level_std
    )
    last_difference = (x_i[:, -1] - x_j[:, -1]) / _safe_scale(last_level_std)
    return np.sqrt(mean_difference**2 + last_difference**2) / np.sqrt(2.0)


def fit_hybrid_distance_calibration(
    train_pool: dict[str, object],
    train_candidates: pd.DataFrame,
) -> HybridDistanceCalibration:
    """Fit all distance scales using only the fixed 50-window train pool."""
    train_windows = np.asarray(train_pool["X"], dtype=float)
    mean_level_std = _safe_scale(float(np.std(train_windows.mean(axis=1))))
    last_level_std = _safe_scale(float(np.std(train_windows[:, -1])))
    windows = _window_lookup(train_pool)
    level_distances = _raw_level_distance(
        train_candidates,
        windows,
        mean_level_std,
        last_level_std,
    )
    return HybridDistanceCalibration(
        mean_level_std=mean_level_std,
        last_level_std=last_level_std,
        shape_distance_median=_safe_scale(
            float(np.median(train_candidates["window_distance"]))
        ),
        level_distance_median=_safe_scale(float(np.median(level_distances))),
    )


def add_hybrid_distances(
    candidates: pd.DataFrame,
    windows: dict[int, np.ndarray],
    calibration: HybridDistanceCalibration,
) -> pd.DataFrame:
    """Add shape, level, and equally weighted normalized hybrid distances."""
    result = candidates.copy()
    level_distance = _raw_level_distance(
        result,
        windows,
        calibration.mean_level_std,
        calibration.last_level_std,
    )
    shape_normalized = (
        result["window_distance"].to_numpy(dtype=float)
        / calibration.shape_distance_median
    )
    level_normalized = level_distance / calibration.level_distance_median
    result["shape_distance"] = result["window_distance"].to_numpy(dtype=float)
    result["level_distance"] = level_distance
    result["shape_distance_normalized"] = shape_normalized
    result["level_distance_normalized"] = level_normalized
    result["hybrid_distance"] = (
        calibration.shape_weight * shape_normalized
        + calibration.level_weight * level_normalized
    )
    return result


def build_gap_training_pairs(
    train_pool: dict[str, object],
    gap: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[float, float],
    HybridDistanceCalibration,
]:
    """Build label-stratified training pairs inside the fixed 50-window pool."""
    candidates = _build_candidate_pairs(
        train_pool,
        train_pool,
        min_gap_months=gap,
    )
    if candidates.empty:
        raise ValueError(f"gap={gap} produced no closed50 training candidates")
    calibration = fit_hybrid_distance_calibration(train_pool, candidates)
    candidates = add_hybrid_distances(
        candidates,
        _window_lookup(train_pool),
        calibration,
    )
    thresholds = _compute_delta_bin_thresholds(candidates)
    binned = _assign_delta_bins(candidates, thresholds)
    selected = _sample_pairs_by_bin(binned).reset_index(drop=True)
    selected["selection_method"] = (
        f"closed50_delta_stratified_train_gap{gap}_hybrid_metadata"
    )
    return selected, candidates, thresholds, calibration


def select_hybrid_references(
    candidates: pd.DataFrame,
    k: int = K_REFERENCES,
) -> pd.DataFrame:
    """Select references without using target CPI labels or prediction errors."""
    if k <= 0:
        raise ValueError("k must be positive")
    required = {"sample_i_id", "sample_j_id", "target_j_date", "hybrid_distance"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"hybrid candidates are missing columns: {sorted(missing)}")
    ordered = candidates.sort_values(
        ["sample_i_id", "hybrid_distance", "target_j_date", "sample_j_id"],
        ascending=[True, True, True, True],
    )
    selected = (
        ordered.groupby("sample_i_id", as_index=False, group_keys=False)
        .head(k)
        .copy()
        .reset_index(drop=True)
    )
    group_sizes = selected.groupby("sample_i_id").size()
    if len(group_sizes) != candidates["sample_i_id"].nunique():
        raise ValueError("hybrid selection dropped one or more targets")
    if not (group_sizes == k).all():
        raise ValueError(f"at least one target has fewer than {k} references")
    selected["selection_method"] = "hybrid_shape_level_closed50_fixed_bank"
    return selected


def build_gap_evaluation_pairs(
    train_pool: dict[str, object],
    evaluation_split: dict[str, object],
    gap: int,
    calibration: HybridDistanceCalibration,
    k: int = K_REFERENCES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and select evaluation references from the fixed train bank only."""
    candidates = _build_candidate_pairs(
        evaluation_split,
        train_pool,
        min_gap_months=gap,
    )
    candidates = add_hybrid_distances(
        candidates,
        _window_lookup(evaluation_split, train_pool),
        calibration,
    )
    selected = select_hybrid_references(candidates, k=k)
    pool_ids = set(train_pool["index"]["sample_id"].astype(int))
    if not set(selected["sample_j_id"].astype(int)).issubset(pool_ids):
        raise ValueError("evaluation reference escaped the fixed closed50 bank")
    return selected, candidates


def _target_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    return regression_metrics(
        predictions["cpi_actual"].to_numpy(dtype=float),
        predictions["cpi_predicted"].to_numpy(dtype=float),
    )


def _save_figures(
    gap_results: pd.DataFrame,
    test_comparison: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(gap_results["gap_months"], gap_results["val_rmse"], "o-", label="RMSE")
    axis.plot(gap_results["gap_months"], gap_results["val_mae"], "s--", label="MAE")
    axis.set_xticks(list(GAPS))
    axis.set_xlabel("Minimum gap (months)")
    axis.set_ylabel("Validation error")
    axis.set_title("Closed-50 hybrid Siamese validation comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "gap_validation_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(test_comparison["target_date"], test_comparison["cpi_actual"], label="Actual")
    axis.plot(
        test_comparison["target_date"],
        test_comparison["cpi_predicted_ordinary"],
        label="Ordinary reservoir (50)",
    )
    axis.plot(
        test_comparison["target_date"],
        test_comparison["cpi_predicted_original_siamese"],
        label="Original Siamese (gap=12)",
    )
    axis.plot(
        test_comparison["target_date"],
        test_comparison["cpi_predicted_improved_siamese"],
        label="Hybrid 100D Siamese",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI")
    axis.set_title("Strict closed-50 test prediction comparison")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "selected_gap_test_predictions.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.axhline(0.0, color="black", linewidth=0.9)
    for column, label in (
        ("residual_ordinary", "Ordinary reservoir (50)"),
        ("residual_original_siamese", "Original Siamese (gap=12)"),
        ("residual_improved_siamese", "Hybrid 100D Siamese"),
    ):
        axis.plot(test_comparison["target_date"], test_comparison[column], label=label)
    axis.set_xlabel("Target month")
    axis.set_ylabel("Prediction - actual")
    axis.set_title("Strict closed-50 test residual comparison")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "selected_gap_test_residuals.png", dpi=180)
    plt.close(fig)


def _markdown_table(table: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    def _format(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return "\n".join(lines)


def _write_readme(
    output_dir: Path,
    gap_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    selected_gap: int,
    calibration: HybridDistanceCalibration,
) -> None:
    best = gap_results.loc[gap_results["gap_months"] == selected_gap].iloc[0]
    improved = model_comparison.loc[
        model_comparison["model"] == "hybrid_100d_siamese_closed50"
    ].iloc[0]
    ordinary = model_comparison.loc[
        model_comparison["model"] == "ordinary_optical_reservoir_closed50"
    ].iloc[0]
    original = model_comparison.loc[
        model_comparison["model"] == "original_siamese_closed50_gap12"
    ].iloc[0]
    vs_ordinary = 100.0 * (
        float(ordinary["test_rmse"]) - float(improved["test_rmse"])
    ) / float(ordinary["test_rmse"])
    vs_original = 100.0 * (
        float(original["test_rmse"]) - float(improved["test_rmse"])
    ) / float(original["test_rmse"])

    lines = [
        "# 严格封闭 50 窗口：gap、混合参考距离与 100 维孪生特征实验",
        "",
        "## 实验约束",
        "",
        "- 整个模型只能访问原训练集最后 50 个时间窗口（样本 ID 162–211）。",
        "- 验证集和测试集的参考窗口只能来自上述固定 50 窗口库。",
        "- 比较 `gap = 1、3、6、12`，先按验证集 RMSE 选择 gap，再运行所选配置的测试集。",
        "- 参考距离同时考虑形状与绝对水平，两部分权重均为 0.5。",
        "- 绝对水平由窗口均值和窗口最后一个 CPI 值共同表示。",
        "- 孪生读出特征为 `[h_i, h_i-h_j]`；储备池状态为 50 维，因此输入读出层为 100 维。",
        "- 每个验证/测试目标选择 K=10 个参考，并按混合距离倒数加权。",
        "",
        "## 验证集选择结果",
        "",
        _markdown_table(gap_results),
        "",
        f"按验证集 RMSE 选出的最佳配置为 **gap={selected_gap}**："
        f"MAE={best['val_mae']:.6f}，RMSE={best['val_rmse']:.6f}，"
        f"训练配对目标={int(best['train_pair_targets'])}，"
        f"训练样本对={int(best['selected_train_pairs'])}。",
        "",
        "## 测试集对比",
        "",
        _markdown_table(model_comparison),
        "",
        f"本次混合 100 维模型相对单光储备池的测试 RMSE 变化为 {vs_ordinary:+.2f}% "
        "（正数表示误差下降）；"
        f"相对原始严格封闭孪生模型的测试 RMSE 变化为 {vs_original:+.2f}%。",
        "",
        "**结论：本次组合改动没有提升最终预测效果。** 虽然 gap=1 在四个新配置中最好，"
        "但它在验证集和测试集上都弱于原始 gap=12、纯状态差、纯形状距离的孪生方案。"
        "因此不应替换当前基线；下一步应把三项改动拆开做单因素消融，判断性能下降来自"
        "混合距离、100 维特征，还是两者的交互。",
        "",
        "## 混合距离定义",
        "",
        "1. 形状距离：分别对两个 12 个月窗口做窗口内 z-score，再计算均方根欧氏距离。",
        "2. 水平距离：窗口均值差和末值差分别除以 50 窗口训练池内的标准差，再合成为二维均方根距离。",
        "3. 两种距离分别除以训练候选对的中位数，使量纲相近；最终取 `0.5×形状 + 0.5×水平`。",
        "",
        f"训练期形状距离中位数={calibration.shape_distance_median:.6f}，"
        f"水平距离中位数={calibration.level_distance_median:.6f}。",
        "",
        "## 结果解释边界",
        "",
        "当前测试集在此前实验中已经被查看过，因此这次属于探索性复分析，不能视为完全独立的盲测。"
        "正式论文或最终结论应固定本次配置后，再使用未参与任何选择的新测试区间检验。",
        "",
        "关键文件：",
        "",
        "- `tables/gap_validation_comparison.csv`：四个 gap 的验证集结果。",
        "- `tables/model_comparison.csv`：单光储备池、原始孪生和改进孪生测试结果。",
        "- `tables/selected_test_prediction_comparison.csv`：47 个测试月份逐月预测。",
        "- `figures/gap_validation_comparison.png`：gap 验证误差图。",
        "- `figures/selected_gap_test_predictions.png`：三种模型测试预测曲线。",
        "- `experiment_manifest.json`：数据边界、最终参数和防泄漏审计。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_PROCESSED_DIR,
    recent_state_dir: Path = RECENT_STATE_DIR,
    ordinary_experiment_dir: Path = ORDINARY_EXPERIMENT_DIR,
    gaps: Iterable[int] = GAPS,
    alphas: Iterable[float] = EXTENDED_ALPHAS,
) -> pd.DataFrame:
    """Run validation gap selection and one selected-gap test evaluation."""
    gaps = tuple(int(gap) for gap in gaps)
    train_pool = build_closed_train_pool(data_dir)
    val_split = _load_split(data_dir, "val")
    validation_states = load_validation_state_lookup(recent_state_dir)
    train_pool_ids = set(train_pool["index"]["sample_id"].astype(int))

    gap_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    validation_predictions: list[pd.DataFrame] = []
    cache: dict[int, dict[str, object]] = {}

    for gap in gaps:
        train_pairs, train_candidates, thresholds, calibration = (
            build_gap_training_pairs(train_pool, gap)
        )
        val_pairs, val_candidates = build_gap_evaluation_pairs(
            train_pool,
            val_split,
            gap,
            calibration,
        )
        bundle, trials = train_readout(
            {"train": train_pairs, "val": val_pairs},
            validation_states,
            feature_mode=FEATURE_MODE,
            aggregation=AGGREGATION,
            alphas=alphas,
        )
        _, val_targets = predict_pairs(bundle, val_pairs, validation_states)
        metrics = _target_metrics(val_targets)
        feature_dimension = build_pair_features(
            train_pairs.iloc[:1],
            validation_states,
            FEATURE_MODE,
        ).shape[1]
        gap_rows.append(
            {
                "gap_months": gap,
                "accessible_train_windows": len(train_pool_ids),
                "legal_train_candidates": len(train_candidates),
                "legal_train_pair_targets": train_candidates[
                    "sample_i_id"
                ].nunique(),
                "selected_train_pairs": len(train_pairs),
                "train_pair_targets": train_pairs["sample_i_id"].nunique(),
                "validation_pairs": len(val_pairs),
                "validation_targets": val_pairs["sample_i_id"].nunique(),
                "feature_dimension": feature_dimension,
                "selected_alpha": float(bundle.model.alpha),
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
            }
        )
        alpha_rows.extend({"gap_months": gap, **trial} for trial in trials)
        val_output = val_targets.copy()
        val_output.insert(0, "gap_months", gap)
        validation_predictions.append(val_output)
        cache[gap] = {
            "bundle": bundle,
            "train_pairs": train_pairs,
            "train_candidates": train_candidates,
            "val_pairs": val_pairs,
            "val_candidates": val_candidates,
            "thresholds": thresholds,
            "calibration": calibration,
        }

    gap_results = pd.DataFrame(gap_rows).sort_values("gap_months").reset_index(drop=True)
    selected_row = min(
        gap_rows,
        key=lambda row: (
            float(row["val_rmse"]),
            float(row["val_mae"]),
            -int(row["gap_months"]),
        ),
    )
    selected_gap = int(selected_row["gap_months"])
    selected = cache[selected_gap]

    # The test split is loaded only after validation has frozen the selected gap.
    test_split = _load_split(data_dir, "test")
    test_pairs, test_candidates = build_gap_evaluation_pairs(
        train_pool,
        test_split,
        selected_gap,
        selected["calibration"],
    )
    state_cache_check = verify_recent_state_cache(BASE_STATE_DIR, recent_state_dir)
    all_states = load_state_lookup(recent_state_dir)
    pair_outputs: dict[str, pd.DataFrame] = {}
    target_outputs: dict[str, pd.DataFrame] = {}
    for split, pairs in (
        ("train", selected["train_pairs"]),
        ("val", selected["val_pairs"]),
        ("test", test_pairs),
    ):
        pair_outputs[split], target_outputs[split] = predict_pairs(
            selected["bundle"],
            pairs,
            all_states,
        )

    selected_metrics_rows = []
    for split in ("train", "val", "test"):
        metrics = _target_metrics(target_outputs[split])
        selected_metrics_rows.append(
            {
                "split": split,
                "num_pairs": len(pair_outputs[split]),
                "num_targets": len(target_outputs[split]),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        )
    selected_metrics = pd.DataFrame(selected_metrics_rows)

    ordinary_metrics = pd.read_csv(
        ordinary_experiment_dir / "tables" / "optical_reservoir_metrics.csv"
    ).set_index("split")
    original_comparison = pd.read_csv(
        RESULTS_DIR
        / "siamese_optical_closed50_20260723"
        / "tables"
        / "closed50_model_comparison.csv"
    )
    original_siamese = original_comparison.loc[
        original_comparison["model"] == "siamese_optical_reservoir_closed50"
    ].iloc[0]
    improved_val = selected_metrics.set_index("split").loc["val"]
    improved_test = selected_metrics.set_index("split").loc["test"]
    model_comparison = pd.DataFrame(
        [
            {
                "model": "ordinary_optical_reservoir_closed50",
                "gap_months": np.nan,
                "feature_mode": "direct_state_50d",
                "val_mae": float(ordinary_metrics.loc["val", "mae"]),
                "val_rmse": float(ordinary_metrics.loc["val", "rmse"]),
                "test_mae": float(ordinary_metrics.loc["test", "mae"]),
                "test_rmse": float(ordinary_metrics.loc["test", "rmse"]),
            },
            {
                "model": "original_siamese_closed50_gap12",
                "gap_months": 12,
                "feature_mode": "h_i_minus_h_j_50d",
                "val_mae": float(original_siamese["val_mae"]),
                "val_rmse": float(original_siamese["val_rmse"]),
                "test_mae": float(original_siamese["test_mae"]),
                "test_rmse": float(original_siamese["test_rmse"]),
            },
            {
                "model": "hybrid_100d_siamese_closed50",
                "gap_months": selected_gap,
                "feature_mode": "h_i_and_h_i_minus_h_j_100d",
                "val_mae": float(improved_val["mae"]),
                "val_rmse": float(improved_val["rmse"]),
                "test_mae": float(improved_test["mae"]),
                "test_rmse": float(improved_test["rmse"]),
            },
        ]
    )

    ordinary_predictions = pd.read_csv(
        ordinary_experiment_dir
        / "tables"
        / "optical_reservoir_predictions_test.csv"
    )[["sample_id", "target_date", "cpi_actual", "cpi_predicted"]].rename(
        columns={
            "sample_id": "sample_i_id",
            "cpi_predicted": "cpi_predicted_ordinary",
        }
    )
    original_predictions = pd.read_csv(
        RESULTS_DIR
        / "siamese_optical_closed50_20260723"
        / "tables"
        / "closed50_predictions_test.csv"
    )[["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]].rename(
        columns={"cpi_predicted": "cpi_predicted_original_siamese"}
    )
    improved_predictions = target_outputs["test"][
        ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(columns={"cpi_predicted": "cpi_predicted_improved_siamese"})
    test_comparison = ordinary_predictions.merge(
        original_predictions,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    ).merge(
        improved_predictions,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    )
    if len(test_comparison) != 47:
        raise ValueError(f"expected 47 test targets, found {len(test_comparison)}")
    for suffix in ("ordinary", "original_siamese", "improved_siamese"):
        test_comparison[f"residual_{suffix}"] = (
            test_comparison[f"cpi_predicted_{suffix}"]
            - test_comparison["cpi_actual"]
        )
        test_comparison[f"absolute_error_{suffix}"] = test_comparison[
            f"residual_{suffix}"
        ].abs()

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    gap_results.to_csv(table_dir / "gap_validation_comparison.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(
        table_dir / "gap_alpha_selection.csv", index=False
    )
    pd.concat(validation_predictions, ignore_index=True).to_csv(
        table_dir / "gap_validation_predictions.csv", index=False
    )
    selected_metrics.to_csv(table_dir / "selected_model_metrics.csv", index=False)
    model_comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    test_comparison.to_csv(
        table_dir / "selected_test_prediction_comparison.csv", index=False
    )
    selected["train_candidates"].to_csv(
        table_dir / "selected_all_legal_train_candidates.csv", index=False
    )
    selected["train_pairs"].to_csv(
        table_dir / "selected_train_pairs.csv", index=False
    )
    selected["val_candidates"].to_csv(
        table_dir / "selected_validation_candidates.csv", index=False
    )
    selected["val_pairs"].to_csv(
        table_dir / "selected_validation_pairs.csv", index=False
    )
    test_candidates.to_csv(table_dir / "selected_test_candidates.csv", index=False)
    test_pairs.to_csv(table_dir / "selected_test_pairs.csv", index=False)
    for split in ("train", "val", "test"):
        pair_outputs[split].to_csv(
            table_dir / f"selected_pair_predictions_{split}.csv", index=False
        )
        target_outputs[split].to_csv(
            table_dir / f"selected_predictions_{split}.csv", index=False
        )
    save_model(
        selected["bundle"],
        model_dir / "siamese_closed50_gap_hybrid_readout.npz",
    )
    _save_figures(gap_results, test_comparison, figure_dir)

    calibration = selected["calibration"]
    manifest = {
        "experiment": "strict closed50 gap and hybrid-reference ablation",
        "evaluation_status": "exploratory re-analysis; test was seen before",
        "selection_protocol": (
            "compare all gaps on validation RMSE, freeze best gap, then run test once"
        ),
        "candidate_gaps_months": list(gaps),
        "selected_gap_months": selected_gap,
        "gap_definition": "x_i_start_month - x_j_end_month",
        "gap_1_interpretation": "adjacent non-overlapping 12-month input windows",
        "accessible_train_windows": len(train_pool_ids),
        "accessible_sample_id_min": min(train_pool_ids),
        "accessible_sample_id_max": max(train_pool_ids),
        "all_reference_ids_inside_closed50": bool(
            set(selected["train_pairs"]["sample_j_id"].astype(int))
            .union(selected["val_pairs"]["sample_j_id"].astype(int))
            .union(test_pairs["sample_j_id"].astype(int))
            .issubset(train_pool_ids)
        ),
        "older_train_references_used": False,
        "validation_or_test_labels_used_for_reference_selection": False,
        "reference_selection_sort_columns": [
            "sample_i_id",
            "hybrid_distance",
            "target_j_date",
            "sample_j_id",
        ],
        "hybrid_distance": {
            "shape_definition": "RMS Euclidean distance after per-window z-score",
            "level_definition": (
                "RMS of standardized window-mean difference and latest-value difference"
            ),
            "normalization": "training-candidate median for each component",
            **asdict(calibration),
        },
        "k_validation_test": K_REFERENCES,
        "feature_mode": FEATURE_MODE,
        "feature_formula": "[h_i, h_i - h_j]",
        "feature_dimension": int(selected_row["feature_dimension"]),
        "aggregation": AGGREGATION,
        "selected_alpha_on_validation": float(selected["bundle"].model.alpha),
        "selected_training_pairs": len(selected["train_pairs"]),
        "selected_training_pair_targets": int(
            selected["train_pairs"]["sample_i_id"].nunique()
        ),
        "state_cache_check": state_cache_check,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        output_dir,
        gap_results,
        model_comparison,
        selected_gap,
        calibration,
    )
    return model_comparison


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
