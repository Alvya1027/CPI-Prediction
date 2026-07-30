"""SSA optimization of Siamese-only parameters in the strict closed-50 setup.

The optical-reservoir state cache is frozen. Sparrow Search Algorithm (SSA)
optimizes only parameters that do not exist in the ordinary single-reservoir
regressor:

    [minimum gap, reference count K, level weight beta,
     distance exponent p, maximum training pairs per delta bin M]

All search decisions use validation data only. The test split is loaded only
after the best validation configuration has been frozen.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
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

from src.config import DATA_PROCESSED_DIR, RESULTS_DIR
from src.create_siamese_pairs import (
    _assign_delta_bins,
    _build_candidate_pairs,
    _compute_delta_bin_thresholds,
    _sample_pairs_by_bin,
)
from src.siamese_closed50_experiment import _load_split, build_closed_train_pool
from src.siamese_closed50_gap_hybrid_experiment import (
    GAPS,
    HybridDistanceCalibration,
    _markdown_table,
    _window_lookup,
    add_hybrid_distances,
    fit_hybrid_distance_calibration,
)
from src.siamese_recent50_experiment import (
    BASE_STATE_DIR,
    ORDINARY_EXPERIMENT_DIR,
    RECENT_STATE_DIR,
    verify_recent_state_cache,
)
from src.siamese_reservoir_regression import (
    DEFAULT_ALPHAS,
    ReadoutBundle,
    build_pair_features,
    load_state_lookup,
    regression_metrics,
    save_model,
)
from src.siamese_validation_search import load_validation_state_lookup


OUTPUT_DIR = RESULTS_DIR / "siamese_optical_closed50_ssa_20260726"
BASELINE_RESULTS_DIR = RESULTS_DIR / "siamese_optical_closed50_20260723"
SEARCH_DIMENSION = 5
LOWER_BOUNDS = np.asarray([0.0, 1.0, 0.0, 0.5, 1.0])
UPPER_BOUNDS = np.asarray([3.0, 15.0, 0.4, 3.0, 5.0])
BASELINE_POSITION = np.asarray([3.0, 10.0, 0.0, 1.0, 2.0])
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_POPULATION = 12
DEFAULT_ITERATIONS = 12
STABILITY_PENALTY = 0.10


@dataclass(frozen=True)
class SiameseSearchConfig:
    gap_months: int
    k_references: int
    level_weight: float
    distance_power: float
    max_pairs_per_bin: int

    @property
    def shape_weight(self) -> float:
        return 1.0 - self.level_weight


@dataclass
class GapCandidatePool:
    train_candidates: pd.DataFrame
    validation_candidates: pd.DataFrame
    thresholds: tuple[float, float]
    calibration: HybridDistanceCalibration


def decode_position(position: np.ndarray) -> SiameseSearchConfig:
    """Convert a bounded continuous SSA position into mixed model parameters."""
    values = np.clip(np.asarray(position, dtype=float), LOWER_BOUNDS, UPPER_BOUNDS)
    gap_index = int(np.clip(np.rint(values[0]), 0, len(GAPS) - 1))
    return SiameseSearchConfig(
        gap_months=int(GAPS[gap_index]),
        k_references=int(np.clip(np.rint(values[1]), 1, 15)),
        level_weight=float(np.round(values[2], 3)),
        distance_power=float(np.round(values[3], 3)),
        max_pairs_per_bin=int(np.clip(np.rint(values[4]), 1, 5)),
    )


def encode_config(config: SiameseSearchConfig) -> np.ndarray:
    gap_index = GAPS.index(int(config.gap_months))
    return np.asarray(
        [
            gap_index,
            config.k_references,
            config.level_weight,
            config.distance_power,
            config.max_pairs_per_bin,
        ],
        dtype=float,
    )


def _config_key(config: SiameseSearchConfig) -> tuple[object, ...]:
    return (
        config.gap_months,
        config.k_references,
        config.level_weight,
        config.distance_power,
        config.max_pairs_per_bin,
    )


def _with_search_distance(
    candidates: pd.DataFrame,
    level_weight: float,
) -> pd.DataFrame:
    result = candidates.copy()
    result["search_distance"] = (
        (1.0 - level_weight)
        * result["shape_distance_normalized"].to_numpy(dtype=float)
        + level_weight
        * result["level_distance_normalized"].to_numpy(dtype=float)
    )
    return result


def select_search_references(
    candidates: pd.DataFrame,
    k: int,
    level_weight: float,
) -> pd.DataFrame:
    """Select K references using input-only shape/level distance."""
    ranked = _with_search_distance(candidates, level_weight).sort_values(
        ["sample_i_id", "search_distance", "target_j_date", "sample_j_id"],
        ascending=[True, True, True, True],
    )
    selected = (
        ranked.groupby("sample_i_id", as_index=False, group_keys=False)
        .head(int(k))
        .copy()
        .reset_index(drop=True)
    )
    sizes = selected.groupby("sample_i_id").size()
    if len(sizes) != candidates["sample_i_id"].nunique():
        raise ValueError("SSA reference selection dropped a target")
    if not (sizes == int(k)).all():
        raise ValueError(f"at least one target has fewer than K={k} references")
    selected["selection_method"] = (
        f"ssa_shape_level_beta{level_weight:.3f}_k{int(k)}"
    )
    return selected


def aggregate_power_predictions(
    pair_predictions: pd.DataFrame,
    distance_power: float,
) -> pd.DataFrame:
    """Aggregate pair estimates with 1/(distance+eps)^p weights."""
    rows: list[dict[str, object]] = []
    for sample_i_id, group in pair_predictions.groupby("sample_i_id", sort=False):
        actual = group["cpi_i"].to_numpy(dtype=float)
        if not np.allclose(actual, actual[0]):
            raise ValueError(f"inconsistent cpi_i for sample {sample_i_id}")
        distances = group["search_distance"].to_numpy(dtype=float)
        estimates = group["cpi_pred_pair"].to_numpy(dtype=float)
        weights = 1.0 / np.maximum(distances, 1e-6) ** float(distance_power)
        rows.append(
            {
                "sample_i_id": int(sample_i_id),
                "target_date": str(group["target_i_date"].iloc[0]),
                "cpi_actual": float(actual[0]),
                "cpi_predicted": float(np.average(estimates, weights=weights)),
                "num_references": int(len(group)),
                "reference_prediction_std": float(np.std(estimates)),
                "mean_search_distance": float(np.mean(distances)),
            }
        )
    result = pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)
    result["error"] = result["cpi_predicted"] - result["cpi_actual"]
    result["absolute_error"] = result["error"].abs()
    return result


def predict_with_power_distance(
    scaler: StandardScaler,
    model: Ridge,
    pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
    distance_power: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = build_pair_features(pairs, state_lookup, "signed_diff")
    delta_prediction = model.predict(scaler.transform(features))
    pair_output = pairs.copy()
    pair_output["delta_cpi_predicted"] = delta_prediction
    pair_output["cpi_pred_pair"] = (
        pair_output["cpi_j"].to_numpy(dtype=float) + delta_prediction
    )
    pair_output["delta_error"] = (
        pair_output["delta_cpi_predicted"] - pair_output["delta_cpi"]
    )
    targets = aggregate_power_predictions(pair_output, distance_power)
    return pair_output, targets


def _validation_block_rmse(predictions: pd.DataFrame) -> list[float]:
    ordered = predictions.sort_values("target_date").reset_index(drop=True)
    blocks = np.array_split(np.arange(len(ordered)), 3)
    values = []
    for block in blocks:
        metrics = regression_metrics(
            ordered.loc[block, "cpi_actual"],
            ordered.loc[block, "cpi_predicted"],
        )
        values.append(metrics["rmse"])
    return values


class CandidateEvaluator:
    """Validation-only evaluator with deterministic pair/feature caches."""

    def __init__(
        self,
        train_pool: dict[str, object],
        validation_split: dict[str, object],
        state_lookup: dict[int, np.ndarray],
        alphas: Iterable[float] = DEFAULT_ALPHAS,
    ) -> None:
        self.train_pool = train_pool
        self.validation_split = validation_split
        self.state_lookup = state_lookup
        self.alphas = tuple(float(value) for value in alphas)
        self.gap_pools = self._prepare_gap_pools()
        self.train_cache: dict[tuple[int, int], tuple[pd.DataFrame, np.ndarray]] = {}
        self.validation_cache: dict[
            tuple[int, int, float], tuple[pd.DataFrame, np.ndarray]
        ] = {}
        self.results: dict[tuple[object, ...], dict[str, object]] = {}
        self.evaluation_rows: list[dict[str, object]] = []

    def _prepare_gap_pools(self) -> dict[int, GapCandidatePool]:
        pools: dict[int, GapCandidatePool] = {}
        train_windows = _window_lookup(self.train_pool)
        validation_windows = _window_lookup(
            self.validation_split,
            self.train_pool,
        )
        for gap in GAPS:
            train_raw = _build_candidate_pairs(
                self.train_pool,
                self.train_pool,
                min_gap_months=gap,
            )
            calibration = fit_hybrid_distance_calibration(
                self.train_pool,
                train_raw,
            )
            train_candidates = add_hybrid_distances(
                train_raw,
                train_windows,
                calibration,
            )
            validation_raw = _build_candidate_pairs(
                self.validation_split,
                self.train_pool,
                min_gap_months=gap,
            )
            validation_candidates = add_hybrid_distances(
                validation_raw,
                validation_windows,
                calibration,
            )
            pools[gap] = GapCandidatePool(
                train_candidates=train_candidates,
                validation_candidates=validation_candidates,
                thresholds=_compute_delta_bin_thresholds(train_candidates),
                calibration=calibration,
            )
        return pools

    def _training_data(
        self,
        config: SiameseSearchConfig,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        key = (config.gap_months, config.max_pairs_per_bin)
        if key not in self.train_cache:
            pool = self.gap_pools[config.gap_months]
            binned = _assign_delta_bins(pool.train_candidates, pool.thresholds)
            selected = _sample_pairs_by_bin(
                binned,
                max_per_bin=config.max_pairs_per_bin,
            ).reset_index(drop=True)
            selected["selection_method"] = (
                f"ssa_delta_stratified_gap{config.gap_months}"
                f"_max{config.max_pairs_per_bin}"
            )
            features = build_pair_features(
                selected,
                self.state_lookup,
                "signed_diff",
            )
            self.train_cache[key] = (selected, features)
        pairs, features = self.train_cache[key]
        return _with_search_distance(pairs, config.level_weight), features

    def _validation_data(
        self,
        config: SiameseSearchConfig,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        key = (
            config.gap_months,
            config.k_references,
            config.level_weight,
        )
        if key not in self.validation_cache:
            candidates = self.gap_pools[
                config.gap_months
            ].validation_candidates
            selected = select_search_references(
                candidates,
                config.k_references,
                config.level_weight,
            )
            features = build_pair_features(
                selected,
                self.state_lookup,
                "signed_diff",
            )
            self.validation_cache[key] = (selected, features)
        return self.validation_cache[key]

    def evaluate(
        self,
        config_or_position: SiameseSearchConfig | np.ndarray,
    ) -> dict[str, object]:
        config = (
            config_or_position
            if isinstance(config_or_position, SiameseSearchConfig)
            else decode_position(config_or_position)
        )
        key = _config_key(config)
        if key in self.results:
            return self.results[key]

        train_pairs, train_features = self._training_data(config)
        val_pairs, val_features = self._validation_data(config)
        train_delta = train_pairs["delta_cpi"].to_numpy(dtype=float)
        scaler = StandardScaler().fit(train_features)
        train_scaled = scaler.transform(train_features)
        val_scaled = scaler.transform(val_features)
        trials: list[dict[str, float]] = []
        fitted_models: dict[float, Ridge] = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            for alpha in self.alphas:
                model = Ridge(alpha=alpha).fit(train_scaled, train_delta)
                pair_output = val_pairs.copy()
                delta_prediction = model.predict(val_scaled)
                pair_output["delta_cpi_predicted"] = delta_prediction
                pair_output["cpi_pred_pair"] = (
                    pair_output["cpi_j"].to_numpy(dtype=float)
                    + delta_prediction
                )
                predictions = aggregate_power_predictions(
                    pair_output,
                    config.distance_power,
                )
                metrics = regression_metrics(
                    predictions["cpi_actual"],
                    predictions["cpi_predicted"],
                )
                trials.append({"alpha": alpha, **metrics})
                fitted_models[alpha] = model

        best_trial = min(
            trials,
            key=lambda row: (row["rmse"], row["mae"], row["alpha"]),
        )
        best_alpha = float(best_trial["alpha"])
        model = fitted_models[best_alpha]
        pair_predictions, predictions = predict_with_power_distance(
            scaler,
            model,
            val_pairs,
            self.state_lookup,
            config.distance_power,
        )
        block_rmse = _validation_block_rmse(predictions)
        block_std = float(np.std(block_rmse))
        fitness = float(best_trial["rmse"] + STABILITY_PENALTY * block_std)
        result: dict[str, object] = {
            "config": config,
            "fitness": fitness,
            "val_mae": float(best_trial["mae"]),
            "val_rmse": float(best_trial["rmse"]),
            "validation_block_rmse_1": block_rmse[0],
            "validation_block_rmse_2": block_rmse[1],
            "validation_block_rmse_3": block_rmse[2],
            "validation_block_rmse_std": block_std,
            "selected_alpha": best_alpha,
            "num_train_pairs": len(train_pairs),
            "num_train_pair_targets": train_pairs["sample_i_id"].nunique(),
            "num_validation_pairs": len(val_pairs),
            "scaler": scaler,
            "model": model,
            "train_pairs": train_pairs,
            "validation_pairs": val_pairs,
            "validation_pair_predictions": pair_predictions,
            "validation_predictions": predictions,
            "alpha_trials": trials,
        }
        self.results[key] = result
        self.evaluation_rows.append(
            {
                "evaluation_index": len(self.evaluation_rows),
                **asdict(config),
                "shape_weight": config.shape_weight,
                "fitness": fitness,
                "val_mae": best_trial["mae"],
                "val_rmse": best_trial["rmse"],
                "validation_block_rmse_std": block_std,
                "selected_alpha": best_alpha,
                "num_train_pairs": len(train_pairs),
                "num_train_pair_targets": train_pairs["sample_i_id"].nunique(),
            }
        )
        return result


def sparrow_search(
    objective,
    seed: int,
    population_size: int = DEFAULT_POPULATION,
    iterations: int = DEFAULT_ITERATIONS,
    producer_ratio: float = 0.20,
    awareness_ratio: float = 0.20,
    safety_threshold: float = 0.80,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    """Minimize a bounded black-box objective using the original SSA roles."""
    if population_size < 4 or iterations <= 0:
        raise ValueError("SSA needs population_size >= 4 and iterations > 0")
    rng = np.random.default_rng(seed)
    positions = rng.uniform(
        LOWER_BOUNDS,
        UPPER_BOUNDS,
        size=(population_size, SEARCH_DIMENSION),
    )
    positions[0] = BASELINE_POSITION
    fitness = np.asarray([float(objective(position)) for position in positions])
    best_index = int(np.argmin(fitness))
    global_best = positions[best_index].copy()
    global_best_fitness = float(fitness[best_index])
    history = [
        {
            "seed": float(seed),
            "iteration": 0.0,
            "best_fitness": global_best_fitness,
            "population_mean_fitness": float(np.mean(fitness)),
        }
    ]

    for iteration in range(1, iterations + 1):
        order = np.argsort(fitness)
        positions = positions[order].copy()
        fitness = fitness[order].copy()
        best = positions[0].copy()
        worst = positions[-1].copy()
        best_fitness = float(fitness[0])
        worst_fitness = float(fitness[-1])
        producer_count = max(1, int(np.ceil(population_size * producer_ratio)))

        updated = positions.copy()
        alarm = rng.random()
        for rank in range(producer_count):
            if alarm < safety_threshold:
                decay = np.exp(
                    -(rank + 1)
                    / (max(rng.random(), 1e-8) * max(iterations, 1))
                )
                updated[rank] = positions[rank] * decay
            else:
                updated[rank] = positions[rank] + rng.normal(
                    size=SEARCH_DIMENSION
                )

        for rank in range(producer_count, population_size):
            if rank >= population_size // 2:
                exponent = np.clip(
                    (worst - positions[rank]) / float((rank + 1) ** 2),
                    -20.0,
                    20.0,
                )
                updated[rank] = rng.normal() * np.exp(exponent)
            else:
                signs = rng.choice((-1.0, 1.0), size=SEARCH_DIMENSION)
                projection = float(
                    np.dot(np.abs(positions[rank] - best), signs)
                    / np.dot(signs, signs)
                )
                updated[rank] = best + projection * signs

        watcher_count = max(1, int(np.ceil(population_size * awareness_ratio)))
        watcher_indices = rng.choice(
            population_size,
            size=watcher_count,
            replace=False,
        )
        for index in watcher_indices:
            if fitness[index] > best_fitness:
                updated[index] = best + rng.normal() * np.abs(
                    positions[index] - best
                )
            else:
                denominator = (
                    fitness[index] - worst_fitness
                    + np.sign(fitness[index] - worst_fitness) * 1e-12
                )
                if abs(denominator) < 1e-12:
                    denominator = -1e-12
                updated[index] = positions[index] + rng.uniform(-1.0, 1.0) * (
                    np.abs(positions[index] - worst) / denominator
                )

        updated = np.clip(updated, LOWER_BOUNDS, UPPER_BOUNDS)
        updated_fitness = np.asarray(
            [float(objective(position)) for position in updated]
        )
        worst_new = int(np.argmax(updated_fitness))
        if global_best_fitness < updated_fitness[worst_new]:
            updated[worst_new] = global_best
            updated_fitness[worst_new] = global_best_fitness
        positions = updated
        fitness = updated_fitness

        current_best = int(np.argmin(fitness))
        if fitness[current_best] < global_best_fitness:
            global_best = positions[current_best].copy()
            global_best_fitness = float(fitness[current_best])
        history.append(
            {
                "seed": float(seed),
                "iteration": float(iteration),
                "best_fitness": global_best_fitness,
                "population_mean_fitness": float(np.mean(fitness)),
            }
        )

    return global_best, global_best_fitness, history


def _build_test_pairs(
    selected: SiameseSearchConfig,
    evaluator: CandidateEvaluator,
    train_pool: dict[str, object],
    test_split: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _build_candidate_pairs(
        test_split,
        train_pool,
        min_gap_months=selected.gap_months,
    )
    calibration = evaluator.gap_pools[selected.gap_months].calibration
    candidates = add_hybrid_distances(
        raw,
        _window_lookup(test_split, train_pool),
        calibration,
    )
    pairs = select_search_references(
        candidates,
        selected.k_references,
        selected.level_weight,
    )
    return pairs, candidates


def _save_figures(
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for seed, group in history.groupby("seed"):
        axis.plot(
            group["iteration"],
            group["best_fitness"],
            marker="o",
            markersize=3,
            label=f"seed={int(seed)}",
        )
    axis.set_xlabel("SSA iteration")
    axis.set_ylabel("Best validation fitness")
    axis.set_title("SSA convergence on Siamese-only parameters")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "ssa_validation_convergence.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(predictions["target_date"], predictions["cpi_actual"], label="Actual CPI")
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_ordinary"],
        label="Ordinary reservoir (closed 50)",
    )
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_siamese_baseline"],
        label="Siamese baseline",
    )
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_ssa"],
        label="SSA-optimized Siamese",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI")
    axis.set_title("Strict closed-50 SSA test prediction comparison")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "ssa_test_prediction_comparison.png", dpi=180)
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    selected: SiameseSearchConfig,
    run_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    unique_evaluations: int,
) -> None:
    baseline = comparison.loc[
        comparison["model"] == "siamese_closed50_baseline"
    ].iloc[0]
    optimized = comparison.loc[
        comparison["model"] == "ssa_optimized_siamese_closed50"
    ].iloc[0]
    val_change = 100.0 * (
        float(baseline["val_rmse"]) - float(optimized["val_rmse"])
    ) / float(baseline["val_rmse"])
    test_change = 100.0 * (
        float(baseline["test_rmse"]) - float(optimized["test_rmse"])
    ) / float(baseline["test_rmse"])
    ordinary = comparison.loc[
        comparison["model"] == "ordinary_optical_reservoir_closed50"
    ].iloc[0]
    versus_ordinary = 100.0 * (
        float(ordinary["test_rmse"]) - float(optimized["test_rmse"])
    ) / float(ordinary["test_rmse"])
    conclusion = (
        "SSA优化后测试RMSE低于孪生基线，取得改善。"
        if test_change > 0
        else "SSA优化后测试RMSE没有低于孪生基线，不能宣称取得改善。"
    )
    lines = [
        "# SSA优化严格封闭50窗口孪生光储备池",
        "",
        "## 优化边界",
        "",
        "- 光储备池、50维状态缓存、mask、输入增益、虚拟节点和延迟参数全部冻结。",
        "- SSA只优化孪生模型独有参数：`gap、K、β、p、M`。",
        "- 特征固定为50维`h_i-h_j`，没有使用此前表现较差的100维特征。",
        "- 每个SSA种群都加入原始孪生基线，防止搜索结果无意中弱于已知起点。",
        "- SSA搜索只读取训练池和验证集；最佳配置冻结后才加载测试集。",
        "",
        "## 参数定义",
        "",
        "- `gap`：目标窗口与参考窗口的最小时间间隔。",
        "- `K`：每个目标使用的历史参考数量。",
        "- `β`：绝对水平距离权重；形状距离权重为`1-β`。",
        "- `p`：参考聚合权重`1/(distance+ε)^p`中的指数。",
        "- `M`：每个delta区间最多选择的训练样本对数量。",
        "",
        "## SSA设置与每次运行结果",
        "",
        _markdown_table(run_summary),
        "",
        f"共实际评估了 **{unique_evaluations}** 组不重复参数。",
        "",
        "## 验证集冻结的最佳参数",
        "",
        "```text",
        f"gap = {selected.gap_months}",
        f"K = {selected.k_references}",
        f"β = {selected.level_weight:.3f}",
        f"形状权重 = {selected.shape_weight:.3f}",
        f"p = {selected.distance_power:.3f}",
        f"M = {selected.max_pairs_per_bin}",
        "```",
        "",
        "## 最终比较",
        "",
        _markdown_table(comparison),
        "",
        f"相对原始孪生基线，验证RMSE下降 **{val_change:.2f}%**。",
        (
            f"测试RMSE下降 **{test_change:.2f}%**。"
            if test_change >= 0
            else f"测试RMSE上升 **{abs(test_change):.2f}%**。"
        ),
        f"相对单光储备池，SSA孪生模型的测试RMSE仍下降 **{versus_ordinary:.2f}%**。",
        "",
        f"**结论：{conclusion}**",
        "",
        "当前测试区间此前已经被查看，因此仍属于探索性复分析。"
        "正式结论应冻结本次参数后，使用新的未见时间区间检验。",
        "",
        "关键文件：",
        "",
        "- `tables/ssa_all_unique_evaluations.csv`：所有不重复候选参数及验证结果。",
        "- `tables/ssa_iteration_history.csv`：不同随机种子的收敛过程。",
        "- `tables/ssa_model_comparison.csv`：单光、孪生基线和SSA孪生的对比。",
        "- `tables/ssa_test_prediction_comparison.csv`：47个测试月份逐月结果。",
        "- `figures/ssa_validation_convergence.png`：SSA收敛曲线。",
        "- `figures/ssa_test_prediction_comparison.png`：最终预测曲线。",
        "- `experiment_manifest.json`：搜索边界和防泄漏记录。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_ssa_optimization(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_PROCESSED_DIR,
    recent_state_dir: Path = RECENT_STATE_DIR,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    population_size: int = DEFAULT_POPULATION,
    iterations: int = DEFAULT_ITERATIONS,
) -> pd.DataFrame:
    """Run multi-seed SSA validation search and one frozen test evaluation."""
    seeds = tuple(int(seed) for seed in seeds)
    train_pool = build_closed_train_pool(data_dir)
    validation_split = _load_split(data_dir, "val")
    validation_states = load_validation_state_lookup(recent_state_dir)
    evaluator = CandidateEvaluator(
        train_pool,
        validation_split,
        validation_states,
    )

    baseline_config = decode_position(BASELINE_POSITION)
    baseline_result = evaluator.evaluate(baseline_config)
    original_comparison = pd.read_csv(
        BASELINE_RESULTS_DIR / "tables" / "closed50_model_comparison.csv"
    )
    original_siamese = original_comparison.loc[
        original_comparison["model"] == "siamese_optical_reservoir_closed50"
    ].iloc[0]
    if not np.isclose(
        baseline_result["val_rmse"],
        float(original_siamese["val_rmse"]),
        atol=1e-10,
    ):
        raise ValueError("SSA baseline does not reproduce the frozen Siamese baseline")

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
        best_config = decode_position(best_position)
        best_result = evaluator.evaluate(best_config)
        best_results.append(best_result)
        histories.append(pd.DataFrame(history))
        run_rows.append(
            {
                "seed": seed,
                **asdict(best_config),
                "shape_weight": best_config.shape_weight,
                "fitness": best_fitness,
                "val_mae": best_result["val_mae"],
                "val_rmse": best_result["val_rmse"],
                "validation_block_rmse_std": best_result[
                    "validation_block_rmse_std"
                ],
                "selected_alpha": best_result["selected_alpha"],
            }
        )

    selected_result = min(
        best_results,
        key=lambda row: (
            float(row["fitness"]),
            float(row["val_rmse"]),
            float(row["val_mae"]),
        ),
    )
    selected_config = selected_result["config"]

    # Test data is first loaded here, after every SSA run and final selection.
    test_split = _load_split(data_dir, "test")
    test_pairs, test_candidates = _build_test_pairs(
        selected_config,
        evaluator,
        train_pool,
        test_split,
    )
    state_cache_check = verify_recent_state_cache(
        BASE_STATE_DIR,
        recent_state_dir,
    )
    all_states = load_state_lookup(recent_state_dir)
    test_pair_predictions, test_predictions = predict_with_power_distance(
        selected_result["scaler"],
        selected_result["model"],
        test_pairs,
        all_states,
        selected_config.distance_power,
    )
    test_metrics = regression_metrics(
        test_predictions["cpi_actual"],
        test_predictions["cpi_predicted"],
    )

    ordinary_metrics = pd.read_csv(
        ORDINARY_EXPERIMENT_DIR / "tables" / "optical_reservoir_metrics.csv"
    ).set_index("split")
    comparison = pd.DataFrame(
        [
            {
                "model": "ordinary_optical_reservoir_closed50",
                "optimized_by_ssa": False,
                "val_mae": float(ordinary_metrics.loc["val", "mae"]),
                "val_rmse": float(ordinary_metrics.loc["val", "rmse"]),
                "test_mae": float(ordinary_metrics.loc["test", "mae"]),
                "test_rmse": float(ordinary_metrics.loc["test", "rmse"]),
            },
            {
                "model": "siamese_closed50_baseline",
                "optimized_by_ssa": False,
                "val_mae": float(original_siamese["val_mae"]),
                "val_rmse": float(original_siamese["val_rmse"]),
                "test_mae": float(original_siamese["test_mae"]),
                "test_rmse": float(original_siamese["test_rmse"]),
            },
            {
                "model": "ssa_optimized_siamese_closed50",
                "optimized_by_ssa": True,
                "val_mae": float(selected_result["val_mae"]),
                "val_rmse": float(selected_result["val_rmse"]),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            },
        ]
    )

    ordinary_predictions = pd.read_csv(
        ORDINARY_EXPERIMENT_DIR
        / "tables"
        / "optical_reservoir_predictions_test.csv"
    )[["sample_id", "target_date", "cpi_actual", "cpi_predicted"]].rename(
        columns={
            "sample_id": "sample_i_id",
            "cpi_predicted": "cpi_predicted_ordinary",
        }
    )
    baseline_predictions = pd.read_csv(
        BASELINE_RESULTS_DIR / "tables" / "closed50_predictions_test.csv"
    )[["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]].rename(
        columns={"cpi_predicted": "cpi_predicted_siamese_baseline"}
    )
    optimized_predictions = test_predictions[
        ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(columns={"cpi_predicted": "cpi_predicted_ssa"})
    unified = ordinary_predictions.merge(
        baseline_predictions,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    ).merge(
        optimized_predictions,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    )
    if len(unified) != 47:
        raise ValueError(f"SSA test comparison has {len(unified)} rows, expected 47")
    for suffix in ("ordinary", "siamese_baseline", "ssa"):
        unified[f"residual_{suffix}"] = (
            unified[f"cpi_predicted_{suffix}"] - unified["cpi_actual"]
        )
        unified[f"absolute_error_{suffix}"] = unified[
            f"residual_{suffix}"
        ].abs()

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluations = pd.DataFrame(evaluator.evaluation_rows).sort_values(
        ["fitness", "val_rmse", "val_mae"]
    )
    history_table = pd.concat(histories, ignore_index=True)
    run_summary = pd.DataFrame(run_rows)
    evaluations.to_csv(
        table_dir / "ssa_all_unique_evaluations.csv",
        index=False,
    )
    history_table.to_csv(
        table_dir / "ssa_iteration_history.csv",
        index=False,
    )
    run_summary.to_csv(table_dir / "ssa_run_summary.csv", index=False)
    comparison.to_csv(table_dir / "ssa_model_comparison.csv", index=False)
    unified.to_csv(
        table_dir / "ssa_test_prediction_comparison.csv",
        index=False,
    )
    selected_result["train_pairs"].to_csv(
        table_dir / "ssa_selected_train_pairs.csv",
        index=False,
    )
    selected_result["validation_pairs"].to_csv(
        table_dir / "ssa_selected_validation_pairs.csv",
        index=False,
    )
    selected_result["validation_pair_predictions"].to_csv(
        table_dir / "ssa_selected_validation_pair_predictions.csv",
        index=False,
    )
    selected_result["validation_predictions"].to_csv(
        table_dir / "ssa_selected_validation_predictions.csv",
        index=False,
    )
    test_candidates.to_csv(
        table_dir / "ssa_selected_test_candidates.csv",
        index=False,
    )
    test_pairs.to_csv(table_dir / "ssa_selected_test_pairs.csv", index=False)
    test_pair_predictions.to_csv(
        table_dir / "ssa_selected_test_pair_predictions.csv",
        index=False,
    )
    test_predictions.to_csv(
        table_dir / "ssa_selected_test_predictions.csv",
        index=False,
    )
    pd.DataFrame(selected_result["alpha_trials"]).to_csv(
        table_dir / "ssa_selected_alpha_trials.csv",
        index=False,
    )
    save_model(
        ReadoutBundle(
            scaler=selected_result["scaler"],
            model=selected_result["model"],
            feature_mode="signed_diff",
            aggregation="ssa_inverse_power_search_distance",
        ),
        model_dir / "ssa_optimized_siamese_readout.npz",
    )
    _save_figures(history_table, unified, figure_dir)

    train_pool_ids = set(train_pool["index"]["sample_id"].astype(int))
    manifest = {
        "experiment": "SSA optimization of Siamese-only closed50 parameters",
        "evaluation_status": "exploratory re-analysis; test was seen before",
        "frozen_optical_reservoir": True,
        "frozen_state_directory": str(recent_state_dir),
        "reservoir_parameters_changed": False,
        "feature_mode": "signed_diff_50d_fixed",
        "optimized_parameters": [
            "gap_months",
            "k_references",
            "level_weight",
            "distance_power",
            "max_pairs_per_bin",
        ],
        "search_bounds": {
            "gap_months": list(GAPS),
            "k_references": [1, 15],
            "level_weight": [0.0, 0.4],
            "distance_power": [0.5, 3.0],
            "max_pairs_per_bin": [1, 5],
        },
        "baseline_inserted_into_each_population": True,
        "baseline_configuration": asdict(baseline_config),
        "ssa": {
            "seeds": list(seeds),
            "population_size": population_size,
            "iterations": iterations,
            "producer_ratio": 0.20,
            "awareness_ratio": 0.20,
            "safety_threshold": 0.80,
            "unique_evaluations": len(evaluations),
        },
        "fitness": (
            "validation RMSE + 0.10 * std(RMSE over three chronological "
            "validation blocks)"
        ),
        "ridge_alpha_rule": (
            "same validation RMSE grid selection as frozen Siamese baseline; "
            "alpha is not an SSA parameter"
        ),
        "selected_configuration": asdict(selected_config),
        "selected_shape_weight": selected_config.shape_weight,
        "selected_alpha": selected_result["selected_alpha"],
        "accessible_train_windows": 50,
        "accessible_sample_id_min": min(train_pool_ids),
        "accessible_sample_id_max": max(train_pool_ids),
        "all_selected_references_inside_closed50": bool(
            set(selected_result["train_pairs"]["sample_j_id"].astype(int))
            .union(
                selected_result["validation_pairs"]["sample_j_id"].astype(int)
            )
            .union(test_pairs["sample_j_id"].astype(int))
            .issubset(train_pool_ids)
        ),
        "validation_or_test_labels_used_for_reference_selection": False,
        "test_loaded_after_ssa_search_and_configuration_freeze": True,
        "state_cache_check": state_cache_check,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        output_dir,
        selected_config,
        run_summary,
        comparison,
        len(evaluations),
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_ssa_optimization(
        output_dir=args.output_dir,
        seeds=args.seeds,
        population_size=args.population_size,
        iterations=args.iterations,
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
