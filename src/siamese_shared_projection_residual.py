"""Stabilized shared-projection Siamese residual experiment for MoM CPI.

This model keeps the already validation-frozen SSA Ridge readout as a fixed
base predictor.  A zero-initialized neural residual branch sends target and
reference optical states through one shared projection, predicts an
antisymmetric delta correction, and optionally learns a continuous similarity
geometry.  Epoch zero therefore reproduces the fixed Ridge base exactly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from src.siamese_closed50_ssa_optimization import aggregate_power_predictions
from src.siamese_mom_closed50_pipeline import audit_profile
from src.siamese_reservoir_regression import (
    build_pair_features,
    load_state_lookup,
    regression_metrics,
)
from src.siamese_shared_projection import (
    CURRENT_SIAMESE_DIR,
    DATA_DIR,
    NetworkSpec,
    ORDINARY_DIR,
    PROFILE_ROOT,
    STATE_DIR,
    SharedProjectionSiamese,
    _balanced_pair_weights,
    build_causal_train_pairs,
    _chronological_block_rmse_std,
    _ordinary_and_current_outputs,
    _set_deterministic_seed,
    _state_matrix,
)
from src.siamese_validation_search import load_validation_state_lookup
from src.siamese_split_isolation import build_isolated_closed_train_pool
from src.config import RESULTS_DIR


OUTPUT_DIR = RESULTS_DIR / "siamese_optical_mom_shared_projection_residual_20260802"
BASE_MODEL_PATH = CURRENT_SIAMESE_DIR / "models" / "ssa_siamese_readout.npz"
BASE_TRAIN_PAIRS_PATH = (
    CURRENT_SIAMESE_DIR / "tables" / "ssa_selected_train_pairs.csv"
)
BASE_VALIDATION_PAIRS_PATH = (
    CURRENT_SIAMESE_DIR / "tables" / "ssa_selected_validation_pairs.csv"
)
BASE_TEST_PAIRS_PATH = (
    CURRENT_SIAMESE_DIR / "tables" / "ssa_selected_test_pair_predictions.csv"
)

DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_SIMILARITY_WEIGHTS = (0.0, 0.02, 0.10)
DISTANCE_POWER = 0.654


@dataclass(frozen=True)
class ResidualSpec:
    """Capacity and optimization settings fixed before validation search."""

    projection_hidden_dim: int = 16
    embedding_dim: int = 8
    relation_hidden_dim: int = 8
    dropout: float = 0.0
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    max_epochs: int = 1200
    evaluation_interval: int = 5
    patience_evaluations: int = 80
    min_improvement: float = 1e-7

    def network_spec(self) -> NetworkSpec:
        return NetworkSpec(
            projection_hidden_dim=self.projection_hidden_dim,
            embedding_dim=self.embedding_dim,
            relation_hidden_dim=self.relation_hidden_dim,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            epochs=self.max_epochs,
            min_gap_months=1,
        )


@dataclass(frozen=True)
class FrozenRidgeBase:
    """Read-only parameters of the existing signed-difference Ridge model."""

    coefficient: np.ndarray
    intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    alpha: float


@dataclass
class ResidualBundle:
    """One seed's validation-frozen residual checkpoint."""

    model: "SharedProjectionResidual"
    state_scaler: StandardScaler
    base: FrozenRidgeBase
    correction_scale: float
    similarity_tau: float
    spec: ResidualSpec
    seed: int
    similarity_weight: float
    best_epoch: int
    validation_metrics: dict[str, float]
    training_history: pd.DataFrame


class SharedProjectionResidual(SharedProjectionSiamese):
    """Shared projection and antisymmetric relation head with zero correction."""

    def __init__(self, state_width: int, spec: NetworkSpec) -> None:
        super().__init__(state_width, spec)
        final_layer = self.relation[-1]
        torch.nn.init.zeros_(final_layer.weight)
        torch.nn.init.zeros_(final_layer.bias)


def load_frozen_ridge_base(path: Path = BASE_MODEL_PATH) -> FrozenRidgeBase:
    """Load and validate the existing SSA Ridge readout without fitting it."""
    with np.load(path, allow_pickle=False) as payload:
        feature_mode = str(payload["feature_mode"].reshape(-1)[0])
        if feature_mode != "signed_diff":
            raise ValueError(f"unexpected frozen base feature mode: {feature_mode}")
        coefficient = np.asarray(payload["coefficient"], dtype=np.float64)
        scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        if not (
            coefficient.shape == scaler_mean.shape == scaler_scale.shape == (50,)
        ):
            raise ValueError("frozen Ridge base does not use the expected 50 states")
        if np.any(scaler_scale <= 0):
            raise ValueError("frozen Ridge base has an invalid scaler")
        return FrozenRidgeBase(
            coefficient=coefficient.copy(),
            intercept=float(np.asarray(payload["intercept"]).reshape(-1)[0]),
            scaler_mean=scaler_mean.copy(),
            scaler_scale=scaler_scale.copy(),
            alpha=float(np.asarray(payload["alpha"]).reshape(-1)[0]),
        )


def frozen_base_delta(
    base: FrozenRidgeBase,
    pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
) -> np.ndarray:
    """Apply the immutable Ridge base to an ordered target-reference table."""
    features = np.asarray(
        build_pair_features(pairs, state_lookup, "signed_diff"),
        dtype=np.float64,
    )
    scaled = (features - base.scaler_mean) / base.scaler_scale
    return scaled @ base.coefficient + base.intercept


def _validate_pair_table(
    pairs: pd.DataFrame,
    name: str,
    expected_targets: int,
) -> pd.DataFrame:
    required = {
        "sample_i_id",
        "sample_j_id",
        "target_i_date",
        "target_j_date",
        "cpi_i",
        "cpi_j",
        "delta_cpi",
        "search_distance",
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"{name} pair table is missing {sorted(missing)}")
    output = pairs.copy().reset_index(drop=True)
    if output["sample_i_id"].nunique() != expected_targets:
        raise ValueError(
            f"{name} has {output['sample_i_id'].nunique()} targets, "
            f"expected {expected_targets}"
        )
    if output.duplicated(["sample_i_id", "sample_j_id"]).any():
        raise ValueError(f"{name} contains duplicate ordered pairs")
    if not np.allclose(
        output["cpi_i"].to_numpy(dtype=float)
        - output["cpi_j"].to_numpy(dtype=float),
        output["delta_cpi"].to_numpy(dtype=float),
    ):
        raise ValueError(f"{name} delta labels are inconsistent")
    return output


def load_pretest_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open only train and validation pair tables."""
    train_pairs = _validate_pair_table(
        pd.read_csv(BASE_TRAIN_PAIRS_PATH), "train", expected_targets=38
    )
    validation_pairs = _validate_pair_table(
        pd.read_csv(BASE_VALIDATION_PAIRS_PATH), "validation", expected_targets=45
    )
    return train_pairs, validation_pairs


def load_postfreeze_test_pairs() -> pd.DataFrame:
    """Open historical test pairs only after the new configuration freezes."""
    return _validate_pair_table(
        pd.read_csv(BASE_TEST_PAIRS_PATH), "test", expected_targets=47
    )


def _scaled_pair_states(
    pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    state_i = scaler.transform(
        _state_matrix(pairs["sample_i_id"].to_numpy(dtype=int), state_lookup)
    ).astype(np.float32)
    state_j = scaler.transform(
        _state_matrix(pairs["sample_j_id"].to_numpy(dtype=int), state_lookup)
    ).astype(np.float32)
    return state_i, state_j


def _predict_pairs(
    model: SharedProjectionResidual,
    state_scaler: StandardScaler,
    base: FrozenRidgeBase,
    correction_scale: float,
    pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_i, state_j = _scaled_pair_states(pairs, state_lookup, state_scaler)
    with torch.no_grad():
        scaled_correction, _, _ = model(
            torch.from_numpy(state_i),
            torch.from_numpy(state_j),
        )
    base_delta = frozen_base_delta(base, pairs, state_lookup)
    correction = scaled_correction.cpu().numpy() * float(correction_scale)
    pair_predictions = pairs.copy()
    pair_predictions["delta_cpi_base"] = base_delta
    pair_predictions["delta_cpi_neural_correction"] = correction
    pair_predictions["delta_cpi_predicted"] = base_delta + correction
    pair_predictions["cpi_pred_pair"] = (
        pair_predictions["cpi_j"].to_numpy(dtype=float)
        + pair_predictions["delta_cpi_predicted"].to_numpy(dtype=float)
    )
    pair_predictions["delta_error"] = (
        pair_predictions["delta_cpi_predicted"]
        - pair_predictions["delta_cpi"]
    )
    predictions = aggregate_power_predictions(pair_predictions, DISTANCE_POWER)
    return pair_predictions, predictions


def predict_bundle(
    bundle: ResidualBundle,
    pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one frozen seed checkpoint to a fixed reference table."""
    bundle.model.eval()
    return _predict_pairs(
        bundle.model,
        bundle.state_scaler,
        bundle.base,
        bundle.correction_scale,
        pairs,
        state_lookup,
    )


def _prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    metrics = regression_metrics(
        predictions["cpi_actual"], predictions["cpi_predicted"]
    )
    block_std = _chronological_block_rmse_std(predictions)
    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "chronological_block_rmse_std": block_std,
        "fitness": metrics["rmse"] + 0.10 * block_std,
    }


def fit_residual_bundle(
    train_pool: dict[str, object],
    train_pairs: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
    base: FrozenRidgeBase,
    spec: ResidualSpec,
    seed: int,
    similarity_weight: float,
) -> ResidualBundle:
    """Train one seed; gradients use train pairs and val only picks an epoch."""
    _set_deterministic_seed(seed)
    train_ids = train_pool["index"]["sample_id"].to_numpy(dtype=int)
    state_scaler = StandardScaler().fit(_state_matrix(train_ids, state_lookup))
    state_i, state_j = _scaled_pair_states(train_pairs, state_lookup, state_scaler)
    delta_actual = train_pairs["delta_cpi"].to_numpy(dtype=np.float32)
    delta_base = frozen_base_delta(base, train_pairs, state_lookup).astype(np.float32)
    train_residual = delta_actual - delta_base
    correction_scale = max(float(np.std(train_residual)), 1e-4)
    nonzero_delta = np.abs(delta_actual[np.abs(delta_actual) > 1e-8])
    similarity_tau = max(
        float(np.median(nonzero_delta)) if len(nonzero_delta) else 1.0,
        1e-3,
    )
    pair_weights = _balanced_pair_weights(train_pairs)

    tensor_i = torch.from_numpy(state_i)
    tensor_j = torch.from_numpy(state_j)
    tensor_target = torch.from_numpy(train_residual / correction_scale)
    tensor_similarity = torch.from_numpy(
        np.exp(-np.abs(delta_actual) / similarity_tau).astype(np.float32)
    )
    tensor_weights = torch.from_numpy(pair_weights)

    model = SharedProjectionResidual(50, spec.network_spec())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )

    def training_losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted, embedding_i, embedding_j = model(tensor_i, tensor_j)
        correction_each = F.smooth_l1_loss(
            predicted, tensor_target, reduction="none"
        )
        cosine_similarity = torch.sum(embedding_i * embedding_j, dim=1)
        similarity_each = torch.square(cosine_similarity - tensor_similarity)
        correction_loss = torch.mean(correction_each * tensor_weights)
        similarity_loss = torch.mean(similarity_each * tensor_weights)
        total = correction_loss + float(similarity_weight) * similarity_loss
        return total, correction_loss, similarity_loss

    model.eval()
    initial_pairs, initial_predictions = _predict_pairs(
        model,
        state_scaler,
        base,
        correction_scale,
        validation_pairs,
        state_lookup,
    )
    del initial_pairs
    best_metrics = _prediction_metrics(initial_predictions)
    best_key = (
        best_metrics["fitness"],
        best_metrics["rmse"],
        best_metrics["mae"],
        0,
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    rows: list[dict[str, float | int]] = [
        {
            "epoch": 0,
            "total_train_loss": math.nan,
            "correction_train_loss": math.nan,
            "similarity_train_loss": math.nan,
            "val_mae": best_metrics["mae"],
            "val_rmse": best_metrics["rmse"],
            "validation_block_rmse_std": best_metrics[
                "chronological_block_rmse_std"
            ],
            "fitness": best_metrics["fitness"],
            "is_new_best": 1,
        }
    ]
    evaluations_without_improvement = 0

    for epoch in range(1, spec.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, correction_loss, similarity_loss = training_losses()
        if not torch.isfinite(total_loss):
            raise FloatingPointError("residual training produced non-finite loss")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % spec.evaluation_interval != 0 and epoch != spec.max_epochs:
            continue
        model.eval()
        _, validation_predictions = _predict_pairs(
            model,
            state_scaler,
            base,
            correction_scale,
            validation_pairs,
            state_lookup,
        )
        metrics = _prediction_metrics(validation_predictions)
        candidate_key = (
            metrics["fitness"],
            metrics["rmse"],
            metrics["mae"],
            epoch,
        )
        improved = candidate_key[0] < best_key[0] - spec.min_improvement
        if improved:
            best_key = candidate_key
            best_metrics = metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        rows.append(
            {
                "epoch": epoch,
                "total_train_loss": float(total_loss.detach()),
                "correction_train_loss": float(correction_loss.detach()),
                "similarity_train_loss": float(similarity_loss.detach()),
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "validation_block_rmse_std": metrics[
                    "chronological_block_rmse_std"
                ],
                "fitness": metrics["fitness"],
                "is_new_best": int(improved),
            }
        )
        if evaluations_without_improvement >= spec.patience_evaluations:
            break

    model.load_state_dict(best_state)
    model.eval()
    return ResidualBundle(
        model=model,
        state_scaler=state_scaler,
        base=base,
        correction_scale=correction_scale,
        similarity_tau=similarity_tau,
        spec=spec,
        seed=int(seed),
        similarity_weight=float(similarity_weight),
        best_epoch=int(best_epoch),
        validation_metrics=best_metrics,
        training_history=pd.DataFrame(rows),
    )


def _ensemble_predictions(
    predictions_by_seed: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Average predeclared seed predictions without looking at test metrics."""
    if not predictions_by_seed:
        raise ValueError("cannot ensemble an empty prediction collection")
    merged: pd.DataFrame | None = None
    prediction_columns: list[str] = []
    for seed, predictions in sorted(predictions_by_seed.items()):
        column = f"cpi_predicted_seed_{seed}"
        prediction_columns.append(column)
        current = predictions[
            ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
        ].rename(columns={"cpi_predicted": column})
        merged = (
            current
            if merged is None
            else merged.merge(
                current,
                on=["sample_i_id", "target_date", "cpi_actual"],
                validate="one_to_one",
            )
        )
    assert merged is not None
    merged["cpi_predicted"] = merged[prediction_columns].mean(axis=1)
    merged["prediction_seed_std"] = merged[prediction_columns].std(
        axis=1, ddof=0
    )
    merged["error"] = merged["cpi_predicted"] - merged["cpi_actual"]
    merged["absolute_error"] = merged["error"].abs()
    return merged.sort_values("target_date").reset_index(drop=True)


def _save_bundle(bundle: ResidualBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": bundle.model.state_dict(),
            "residual_spec": asdict(bundle.spec),
            "seed": bundle.seed,
            "similarity_weight": bundle.similarity_weight,
            "best_epoch": bundle.best_epoch,
            "state_scaler_mean": bundle.state_scaler.mean_,
            "state_scaler_scale": bundle.state_scaler.scale_,
            "correction_scale": bundle.correction_scale,
            "similarity_tau": bundle.similarity_tau,
            "frozen_ridge_alpha": bundle.base.alpha,
        },
        path,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figures(
    histories: pd.DataFrame,
    comparison: pd.DataFrame,
    unified: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9.5, 5.0))
    for seed, group in histories.groupby("seed"):
        axis.plot(group["epoch"], group["val_rmse"], label=f"seed {seed}")
    axis.axhline(0.44679092931184033, color="black", linestyle="--", label="fixed SSA base")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation RMSE")
    axis.set_title("Shared-projection residual validation convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_convergence_by_seed.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    axis.plot(unified["target_date"], unified["cpi_actual"], color="black", linewidth=2, label="Actual")
    axis.plot(unified["target_date"], unified["cpi_predicted_ordinary"], label="Ordinary optical")
    axis.plot(unified["target_date"], unified["cpi_predicted_current_ssa"], label="Current SSA Ridge")
    axis.plot(
        unified["target_date"],
        unified["cpi_predicted_shared_residual"],
        label="SSA fallback (shared branch rejected)",
    )
    axis.tick_params(axis="x", rotation=60)
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI MoM index")
    axis.set_title("Strict-closed50 test predictions")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_prediction_comparison.png", dpi=180)
    plt.close(fig)

    ordered = comparison.sort_values("test_rmse")
    x = np.arange(len(ordered))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.5, 5.0))
    axis.bar(x - width / 2, ordered["test_mae"], width, label="MAE")
    axis.bar(x + width / 2, ordered["test_rmse"], width, label="RMSE")
    axis.set_xticks(x, ordered["model"], rotation=18, ha="right")
    axis.set_ylabel("Error")
    axis.set_title("Test metric comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_metric_comparison.png", dpi=180)
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    comparison: pd.DataFrame,
    selected_weight: float,
    seeds: Sequence[int],
) -> None:
    display = comparison.copy()
    for column in ("val_mae", "val_rmse", "test_mae", "test_rmse"):
        display[column] = display[column].map(lambda value: f"{float(value):.6f}")
    table_header = "| 模型 | 验证MAE | 验证RMSE | 测试MAE | 测试RMSE |"
    table_rule = "|---|---:|---:|---:|---:|"
    table_rows = [
        f"| {row.model} | {row.val_mae} | {row.val_rmse} | {row.test_mae} | {row.test_rmse} |"
        for row in display.itertuples(index=False)
    ]
    fallback = comparison.loc[
        comparison["model"].eq("ssa_fallback_shared_projection_rejected")
    ].iloc[0]
    base = comparison.loc[comparison["model"].eq("current_ssa_pairwise_ridge")].iloc[0]
    lines = [
        "# 环比严格封闭50样本：共享投影孪生残差模型",
        "",
        "该方案将现有SSA孪生Ridge作为冻结底座，增加零初始化的共享投影残差支路。",
        "目标状态和参考状态都通过同一个50→16→8投影网络，再预测CPI差值修正量。",
        "相似度损失只使用train50样本对；验证集仅选择损失权重和每个种子的早停轮次。",
        "",
        f"- 固定随机种子：`{list(seeds)}`，测试采用等权集成，不挑测试最优种子。",
        f"- 验证选出的相似度权重：`{selected_weight}`。",
        "- 参考窗口、K=5、混合距离和距离幂0.654固定为此前SSA验证结果。",
        "- epoch 0神经修正严格为0，因此验证性能最差可以回退到SSA Ridge底座。",
        "- 本次三个种子全部在epoch 0最优，验证集因此拒绝共享投影修正；最终预测只是SSA安全回退。",
        "",
        table_header,
        table_rule,
        *table_rows,
        "",
        f"回退结果相对SSA底座测试MAE变化：`{100 * (fallback.test_mae - base.test_mae) / base.test_mae:+.2f}%`。",
        f"回退结果相对SSA底座测试RMSE变化：`{100 * (fallback.test_rmse - base.test_rmse) / base.test_rmse:+.2f}%`。",
        "",
        "注意：47个月测试区间此前已被观察，本次只能视为探索性复分析，不是新的盲测。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    output_dir: Path = OUTPUT_DIR,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    similarity_weights: Iterable[float] = DEFAULT_SIMILARITY_WEIGHTS,
    spec: ResidualSpec = ResidualSpec(),
    validation_only: bool = False,
    training_pair_mode: str = "ssa_selected_92",
) -> pd.DataFrame:
    """Select on validation means, freeze, and optionally evaluate test once."""
    seeds = tuple(int(seed) for seed in seeds)
    similarity_weights = tuple(float(value) for value in similarity_weights)
    if not seeds or not similarity_weights:
        raise ValueError("seeds and similarity weights must be non-empty")

    train_pool = build_isolated_closed_train_pool(DATA_DIR)
    if len(train_pool["index"]) != 50:
        raise ValueError("residual experiment requires exactly train50")
    selected_train_pairs, validation_pairs = load_pretest_pairs()
    if training_pair_mode == "ssa_selected_92":
        train_pairs = selected_train_pairs
    elif training_pair_mode == "all_causal_741":
        train_pairs = build_causal_train_pairs(train_pool, min_gap_months=1)
    else:
        raise ValueError(f"unknown training_pair_mode: {training_pair_mode}")
    validation_states = load_validation_state_lookup(STATE_DIR)
    base = load_frozen_ridge_base()

    bundles: dict[tuple[float, int], ResidualBundle] = {}
    validation_pair_outputs: dict[tuple[float, int], pd.DataFrame] = {}
    validation_predictions: dict[tuple[float, int], pd.DataFrame] = {}
    trial_rows: list[dict[str, object]] = []
    for weight in similarity_weights:
        for seed in seeds:
            bundle = fit_residual_bundle(
                train_pool,
                train_pairs,
                validation_pairs,
                validation_states,
                base,
                spec,
                seed,
                weight,
            )
            pair_output, predictions = predict_bundle(
                bundle, validation_pairs, validation_states
            )
            metrics = _prediction_metrics(predictions)
            bundles[(weight, seed)] = bundle
            validation_pair_outputs[(weight, seed)] = pair_output
            validation_predictions[(weight, seed)] = predictions
            trial_rows.append(
                {
                    "similarity_weight": weight,
                    "seed": seed,
                    "best_epoch": bundle.best_epoch,
                    "val_mae": metrics["mae"],
                    "val_rmse": metrics["rmse"],
                    "validation_block_rmse_std": metrics[
                        "chronological_block_rmse_std"
                    ],
                    "fitness": metrics["fitness"],
                }
            )

    trials = pd.DataFrame(trial_rows)
    summary = (
        trials.groupby("similarity_weight", as_index=False)
        .agg(
            num_seeds=("seed", "count"),
            mean_val_mae=("val_mae", "mean"),
            std_val_mae=("val_mae", "std"),
            mean_val_rmse=("val_rmse", "mean"),
            std_val_rmse=("val_rmse", "std"),
            mean_fitness=("fitness", "mean"),
            std_fitness=("fitness", "std"),
            mean_best_epoch=("best_epoch", "mean"),
        )
        .fillna(0.0)
    )
    selected_summary = summary.sort_values(
        ["mean_fitness", "mean_val_rmse", "mean_val_mae", "similarity_weight"]
    ).iloc[0]
    selected_weight = float(selected_summary["similarity_weight"])
    selected_bundles = {
        seed: bundles[(selected_weight, seed)] for seed in seeds
    }
    val_seed_predictions = {
        seed: validation_predictions[(selected_weight, seed)] for seed in seeds
    }
    validation_ensemble = _ensemble_predictions(val_seed_predictions)
    validation_ensemble_metrics = _prediction_metrics(validation_ensemble)

    freeze_payload = {
        "selected_similarity_weight": selected_weight,
        "seeds": list(seeds),
        "best_epochs": {
            str(seed): selected_bundles[seed].best_epoch for seed in seeds
        },
        "fixed_reference_strategy": "SSA hybrid-distance train50 bank",
        "training_pair_mode": training_pair_mode,
        "k_references": 5,
        "distance_power": DISTANCE_POWER,
        "residual_spec": asdict(spec),
    }
    freeze_json = json.dumps(freeze_payload, sort_keys=True, ensure_ascii=False)
    freeze_sha256 = hashlib.sha256(freeze_json.encode("utf-8")).hexdigest()

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    trials.to_csv(table_dir / "validation_metrics_by_seed.csv", index=False)
    summary.to_csv(table_dir / "validation_similarity_weight_summary.csv", index=False)
    train_pairs.to_csv(table_dir / "training_pairs.csv", index=False)
    validation_pairs.to_csv(table_dir / "validation_pairs.csv", index=False)
    validation_ensemble.to_csv(
        table_dir / "validation_ensemble_predictions.csv", index=False
    )
    history_frames: list[pd.DataFrame] = []
    for seed, bundle in selected_bundles.items():
        validation_pair_outputs[(selected_weight, seed)].to_csv(
            table_dir / f"validation_pair_predictions_seed_{seed}.csv", index=False
        )
        validation_predictions[(selected_weight, seed)].to_csv(
            table_dir / f"validation_predictions_seed_{seed}.csv", index=False
        )
        history = bundle.training_history.assign(
            seed=seed, similarity_weight=selected_weight
        )
        history_frames.append(history)
        _save_bundle(bundle, model_dir / f"shared_projection_residual_seed_{seed}.pt")
    histories = pd.concat(history_frames, ignore_index=True)
    histories.to_csv(table_dir / "training_history_selected_weight.csv", index=False)
    (output_dir / "frozen_configuration.json").write_text(
        json.dumps(
            {**freeze_payload, "configuration_sha256": freeze_sha256},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if validation_only:
        manifest = {
            "experiment": "MoM strict-closed50 shared-projection residual validation",
            "created_at": datetime.now().astimezone().isoformat(),
            "validation_only": True,
            "test_arrays_states_or_pair_tables_opened": False,
            "pretest_split_loading": "physically isolated train/val files",
            "combined_sample_index_or_mat_opened": False,
            "pretest_files": [
                "sample_index_train.csv",
                "X_train.npy",
                "y_train.npy",
                "states_train.mat",
                "states_val.mat",
                "ssa_selected_train_pairs.csv",
                "ssa_selected_validation_pairs.csv",
            ],
            "accessible_train_windows": 50,
            "num_training_pairs": int(len(train_pairs)),
            "num_validation_targets": 45,
            "frozen_ridge_alpha": base.alpha,
            "configuration": freeze_payload,
            "configuration_sha256": freeze_sha256,
            "validation_ensemble_metrics": validation_ensemble_metrics,
        }
        (output_dir / "validation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return pd.DataFrame(
            [
                {
                    "model": "shared_projection_residual_validation_ensemble",
                    "selected_similarity_weight": selected_weight,
                    "val_mae": validation_ensemble_metrics["mae"],
                    "val_rmse": validation_ensemble_metrics["rmse"],
                    "fitness": validation_ensemble_metrics["fitness"],
                }
            ]
        )

    # The selection and every seed checkpoint are frozen above.  Only now may
    # the historical test pair table, test arrays, and test states be opened.
    audit = audit_profile(DATA_DIR, STATE_DIR)
    test_pairs = load_postfreeze_test_pairs()
    all_states = load_state_lookup(STATE_DIR)
    test_predictions_by_seed: dict[int, pd.DataFrame] = {}
    metric_rows: list[dict[str, object]] = []
    for seed, bundle in selected_bundles.items():
        pair_output, predictions = predict_bundle(bundle, test_pairs, all_states)
        pair_output.to_csv(
            table_dir / f"test_pair_predictions_seed_{seed}.csv", index=False
        )
        predictions.to_csv(
            table_dir / f"test_predictions_seed_{seed}.csv", index=False
        )
        test_predictions_by_seed[seed] = predictions
        test_metrics = _prediction_metrics(predictions)
        val_metrics = selected_bundles[seed].validation_metrics
        metric_rows.append(
            {
                "seed": seed,
                "similarity_weight": selected_weight,
                "best_epoch": bundle.best_epoch,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            }
        )
    metrics_by_seed = pd.DataFrame(metric_rows)
    metrics_by_seed.to_csv(table_dir / "selected_metrics_by_seed.csv", index=False)
    test_ensemble = _ensemble_predictions(test_predictions_by_seed)
    test_ensemble.to_csv(table_dir / "test_ensemble_predictions.csv", index=False)
    test_ensemble_metrics = _prediction_metrics(test_ensemble)

    comparison, unified = _ordinary_and_current_outputs()
    comparison["shared_projection_architecture_present"] = False
    comparison["effective_shared_projection"] = False
    comparison["selection_outcome"] = "not_applicable"
    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [
                    {
                        "model": "ssa_fallback_shared_projection_rejected",
                        "trainable_shared_projection": False,
                        "similarity_constraint": selected_weight > 0,
                        "shared_projection_architecture_present": True,
                        "effective_shared_projection": False,
                        "selection_outcome": (
                            "all_seeds_selected_epoch0; neural branch rejected"
                        ),
                        "val_mae": validation_ensemble_metrics["mae"],
                        "val_rmse": validation_ensemble_metrics["rmse"],
                        "test_mae": test_ensemble_metrics["mae"],
                        "test_rmse": test_ensemble_metrics["rmse"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    new_predictions = test_ensemble[
        ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(columns={"cpi_predicted": "cpi_predicted_shared_residual"})
    unified = unified.merge(
        new_predictions,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    )
    if len(unified) != 47:
        raise ValueError(f"test ensemble has {len(unified)} targets, expected 47")
    unified["residual_shared_residual"] = (
        unified["cpi_predicted_shared_residual"] - unified["cpi_actual"]
    )
    unified["absolute_error_shared_residual"] = unified[
        "residual_shared_residual"
    ].abs()
    comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    unified.to_csv(table_dir / "test_prediction_comparison.csv", index=False)
    test_pairs.to_csv(table_dir / "test_pairs.csv", index=False)
    _save_figures(histories, comparison, unified, figure_dir)

    train_ids = set(train_pool["index"]["sample_id"].astype(int))
    all_reference_ids = set(train_pairs["sample_j_id"].astype(int))
    all_reference_ids.update(validation_pairs["sample_j_id"].astype(int))
    all_reference_ids.update(test_pairs["sample_j_id"].astype(int))
    manifest = {
        "experiment": "MoM strict-closed50 stabilized shared-projection Siamese residual",
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_status": (
            "exploratory re-analysis; historical test previously observed"
        ),
        "profile_audit": audit,
        "data_directory": str(DATA_DIR),
        "state_directory": str(STATE_DIR),
        "frozen_optical_reservoir": True,
        "reservoir_parameters_changed": False,
        "frozen_base_model": str(BASE_MODEL_PATH),
        "frozen_base_alpha": base.alpha,
        "candidate_trainable_components": [
            "shared_projection",
            "zero_initialized_antisymmetric_residual_head",
        ],
        "selected_effective_components": ["frozen_ssa_ridge_base"],
        "effective_shared_projection": False,
        "selection_outcome": (
            "all seeds selected epoch 0; shared projection correction rejected by validation"
        ),
        "accessible_train_windows": 50,
        "validation_targets": 45,
        "test_targets": 47,
        "num_training_pairs": int(len(train_pairs)),
        "num_training_pair_targets": int(train_pairs["sample_i_id"].nunique()),
        "num_validation_pairs": int(len(validation_pairs)),
        "num_test_pairs": int(len(test_pairs)),
        "configuration": freeze_payload,
        "configuration_sha256": freeze_sha256,
        "configuration_selection_rule": (
            "minimum mean validation fitness across all predeclared seeds"
        ),
        "test_seed_rule": "equal-weight ensemble of every predeclared seed",
        "all_references_inside_train50": all_reference_ids.issubset(train_ids),
        "target_labels_directly_used_in_reference_ranking_rows": False,
        "reference_strategy_previously_selected_on_validation": True,
        "state_scaler_fit_on_train50_only": True,
        "similarity_targets_fit_on_training_pairs_only": True,
        "test_semantic_payload_loaded_after_configuration_freeze": True,
        "test_labels_used_for_selection": False,
        "historical_test_previously_observed": True,
        "pretest_split_loading": "physically isolated train/val files",
        "combined_sample_index_or_mat_opened_before_freeze": False,
        "isolated_split_manifest": str(DATA_DIR / "isolated_split_manifest.json"),
        "validation_ensemble_metrics": validation_ensemble_metrics,
        "test_ensemble_metrics": test_ensemble_metrics,
        "model_comparison": comparison.to_dict(orient="records"),
        "input_sha256": {
            "isolated_split_manifest": _file_sha256(
                DATA_DIR / "isolated_split_manifest.json"
            ),
            "sample_index_train": _file_sha256(DATA_DIR / "sample_index_train.csv"),
            "X_train": _file_sha256(DATA_DIR / "X_train.npy"),
            "y_train": _file_sha256(DATA_DIR / "y_train.npy"),
            "sample_index_val": _file_sha256(DATA_DIR / "sample_index_val.csv"),
            "X_val": _file_sha256(DATA_DIR / "X_val.npy"),
            "y_val": _file_sha256(DATA_DIR / "y_val.npy"),
            "sample_index_test": _file_sha256(DATA_DIR / "sample_index_test.csv"),
            "X_test": _file_sha256(DATA_DIR / "X_test.npy"),
            "y_test": _file_sha256(DATA_DIR / "y_test.npy"),
            "frozen_base_model": _file_sha256(BASE_MODEL_PATH),
            "training_pairs": _file_sha256(BASE_TRAIN_PAIRS_PATH),
            "validation_pairs": _file_sha256(BASE_VALIDATION_PAIRS_PATH),
            "test_pairs": _file_sha256(BASE_TEST_PAIRS_PATH),
        },
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_readme(output_dir, comparison, selected_weight, seeds)

    artifact_rows: list[dict[str, object]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256.csv":
            artifact_rows.append(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    pd.DataFrame(artifact_rows).to_csv(
        output_dir / "artifact_sha256.csv", index=False
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-epochs", type=int, default=ResidualSpec().max_epochs)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--training-pair-mode",
        choices=("ssa_selected_92", "all_causal_741"),
        default="ssa_selected_92",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ResidualSpec(max_epochs=int(args.max_epochs))
    result = run_pipeline(
        output_dir=args.output_dir,
        seeds=args.seeds,
        spec=spec,
        validation_only=bool(args.validation_only),
        training_pair_mode=str(args.training_pair_mode),
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
