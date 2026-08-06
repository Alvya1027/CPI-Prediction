"""Train a true shared-projection Siamese readout on frozen optical states.

The optical reservoir remains the common, frozen temporal encoder.  A small
projection network is called for both the target and reference branches with
the exact same parameters.  A relation head predicts the continuous CPI
difference, while a continuous similarity loss shapes the learned embedding.

The formal experiment is leakage-safe: model and reference hyperparameters are
selected with the training and validation splits only.  Test arrays and states
are first opened after both the delta-only ablation and the similarity-
constrained configuration have been frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
from torch import nn
from torch.nn import functional as F

from src.config import RESULTS_DIR, ROOT_DIR
from src.create_siamese_pairs import _build_candidate_pairs
from src.siamese_mom_closed50_pipeline import audit_profile
from src.siamese_reservoir_regression import load_state_lookup, regression_metrics
from src.siamese_split_isolation import (
    build_isolated_closed_train_pool,
    load_isolated_split,
)
from src.siamese_validation_search import load_validation_state_lookup


PROFILE_ROOT = ROOT_DIR / "matlab" / "optical_reservoir_cpi_mom_recent50_20260730"
DATA_DIR = PROFILE_ROOT / "data"
STATE_DIR = PROFILE_ROOT / "states"
ORDINARY_DIR = RESULTS_DIR / "optical_reservoir_mom_recent50_20260730"
CURRENT_SIAMESE_DIR = RESULTS_DIR / "siamese_optical_mom_closed50_ssa_20260730"
OUTPUT_DIR = RESULTS_DIR / "siamese_optical_mom_shared_projection_20260802"

DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_SIMILARITY_WEIGHTS = (0.0, 0.05, 0.20)
DEFAULT_K_VALUES = (3, 5, 8)
DEFAULT_ATTENTION_TEMPERATURES = (0.10, 0.25, 0.50)


@dataclass(frozen=True)
class NetworkSpec:
    """Fixed small-network capacity used for every validation candidate."""

    projection_hidden_dim: int = 16
    embedding_dim: int = 8
    relation_hidden_dim: int = 8
    dropout: float = 0.10
    learning_rate: float = 0.005
    weight_decay: float = 0.001
    epochs: int = 600
    min_gap_months: int = 1


@dataclass(frozen=True)
class SelectedConfig:
    """One validation-frozen model and inference configuration."""

    seed: int
    similarity_weight: float
    k_references: int
    attention_temperature: float


@dataclass
class ProjectionBundle:
    """Fitted shared projection model plus training-only scaling metadata."""

    model: "SharedProjectionSiamese"
    scaler: StandardScaler
    target_scale: float
    similarity_tau: float
    spec: NetworkSpec
    seed: int
    similarity_weight: float
    training_history: pd.DataFrame


class SharedProjectionSiamese(nn.Module):
    """One shared projection called twice, followed by an antisymmetric head."""

    def __init__(self, state_width: int, spec: NetworkSpec) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(state_width, spec.projection_hidden_dim),
            nn.Tanh(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.projection_hidden_dim, spec.embedding_dim),
        )
        relation_width = 3 * spec.embedding_dim
        self.relation = nn.Sequential(
            nn.Linear(relation_width, spec.relation_hidden_dim),
            nn.Tanh(),
            nn.Linear(spec.relation_hidden_dim, 1),
        )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        """Map frozen optical states into one learned normalized embedding."""
        return F.normalize(self.projection(states), p=2, dim=1, eps=1e-8)

    @staticmethod
    def _relation_features(
        target_embedding: torch.Tensor,
        reference_embedding: torch.Tensor,
    ) -> torch.Tensor:
        signed = target_embedding - reference_embedding
        return torch.cat(
            [signed, torch.abs(signed), target_embedding * reference_embedding],
            dim=1,
        )

    def predict_scaled_delta_from_embeddings(
        self,
        target_embedding: torch.Tensor,
        reference_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict an ordered delta and enforce f(i,j) = -f(j,i)."""
        forward = self.relation(
            self._relation_features(target_embedding, reference_embedding)
        ).reshape(-1)
        reverse = self.relation(
            self._relation_features(reference_embedding, target_embedding)
        ).reshape(-1)
        return 0.5 * (forward - reverse)

    def forward(
        self,
        target_states: torch.Tensor,
        reference_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_embedding = self.encode(target_states)
        reference_embedding = self.encode(reference_states)
        delta = self.predict_scaled_delta_from_embeddings(
            target_embedding,
            reference_embedding,
        )
        return delta, target_embedding, reference_embedding


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # The CPU operations used here are deterministic on supported builds.
        pass


def build_causal_train_pairs(
    train_pool: dict[str, object],
    min_gap_months: int = 1,
) -> pd.DataFrame:
    """Build every legal ordered pair while keeping both sides inside train50."""
    pairs = _build_candidate_pairs(
        train_pool,
        train_pool,
        min_gap_months=min_gap_months,
    ).reset_index(drop=True)
    if pairs.empty:
        raise ValueError("shared-projection training pair pool is empty")
    train_ids = set(train_pool["index"]["sample_id"].astype(int))
    pair_ids = set(pairs["sample_i_id"].astype(int)).union(
        pairs["sample_j_id"].astype(int)
    )
    if not pair_ids.issubset(train_ids):
        raise ValueError("a shared-projection training pair escaped train50")
    target_dates = pd.to_datetime(pairs["target_i_date"])
    reference_dates = pd.to_datetime(pairs["target_j_date"])
    if not (reference_dates < target_dates).all():
        raise ValueError("training pairs are not chronologically causal")
    if pairs.duplicated(["sample_i_id", "sample_j_id"]).any():
        raise ValueError("training pair pool contains duplicates")
    return pairs


def _state_matrix(
    sample_ids: Sequence[int] | np.ndarray,
    state_lookup: dict[int, np.ndarray],
) -> np.ndarray:
    missing = set(int(value) for value in sample_ids).difference(state_lookup)
    if missing:
        raise ValueError(f"missing optical states for sample IDs: {sorted(missing)[:10]}")
    # Reservoir states are around 1e12.  Preserve their smaller variations in
    # float64 until the train-only StandardScaler has centered and scaled them.
    return np.vstack([state_lookup[int(value)] for value in sample_ids]).astype(
        np.float64
    )


def _balanced_pair_weights(pairs: pd.DataFrame) -> np.ndarray:
    """Give each target month equal total influence despite unequal pair counts."""
    counts = pairs.groupby("sample_i_id")["sample_j_id"].transform("count")
    weights = 1.0 / counts.to_numpy(dtype=np.float32)
    return weights / float(np.mean(weights))


def fit_projection_model(
    train_pool: dict[str, object],
    train_pairs: pd.DataFrame,
    state_lookup: dict[int, np.ndarray],
    spec: NetworkSpec,
    seed: int,
    similarity_weight: float,
) -> ProjectionBundle:
    """Fit one full-batch model without using validation or test labels."""
    _set_deterministic_seed(seed)
    train_ids = train_pool["index"]["sample_id"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(_state_matrix(train_ids, state_lookup))
    state_i = scaler.transform(
        _state_matrix(train_pairs["sample_i_id"].to_numpy(dtype=int), state_lookup)
    ).astype(np.float32)
    state_j = scaler.transform(
        _state_matrix(train_pairs["sample_j_id"].to_numpy(dtype=int), state_lookup)
    ).astype(np.float32)
    delta = train_pairs["delta_cpi"].to_numpy(dtype=np.float32)
    train_targets = np.asarray(train_pool["y"], dtype=np.float32).reshape(-1)
    target_scale = max(float(np.std(train_targets)), 1e-6)
    nonzero_delta = np.abs(delta[np.abs(delta) > 1e-8])
    similarity_tau = max(
        float(np.median(nonzero_delta)) if len(nonzero_delta) else target_scale,
        1e-3,
    )
    pair_weights = _balanced_pair_weights(train_pairs)

    tensor_i = torch.from_numpy(state_i)
    tensor_j = torch.from_numpy(state_j)
    tensor_delta = torch.from_numpy(delta / target_scale)
    tensor_similarity = torch.from_numpy(
        np.exp(-np.abs(delta) / similarity_tau).astype(np.float32)
    )
    tensor_weights = torch.from_numpy(pair_weights)

    model = SharedProjectionSiamese(state_i.shape[1], spec)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, spec.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted_delta, embedding_i, embedding_j = model(tensor_i, tensor_j)
        delta_each = F.smooth_l1_loss(
            predicted_delta,
            tensor_delta,
            reduction="none",
        )
        cosine_similarity = torch.sum(embedding_i * embedding_j, dim=1)
        similarity_each = torch.square(cosine_similarity - tensor_similarity)
        delta_loss = torch.mean(delta_each * tensor_weights)
        metric_loss = torch.mean(similarity_each * tensor_weights)
        loss = delta_loss + float(similarity_weight) * metric_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("shared-projection training produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == spec.epochs:
            rows.append(
                {
                    "epoch": epoch,
                    "total_loss": float(loss.detach()),
                    "delta_loss": float(delta_loss.detach()),
                    "similarity_loss": float(metric_loss.detach()),
                }
            )
    model.eval()
    return ProjectionBundle(
        model=model,
        scaler=scaler,
        target_scale=target_scale,
        similarity_tau=similarity_tau,
        spec=spec,
        seed=int(seed),
        similarity_weight=float(similarity_weight),
        training_history=pd.DataFrame(rows),
    )


def _encode_numpy(bundle: ProjectionBundle, states: np.ndarray) -> np.ndarray:
    scaled = bundle.scaler.transform(np.asarray(states, dtype=np.float64)).astype(
        np.float32
    )
    with torch.no_grad():
        return bundle.model.encode(torch.from_numpy(scaled)).cpu().numpy()


def select_embedding_references(
    candidates: pd.DataFrame,
    target_embeddings: dict[int, np.ndarray],
    reference_embeddings: dict[int, np.ndarray],
    k_references: int,
) -> pd.DataFrame:
    """Select K references by learned embeddings; labels never affect ranking."""
    output = candidates.copy()
    output["embedding_distance"] = [
        float(
            np.linalg.norm(
                target_embeddings[int(i)] - reference_embeddings[int(j)]
            )
        )
        for i, j in zip(output["sample_i_id"], output["sample_j_id"])
    ]
    output = output.sort_values(
        ["sample_i_id", "embedding_distance", "target_j_date", "sample_j_id"],
        ascending=[True, True, False, True],
    )
    selected = (
        output.groupby("sample_i_id", sort=False, group_keys=False)
        .head(int(k_references))
        .copy()
        .reset_index(drop=True)
    )
    sizes = selected.groupby("sample_i_id").size()
    if len(sizes) != candidates["sample_i_id"].nunique():
        raise ValueError("embedding reference selection dropped a target")
    if not (sizes == int(k_references)).all():
        raise ValueError(f"a target has fewer than K={k_references} references")
    selected["selection_method"] = f"learned_embedding_k{int(k_references)}"
    return selected


def predict_split(
    bundle: ProjectionBundle,
    train_pool: dict[str, object],
    target_split: dict[str, object],
    state_lookup: dict[int, np.ndarray],
    k_references: int,
    attention_temperature: float,
    min_gap_months: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Predict one split from a fixed train50 support bank."""
    if attention_temperature <= 0:
        raise ValueError("attention_temperature must be positive")
    candidates = _build_candidate_pairs(
        target_split,
        train_pool,
        min_gap_months=min_gap_months,
    ).reset_index(drop=True)
    if candidates.empty:
        raise ValueError("evaluation candidate pool is empty")
    train_ids = train_pool["index"]["sample_id"].to_numpy(dtype=int)
    target_ids = target_split["index"]["sample_id"].to_numpy(dtype=int)
    train_embedding_values = _encode_numpy(
        bundle,
        _state_matrix(train_ids, state_lookup),
    )
    target_embedding_values = _encode_numpy(
        bundle,
        _state_matrix(target_ids, state_lookup),
    )
    train_embeddings = dict(zip(train_ids.tolist(), train_embedding_values))
    target_embeddings = dict(zip(target_ids.tolist(), target_embedding_values))
    selected = select_embedding_references(
        candidates,
        target_embeddings,
        train_embeddings,
        k_references,
    )

    states_i = bundle.scaler.transform(
        _state_matrix(selected["sample_i_id"].to_numpy(dtype=int), state_lookup)
    ).astype(np.float32)
    states_j = bundle.scaler.transform(
        _state_matrix(selected["sample_j_id"].to_numpy(dtype=int), state_lookup)
    ).astype(np.float32)
    with torch.no_grad():
        scaled_delta, _, _ = bundle.model(
            torch.from_numpy(states_i),
            torch.from_numpy(states_j),
        )
    pair_predictions = selected.copy()
    pair_predictions["delta_cpi_predicted"] = (
        scaled_delta.cpu().numpy() * bundle.target_scale
    )
    pair_predictions["cpi_pred_pair"] = (
        pair_predictions["cpi_j"].to_numpy(dtype=float)
        + pair_predictions["delta_cpi_predicted"].to_numpy(dtype=float)
    )
    pair_predictions["delta_error"] = (
        pair_predictions["delta_cpi_predicted"] - pair_predictions["delta_cpi"]
    )

    rows: list[dict[str, object]] = []
    for sample_i_id, group in pair_predictions.groupby("sample_i_id", sort=False):
        scaled_logits = (
            -group["embedding_distance"].to_numpy(dtype=float)
            / float(attention_temperature)
        )
        scaled_logits -= float(np.max(scaled_logits))
        weights = np.exp(scaled_logits)
        weights /= float(np.sum(weights))
        estimates = group["cpi_pred_pair"].to_numpy(dtype=float)
        actual = group["cpi_i"].to_numpy(dtype=float)
        if not np.allclose(actual, actual[0]):
            raise ValueError(f"inconsistent target labels for sample {sample_i_id}")
        rows.append(
            {
                "sample_i_id": int(sample_i_id),
                "target_date": str(group["target_i_date"].iloc[0]),
                "cpi_actual": float(actual[0]),
                "cpi_predicted": float(np.sum(weights * estimates)),
                "num_references": int(len(group)),
                "mean_embedding_distance": float(
                    group["embedding_distance"].mean()
                ),
                "reference_prediction_std": float(np.std(estimates)),
                "attention_effective_references": float(
                    1.0 / np.sum(np.square(weights))
                ),
            }
        )
    predictions = pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)
    predictions["error"] = predictions["cpi_predicted"] - predictions["cpi_actual"]
    predictions["absolute_error"] = predictions["error"].abs()
    return candidates, pair_predictions, predictions


def _chronological_block_rmse_std(predictions: pd.DataFrame) -> float:
    ordered = predictions.sort_values("target_date").reset_index(drop=True)
    index_blocks = np.array_split(np.arange(len(ordered)), 3)
    values = [
        math.sqrt(
            float(
                np.mean(
                    np.square(
                        ordered.iloc[index_block]["error"].to_numpy(dtype=float)
                    )
                )
            )
        )
        for index_block in index_blocks
    ]
    return float(np.std(values))


def _selection_row(
    bundle: ProjectionBundle,
    predictions: pd.DataFrame,
    k_references: int,
    attention_temperature: float,
) -> dict[str, object]:
    metrics = regression_metrics(predictions["cpi_actual"], predictions["cpi_predicted"])
    block_std = _chronological_block_rmse_std(predictions)
    return {
        "seed": bundle.seed,
        "similarity_weight": bundle.similarity_weight,
        "k_references": int(k_references),
        "attention_temperature": float(attention_temperature),
        "val_mae": metrics["mae"],
        "val_rmse": metrics["rmse"],
        "validation_block_rmse_std": block_std,
        "fitness": metrics["rmse"] + 0.10 * block_std,
    }


def _config_key(row: dict[str, object]) -> tuple[float, ...]:
    return (
        float(row["fitness"]),
        float(row["val_rmse"]),
        float(row["val_mae"]),
        float(row["similarity_weight"]),
        float(row["k_references"]),
        float(row["attention_temperature"]),
        float(row["seed"]),
    )


def _select_configuration(
    trials: list[dict[str, object]],
    constrained: bool,
) -> dict[str, object]:
    eligible = [
        row
        for row in trials
        if (float(row["similarity_weight"]) > 0.0) == constrained
    ]
    if not eligible:
        raise ValueError("validation trials do not contain the requested model family")
    table = pd.DataFrame(eligible)
    group_columns = [
        "similarity_weight",
        "k_references",
        "attention_temperature",
    ]
    summary = (
        table.groupby(group_columns, as_index=False)
        .agg(
            selection_num_seeds=("seed", "count"),
            selection_mean_fitness=("fitness", "mean"),
            selection_mean_val_rmse=("val_rmse", "mean"),
            selection_mean_val_mae=("val_mae", "mean"),
            selection_std_fitness=("fitness", "std"),
        )
        .fillna(0.0)
        .sort_values(
            [
                "selection_mean_fitness",
                "selection_mean_val_rmse",
                "selection_mean_val_mae",
                "similarity_weight",
                "k_references",
                "attention_temperature",
            ]
        )
    )
    selected_family = summary.iloc[0]
    matching = table.copy()
    for column in group_columns:
        matching = matching.loc[
            np.isclose(
                matching[column].to_numpy(dtype=float),
                float(selected_family[column]),
            )
        ]
    # This from-scratch model is an ablation, so use the smallest predeclared
    # seed after family selection rather than cherry-picking the best seed.
    selected = matching.sort_values("seed").iloc[0].to_dict()
    for column in (
        "selection_num_seeds",
        "selection_mean_fitness",
        "selection_mean_val_rmse",
        "selection_mean_val_mae",
        "selection_std_fitness",
    ):
        selected[column] = float(selected_family[column])
    return selected


def _ordinary_and_current_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    ordinary_metrics = pd.read_csv(
        ORDINARY_DIR / "tables" / "optical_reservoir_metrics.csv"
    ).set_index("split")
    current_comparison = pd.read_csv(
        CURRENT_SIAMESE_DIR / "tables" / "model_comparison.csv"
    ).set_index("model")
    comparison = pd.DataFrame(
        [
            {
                "model": "ordinary_optical_reservoir",
                "trainable_shared_projection": False,
                "similarity_constraint": False,
                "val_mae": float(ordinary_metrics.loc["val", "mae"]),
                "val_rmse": float(ordinary_metrics.loc["val", "rmse"]),
                "test_mae": float(ordinary_metrics.loc["test", "mae"]),
                "test_rmse": float(ordinary_metrics.loc["test", "rmse"]),
            },
            {
                "model": "current_ssa_pairwise_ridge",
                "trainable_shared_projection": False,
                "similarity_constraint": False,
                "val_mae": float(
                    current_comparison.loc["siamese_closed50_ssa", "val_mae"]
                ),
                "val_rmse": float(
                    current_comparison.loc["siamese_closed50_ssa", "val_rmse"]
                ),
                "test_mae": float(
                    current_comparison.loc["siamese_closed50_ssa", "test_mae"]
                ),
                "test_rmse": float(
                    current_comparison.loc["siamese_closed50_ssa", "test_rmse"]
                ),
            },
        ]
    )
    ordinary_predictions = pd.read_csv(
        ORDINARY_DIR / "tables" / "optical_reservoir_predictions_test.csv"
    )[["sample_id", "target_date", "cpi_actual", "cpi_predicted"]].rename(
        columns={
            "sample_id": "sample_i_id",
            "cpi_predicted": "cpi_predicted_ordinary",
        }
    )
    current_predictions = pd.read_csv(
        CURRENT_SIAMESE_DIR / "tables" / "ssa_selected_test_predictions.csv"
    )[["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]].rename(
        columns={"cpi_predicted": "cpi_predicted_current_ssa"}
    )
    predictions = ordinary_predictions.merge(
        current_predictions,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    )
    return comparison, predictions


def _save_model(
    bundle: ProjectionBundle,
    config: SelectedConfig,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": bundle.model.state_dict(),
            "network_spec": asdict(bundle.spec),
            "selected_config": asdict(config),
            "state_scaler_mean": bundle.scaler.mean_,
            "state_scaler_scale": bundle.scaler.scale_,
            "target_scale": bundle.target_scale,
            "similarity_tau": bundle.similarity_tau,
            "state_width": int(bundle.scaler.mean_.shape[0]),
        },
        path,
    )


def _save_figures(
    trials: pd.DataFrame,
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    best_by_weight = trials.groupby("similarity_weight", as_index=False)["val_rmse"].min()
    axis.plot(
        best_by_weight["similarity_weight"],
        best_by_weight["val_rmse"],
        marker="o",
    )
    axis.set_xlabel("Similarity-loss weight")
    axis.set_ylabel("Best validation RMSE")
    axis.set_title("Shared-projection similarity constraint selection")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_similarity_weight.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    axis.plot(
        predictions["target_date"],
        predictions["cpi_actual"],
        color="black",
        linewidth=2,
        label="Actual MoM CPI",
    )
    for column, label in (
        ("cpi_predicted_ordinary", "Ordinary optical reservoir"),
        ("cpi_predicted_current_ssa", "Current SSA pairwise Ridge"),
        ("cpi_predicted_projection_delta_only", "Shared projection: delta only"),
        (
            "cpi_predicted_projection_similarity",
            "Shared projection + similarity constraint",
        ),
    ):
        axis.plot(predictions["target_date"], predictions[column], label=label)
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI MoM index (previous month=100)")
    axis.set_title("MoM strict-closed50 shared-projection test predictions")
    axis.tick_params(axis="x", rotation=60)
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
    axis.set_title("Shared-projection test error comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_metric_comparison.png", dpi=180)
    plt.close(fig)


def _markdown_table(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns.astype(str)) + " |"
    divider = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _write_readme(
    output_dir: Path,
    comparison: pd.DataFrame,
    delta_config: SelectedConfig,
    similarity_config: SelectedConfig,
) -> None:
    display = comparison.copy()
    for column in ("val_mae", "val_rmse", "test_mae", "test_rmse"):
        display[column] = display[column].map(lambda value: f"{float(value):.6f}")
    ordinary = comparison.loc[comparison["model"].eq("ordinary_optical_reservoir")].iloc[0]
    primary = comparison.loc[
        comparison["model"].eq("shared_projection_similarity_constrained")
    ].iloc[0]
    mae_change = 100.0 * (primary["test_mae"] - ordinary["test_mae"]) / ordinary[
        "test_mae"
    ]
    rmse_change = 100.0 * (primary["test_rmse"] - ordinary["test_rmse"]) / ordinary[
        "test_rmse"
    ]
    conclusion = (
        "新方案同时降低了测试MAE和RMSE。"
        if mae_change < 0 and rmse_change < 0
        else "新方案没有同时降低测试MAE和RMSE，暂不能宣称全面超过单光储备池。"
    )
    lines = [
        "# 环比严格封闭50样本：共享投影孪生光储备池",
        "",
        "## 模型",
        "",
        "- 光储备池及50维状态完全冻结。",
        "- 目标和参考状态调用同一个可训练投影网络（50→16→8）。",
        "- 关系头预测连续CPI差值，并通过反向计算强制差值反对称。",
        "- 主模型加入连续相似度约束，使嵌入距离反映CPI差异。",
        "- 验证/测试参考均从固定train50支持库中按学习后的嵌入距离选择。",
        "",
        "## 防泄漏与公平边界",
        "",
        "- 训练、验证、测试仍为50/45/47个目标，使用同一份光储备池状态。",
        "- 所有网络权重只使用train50样本对训练。",
        "- 相似度权重、K和注意力温度按三个随机种子的平均验证表现选择。",
        "- 消融结果固定使用最小预声明种子，不挑选测试或验证最优种子。",
        "- 两个配置冻结后才读取测试数组和测试状态。",
        "- 测试参考全部来自train50；参考排序不读取验证或测试标签。",
        "",
        "## 验证冻结配置",
        "",
        f"- 差值回归消融：`{asdict(delta_config)}`",
        f"- 相似度约束主模型：`{asdict(similarity_config)}`",
        "",
        "## 结果",
        "",
        _markdown_table(display),
        "",
        f"相对单光储备池，主模型测试MAE变化 `{mae_change:+.2f}%`，RMSE变化 `{rmse_change:+.2f}%`。",
        "",
        f"**结论：{conclusion}**",
        "",
        "该47个月测试区间此前已在仓库其他实验中被观察，因此本次属于探索性复分析。",
        "后续正式结论应冻结本配置后，在新的未见月份上复验。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_DIR,
    state_dir: Path = STATE_DIR,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    similarity_weights: Iterable[float] = DEFAULT_SIMILARITY_WEIGHTS,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    attention_temperatures: Iterable[float] = DEFAULT_ATTENTION_TEMPERATURES,
    spec: NetworkSpec = NetworkSpec(),
    validation_only: bool = True,
) -> pd.DataFrame:
    """Run validation selection and one post-freeze test evaluation."""
    if not validation_only:
        raise ValueError(
            "the from-scratch shared-projection model is a validation-only "
            "ablation; use the stabilized residual runner for test evaluation"
        )
    seeds = tuple(int(value) for value in seeds)
    similarity_weights = tuple(float(value) for value in similarity_weights)
    k_values = tuple(int(value) for value in k_values)
    attention_temperatures = tuple(
        float(value) for value in attention_temperatures
    )
    if not seeds or not similarity_weights or not k_values or not attention_temperatures:
        raise ValueError("validation search axes must all be non-empty")
    train_pool = build_isolated_closed_train_pool(data_dir)
    if len(train_pool["index"]) != 50:
        raise ValueError("shared-projection experiment requires exactly train50")
    train_pairs = build_causal_train_pairs(train_pool, spec.min_gap_months)

    # This loader intentionally opens train and validation states only.
    validation_split = load_isolated_split(data_dir, "val")
    validation_states = load_validation_state_lookup(state_dir)
    bundles: dict[tuple[int, float], ProjectionBundle] = {}
    validation_outputs: dict[
        tuple[int, float, int, float], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ] = {}
    trials: list[dict[str, object]] = []
    for seed in seeds:
        for similarity_weight in similarity_weights:
            bundle = fit_projection_model(
                train_pool,
                train_pairs,
                validation_states,
                spec,
                seed,
                similarity_weight,
            )
            bundles[(seed, similarity_weight)] = bundle
            for k_references in k_values:
                for temperature in attention_temperatures:
                    key = (seed, similarity_weight, k_references, temperature)
                    outputs = predict_split(
                        bundle,
                        train_pool,
                        validation_split,
                        validation_states,
                        k_references,
                        temperature,
                        spec.min_gap_months,
                    )
                    validation_outputs[key] = outputs
                    trials.append(
                        _selection_row(
                            bundle,
                            outputs[2],
                            k_references,
                            temperature,
                        )
                    )

    delta_row = _select_configuration(trials, constrained=False)
    similarity_row = _select_configuration(trials, constrained=True)
    delta_config = SelectedConfig(
        seed=int(delta_row["seed"]),
        similarity_weight=float(delta_row["similarity_weight"]),
        k_references=int(delta_row["k_references"]),
        attention_temperature=float(delta_row["attention_temperature"]),
    )
    similarity_config = SelectedConfig(
        seed=int(similarity_row["seed"]),
        similarity_weight=float(similarity_row["similarity_weight"]),
        k_references=int(similarity_row["k_references"]),
        attention_temperature=float(similarity_row["attention_temperature"]),
    )
    frozen = {
        "projection_delta_only": delta_config,
        "projection_similarity": similarity_config,
    }

    if validation_only:
        table_dir = output_dir / "tables"
        model_dir = output_dir / "models"
        table_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        trial_table = pd.DataFrame(trials).sort_values(
            ["fitness", "val_rmse", "val_mae"]
        )
        trial_table.to_csv(
            table_dir / "validation_configuration_trials.csv", index=False
        )
        train_pairs.to_csv(table_dir / "training_pairs.csv", index=False)
        selected_rows: list[dict[str, object]] = []
        for name, config, selected_row in (
            ("projection_delta_only", delta_config, delta_row),
            ("projection_similarity", similarity_config, similarity_row),
        ):
            key = (
                config.seed,
                config.similarity_weight,
                config.k_references,
                config.attention_temperature,
            )
            candidates, pair_predictions, predictions = validation_outputs[key]
            candidates.to_csv(
                table_dir / f"{name}_validation_candidates.csv", index=False
            )
            pair_predictions.to_csv(
                table_dir / f"{name}_validation_pair_predictions.csv", index=False
            )
            predictions.to_csv(
                table_dir / f"{name}_validation_predictions.csv", index=False
            )
            bundle = bundles[(config.seed, config.similarity_weight)]
            bundle.training_history.to_csv(
                table_dir / f"{name}_training_history.csv", index=False
            )
            _save_model(bundle, config, model_dir / f"{name}.pt")
            selected_rows.append({"model": name, **selected_row})
        manifest = {
            "experiment": (
                "MoM strict-closed50 shared-projection Siamese validation search"
            ),
            "created_at": datetime.now().astimezone().isoformat(),
            "validation_only": True,
            "test_arrays_or_states_opened": False,
            "pretest_split_loading": "physically isolated train/val files",
            "combined_sample_index_or_mat_opened": False,
            "pretest_files": [
                "sample_index_train.csv",
                "X_train.npy",
                "y_train.npy",
                "sample_index_val.csv",
                "X_val.npy",
                "y_val.npy",
                "states_train.mat",
                "states_val.mat",
            ],
            "network_spec": asdict(spec),
            "accessible_train_windows": 50,
            "validation_targets": 45,
            "num_causal_training_pairs": int(len(train_pairs)),
            "selection_grid": {
                "seeds": list(seeds),
                "similarity_weights": list(similarity_weights),
                "k_references": list(k_values),
                "attention_temperatures": list(attention_temperatures),
            },
            "delta_only_selected_config": asdict(delta_config),
            "similarity_constrained_selected_config": asdict(similarity_config),
        }
        (output_dir / "validation_search_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return pd.DataFrame(selected_rows)

    # The full audit opens test arrays/states, so it deliberately runs only
    # after both validation-selected configurations have been frozen.
    audit = audit_profile(data_dir, state_dir)
    test_split = load_isolated_split(data_dir, "test")
    all_states = load_state_lookup(state_dir)
    test_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for name, config in frozen.items():
        bundle = bundles[(config.seed, config.similarity_weight)]
        test_outputs[name] = predict_split(
            bundle,
            train_pool,
            test_split,
            all_states,
            config.k_references,
            config.attention_temperature,
            spec.min_gap_months,
        )

    comparison, unified = _ordinary_and_current_outputs()
    new_rows: list[dict[str, object]] = []
    for name, config, row in (
        ("projection_delta_only", delta_config, delta_row),
        ("projection_similarity", similarity_config, similarity_row),
    ):
        test_predictions = test_outputs[name][2]
        test_metrics = regression_metrics(
            test_predictions["cpi_actual"], test_predictions["cpi_predicted"]
        )
        model_name = (
            "shared_projection_delta_only"
            if name == "projection_delta_only"
            else "shared_projection_similarity_constrained"
        )
        new_rows.append(
            {
                "model": model_name,
                "trainable_shared_projection": True,
                "similarity_constraint": config.similarity_weight > 0,
                "val_mae": float(row["val_mae"]),
                "val_rmse": float(row["val_rmse"]),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            }
        )
        suffix = (
            "projection_delta_only"
            if name == "projection_delta_only"
            else "projection_similarity"
        )
        selected_predictions = test_predictions[
            ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
        ].rename(columns={"cpi_predicted": f"cpi_predicted_{suffix}"})
        unified = unified.merge(
            selected_predictions,
            on=["sample_i_id", "target_date", "cpi_actual"],
            validate="one_to_one",
        )
    comparison = pd.concat([comparison, pd.DataFrame(new_rows)], ignore_index=True)
    if len(unified) != 47:
        raise ValueError(f"shared-projection comparison has {len(unified)} rows, expected 47")
    for suffix in (
        "ordinary",
        "current_ssa",
        "projection_delta_only",
        "projection_similarity",
    ):
        unified[f"residual_{suffix}"] = (
            unified[f"cpi_predicted_{suffix}"] - unified["cpi_actual"]
        )
        unified[f"absolute_error_{suffix}"] = unified[f"residual_{suffix}"].abs()

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    trial_table = pd.DataFrame(trials).sort_values(
        ["fitness", "val_rmse", "val_mae"]
    )
    trial_table.to_csv(table_dir / "validation_configuration_trials.csv", index=False)
    train_pairs.to_csv(table_dir / "training_pairs.csv", index=False)
    comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    unified.to_csv(table_dir / "test_prediction_comparison.csv", index=False)
    for name, config, selected_row in (
        ("projection_delta_only", delta_config, delta_row),
        ("projection_similarity", similarity_config, similarity_row),
    ):
        key = (
            config.seed,
            config.similarity_weight,
            config.k_references,
            config.attention_temperature,
        )
        val_candidates, val_pairs, val_predictions = validation_outputs[key]
        test_candidates, test_pairs, test_predictions = test_outputs[name]
        val_candidates.to_csv(table_dir / f"{name}_validation_candidates.csv", index=False)
        val_pairs.to_csv(table_dir / f"{name}_validation_pair_predictions.csv", index=False)
        val_predictions.to_csv(table_dir / f"{name}_validation_predictions.csv", index=False)
        test_candidates.to_csv(table_dir / f"{name}_test_candidates.csv", index=False)
        test_pairs.to_csv(table_dir / f"{name}_test_pair_predictions.csv", index=False)
        test_predictions.to_csv(table_dir / f"{name}_test_predictions.csv", index=False)
        bundle = bundles[(config.seed, config.similarity_weight)]
        bundle.training_history.to_csv(
            table_dir / f"{name}_training_history.csv", index=False
        )
        _save_model(bundle, config, model_dir / f"{name}.pt")
        (table_dir / f"{name}_selected_config.json").write_text(
            json.dumps(
                {
                    "selected_config": asdict(config),
                    "validation_metrics": {
                        key: float(selected_row[key])
                        for key in (
                            "val_mae",
                            "val_rmse",
                            "validation_block_rmse_std",
                            "fitness",
                        )
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    _save_figures(trial_table, comparison, unified, figure_dir)
    train_ids = set(train_pool["index"]["sample_id"].astype(int))
    selected_reference_ids: set[int] = set()
    for outputs in test_outputs.values():
        selected_reference_ids.update(outputs[1]["sample_j_id"].astype(int))
    parameter_count = sum(
        parameter.numel()
        for parameter in bundles[
            (similarity_config.seed, similarity_config.similarity_weight)
        ].model.parameters()
        if parameter.requires_grad
    )
    manifest = {
        "experiment": "MoM strict-closed50 shared-projection Siamese optical reservoir",
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_status": (
            "exploratory re-analysis; this historical 47-month test interval was already observed"
        ),
        "profile_audit": audit,
        "frozen_optical_reservoir": True,
        "reservoir_parameters_changed": False,
        "trainable_components": ["shared_projection", "antisymmetric_relation_head"],
        "shared_projection_called_for_both_branches": True,
        "network_spec": asdict(spec),
        "trainable_parameter_count": int(parameter_count),
        "accessible_train_windows": 50,
        "validation_targets": 45,
        "test_targets": 47,
        "num_causal_training_pairs": int(len(train_pairs)),
        "num_training_pair_targets": int(train_pairs["sample_i_id"].nunique()),
        "num_training_pair_references": int(train_pairs["sample_j_id"].nunique()),
        "training_pair_union_covers_train50": bool(
            set(train_pairs["sample_i_id"].astype(int))
            .union(train_pairs["sample_j_id"].astype(int))
            == train_ids
        ),
        "selection_grid": {
            "seeds": list(seeds),
            "similarity_weights": list(similarity_weights),
            "k_references": list(k_values),
            "attention_temperatures": list(attention_temperatures),
        },
        "delta_only_selected_config": asdict(delta_config),
        "similarity_constrained_selected_config": asdict(similarity_config),
        "fitness": "validation RMSE + 0.10 * chronological-block RMSE std",
        "configuration_selection_rule": (
            "minimum mean validation fitness across predeclared seeds"
        ),
        "ablation_seed_rule": "smallest predeclared seed after family selection",
        "all_test_references_inside_train50": selected_reference_ids.issubset(train_ids),
        "target_labels_directly_used_in_per_target_reference_ranking": False,
        "validation_labels_used_for_configuration_selection": True,
        "test_semantic_payload_loaded_after_configuration_freeze": True,
        "test_labels_used_for_selection": False,
        "historical_test_previously_observed": True,
        "model_comparison": comparison.to_dict(orient="records"),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_readme(output_dir, comparison, delta_config, similarity_config)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=NetworkSpec().epochs)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--validation-only",
        action="store_true",
        default=True,
        help="retained for explicitness; this ablation always stops before test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = NetworkSpec(epochs=int(args.epochs))
    comparison = run_pipeline(
        output_dir=args.output_dir,
        seeds=args.seeds,
        spec=spec,
        validation_only=bool(args.validation_only),
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
