"""Run the teacher-style PyTorch Siamese regressor on YoY reservoir states.

The 50-dimensional optical-reservoir states are fixed MATLAB outputs.  A
shared Linear-ReLU-BatchNorm backbone maps target and reference states to
32-dimensional embeddings.  A nonlinear head predicts their CPI difference;
the known training-reference CPI is then added back.  All architecture and
training settings are fixed before test evaluation, and every test reference
comes from train45.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_teacher_shared_readout_yoy_train45_noval import (
    DATA_DIR,
    STATE_DIR,
    _load_official_single_test,
    _load_split,
    _load_states,
    _write_json,
)
from src.config import RESULTS_DIR
from src.siamese_reservoir_regression import regression_metrics
from src.teacher_shared_readout_pipeline import (
    _state_matrix,
    build_evaluation_candidates,
    build_train_pairs,
    select_references,
)


OUTPUT_DIR = (
    RESULTS_DIR / "siamese_optical_yoy_teacher_pytorch_train45_noval_20260807"
)


@dataclass(frozen=True)
class ExperimentConfig:
    input_dim: int = 50
    hidden_dim: int = 64
    embedding_dim: int = 32
    head_hidden_dim: int = 32
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    min_gap_months: int = 1
    k_references: int = 5
    aggregation: str = "mean"
    seeds: tuple[int, ...] = (42, 52, 62)


CONFIG = ExperimentConfig()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class TeacherSiameseRegressor(nn.Module):
    """Teacher-provided shared backbone adapted from input_dim=10 to 50."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.head_hidden_dim, 1),
        )

    def forward(self, state_i: torch.Tensor, state_j: torch.Tensor) -> torch.Tensor:
        embedding_i = self.backbone(state_i)
        embedding_j = self.backbone(state_j)
        return self.regression_head(embedding_i - embedding_j).squeeze(-1)


@dataclass(frozen=True)
class TrainOnlyStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train_states: np.ndarray) -> "TrainOnlyStandardizer":
        mean = np.mean(train_states, axis=0, dtype=np.float64)
        scale = np.std(train_states, axis=0, dtype=np.float64)
        scale[scale == 0] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, states: np.ndarray) -> np.ndarray:
        values = (np.asarray(states, dtype=np.float64) - self.mean) / self.scale
        if not np.isfinite(values).all():
            raise ValueError("standardized states contain non-finite values")
        return np.asarray(values, dtype=np.float32)


def _fit_one_seed(
    seed: int,
    states: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    targets: np.ndarray,
) -> tuple[TeacherSiameseRegressor, pd.DataFrame]:
    set_seed(seed)
    model = TeacherSiameseRegressor(CONFIG)
    pair_state_i = torch.from_numpy(states[pair_i])
    pair_state_j = torch.from_numpy(states[pair_j])
    pair_delta = torch.from_numpy(
        np.asarray(targets[pair_i] - targets[pair_j], dtype=np.float32)
    )
    dataset = TensorDataset(pair_state_i, pair_state_j, pair_delta)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    rows: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(1, CONFIG.epochs + 1):
        total_squared_error = 0.0
        total_rows = 0
        for state_i, state_j, label_delta in loader:
            predicted_delta = model(state_i, state_j)
            loss = criterion(predicted_delta, label_delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_squared_error += float(loss.item()) * len(label_delta)
            total_rows += len(label_delta)
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_pair_mse": total_squared_error / total_rows,
            }
        )
    return model, pd.DataFrame(rows)


def _predict_pairs(
    model: TeacherSiameseRegressor,
    selected: pd.DataFrame,
    standardized_lookup: dict[int, np.ndarray],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_i = torch.from_numpy(
        np.vstack([standardized_lookup[int(v)] for v in selected["sample_i_id"]])
    )
    state_j = torch.from_numpy(
        np.vstack([standardized_lookup[int(v)] for v in selected["sample_j_id"]])
    )
    model.eval()
    with torch.no_grad():
        predicted_delta = model(state_i, state_j).cpu().numpy().astype(np.float64)
    pair_output = selected.copy()
    pair_output["seed"] = seed
    pair_output["delta_cpi_predicted"] = predicted_delta
    pair_output["cpi_pred_pair"] = (
        pair_output["cpi_j"].to_numpy(dtype=float) + predicted_delta
    )
    pair_output["delta_error"] = (
        pair_output["delta_cpi_predicted"]
        - pair_output["delta_cpi"].to_numpy(dtype=float)
    )
    grouped = (
        pair_output.groupby("sample_i_id", sort=False)
        .agg(
            target_date=("target_i_date", "first"),
            cpi_actual=("cpi_i", "first"),
            cpi_predicted=("cpi_pred_pair", "mean"),
            num_references=("sample_j_id", "size"),
            reference_prediction_std=("cpi_pred_pair", "std"),
        )
        .reset_index()
    )
    grouped["seed"] = seed
    grouped["error"] = grouped["cpi_predicted"] - grouped["cpi_actual"]
    grouped["absolute_error"] = grouped["error"].abs()
    grouped = grouped.sort_values("target_date").reset_index(drop=True)
    return pair_output, grouped


def _save_figures(
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    losses: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    axis.plot(
        predictions["target_date"],
        predictions["cpi_actual"],
        color="black",
        linewidth=2,
        label="Actual YoY CPI",
    )
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_single"],
        label="Single optical reservoir",
    )
    axis.plot(
        predictions["target_date"],
        predictions["cpi_predicted_siamese_ensemble"],
        label="Teacher PyTorch Siamese (3-seed mean)",
    )
    axis.set_xlabel("Test target month")
    axis.set_ylabel("CPI YoY index (previous-year same month = 100)")
    axis.set_title("YoY CPI: single vs teacher-style PyTorch Siamese")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "yoy_pytorch_siamese_prediction_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(comparison))
    width = 0.36
    axis.bar(x - width / 2, comparison["test_mae"], width, label="MAE")
    axis.bar(x + width / 2, comparison["test_rmse"], width, label="RMSE")
    axis.set_xticks(x, comparison["display_name"], rotation=10, ha="right")
    axis.set_ylabel("Test error")
    axis.set_title("YoY CPI test metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "yoy_pytorch_siamese_metric_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    for seed, group in losses.groupby("seed", sort=True):
        axis.plot(group["epoch"], group["train_pair_mse"], label=f"seed={seed}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training pair MSE")
    axis.set_title("Teacher-style Siamese training curves")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "training_loss_three_seeds.png", dpi=180)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(1)
    train = _load_split("train")
    test = _load_split("test")
    state_lookup, state_audit = _load_states({"train": train, "test": test})
    train_ids = train["index"]["sample_id"].to_numpy(dtype=int)
    test_ids = test["index"]["sample_id"].to_numpy(dtype=int)
    train_states_raw = _state_matrix(train_ids, state_lookup)
    test_states_raw = _state_matrix(test_ids, state_lookup)
    standardizer = TrainOnlyStandardizer.fit(train_states_raw)
    train_states = standardizer.transform(train_states_raw)
    test_states = standardizer.transform(test_states_raw)
    standardized_lookup = {
        int(sample_id): state
        for sample_id, state in zip(train_ids, train_states)
    }
    standardized_lookup.update(
        {int(sample_id): state for sample_id, state in zip(test_ids, test_states)}
    )
    train_targets = np.asarray(train["y"], dtype=np.float64).reshape(-1)
    train_pairs, pair_i, pair_j = build_train_pairs(
        train, min_gap_months=CONFIG.min_gap_months
    )
    if len(train_pairs) != 561:
        raise ValueError(f"train45 gap=1 must produce 561 pairs; got {len(train_pairs)}")

    candidates = build_evaluation_candidates(
        test, train, min_gap_months=CONFIG.min_gap_months
    )
    selected = select_references(candidates, CONFIG.k_references)
    train_id_set = set(train_ids.tolist())
    if not set(selected["sample_j_id"].astype(int)).issubset(train_id_set):
        raise ValueError("a test reference escaped train45")
    if not selected.groupby("sample_i_id").size().eq(CONFIG.k_references).all():
        raise ValueError("a test target does not have exactly K references")

    table_dir = OUTPUT_DIR / "tables"
    model_dir = OUTPUT_DIR / "models"
    figure_dir = OUTPUT_DIR / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_losses: list[pd.DataFrame] = []
    all_pair_predictions: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    per_seed_metrics: list[dict[str, float | int]] = []
    for seed in CONFIG.seeds:
        model, losses = _fit_one_seed(
            seed, train_states, pair_i, pair_j, train_targets
        )
        pair_output, predictions = _predict_pairs(
            model, selected, standardized_lookup, seed
        )
        metrics = regression_metrics(
            predictions["cpi_actual"].to_numpy(dtype=float),
            predictions["cpi_predicted"].to_numpy(dtype=float),
        )
        per_seed_metrics.append({"seed": seed, **metrics})
        all_losses.append(losses)
        all_pair_predictions.append(pair_output)
        all_predictions.append(predictions)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(CONFIG),
                "seed": seed,
            },
            model_dir / f"teacher_pytorch_siamese_seed{seed}.pt",
        )

    seed_predictions = pd.concat(all_predictions, ignore_index=True)
    wide = seed_predictions.pivot(
        index=["sample_i_id", "target_date", "cpi_actual"],
        columns="seed",
        values="cpi_predicted",
    ).reset_index()
    wide.columns = [
        f"cpi_predicted_seed{int(column)}" if isinstance(column, (int, np.integer)) else column
        for column in wide.columns
    ]
    seed_columns = [f"cpi_predicted_seed{seed}" for seed in CONFIG.seeds]
    wide["cpi_predicted_siamese_ensemble"] = wide[seed_columns].mean(axis=1)
    ensemble_metrics = regression_metrics(
        wide["cpi_actual"].to_numpy(dtype=float),
        wide["cpi_predicted_siamese_ensemble"].to_numpy(dtype=float),
    )

    single_predictions, single_metrics = _load_official_single_test()
    single_for_merge = single_predictions[
        ["sample_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(
        columns={
            "sample_id": "sample_i_id",
            "cpi_predicted": "cpi_predicted_single",
        }
    )
    final_predictions = single_for_merge.merge(
        wide,
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    )
    for suffix in ("single", "siamese_ensemble"):
        final_predictions[f"residual_{suffix}"] = (
            final_predictions[f"cpi_predicted_{suffix}"]
            - final_predictions["cpi_actual"]
        )
        final_predictions[f"absolute_error_{suffix}"] = final_predictions[
            f"residual_{suffix}"
        ].abs()

    comparison = pd.DataFrame(
        [
            {
                "model": "single_optical_reservoir",
                "display_name": "Single optical reservoir",
                "test_mae": single_metrics["mae"],
                "test_rmse": single_metrics["rmse"],
            },
            {
                "model": "teacher_pytorch_siamese_three_seed_mean",
                "display_name": "Teacher PyTorch Siamese",
                "test_mae": ensemble_metrics["mae"],
                "test_rmse": ensemble_metrics["rmse"],
            },
        ]
    )
    comparison["mae_change_vs_single_pct"] = (
        comparison["test_mae"] / single_metrics["mae"] - 1.0
    ) * 100.0
    comparison["rmse_change_vs_single_pct"] = (
        comparison["test_rmse"] / single_metrics["rmse"] - 1.0
    ) * 100.0

    losses = pd.concat(all_losses, ignore_index=True)
    pair_predictions = pd.concat(all_pair_predictions, ignore_index=True)
    per_seed_table = pd.DataFrame(per_seed_metrics)
    train_pairs.to_csv(table_dir / "train_pair_relations.csv", index=False)
    selected.to_csv(table_dir / "selected_test_references.csv", index=False)
    pair_predictions.to_csv(
        table_dir / "selected_test_pair_predictions_all_seeds.csv", index=False
    )
    seed_predictions.to_csv(table_dir / "test_predictions_all_seeds.csv", index=False)
    final_predictions.to_csv(table_dir / "test_prediction_comparison.csv", index=False)
    per_seed_table.to_csv(table_dir / "per_seed_test_metrics.csv", index=False)
    comparison.to_csv(table_dir / "test_model_comparison.csv", index=False)
    losses.to_csv(table_dir / "training_loss_curves.csv", index=False)
    np.savez_compressed(
        model_dir / "train_only_state_standardizer.npz",
        mean=standardizer.mean,
        scale=standardizer.scale,
    )
    _save_figures(final_predictions, comparison, losses, figure_dir)

    manifest = {
        "experiment": "teacher-style PyTorch Siamese on YoY optical states",
        "created_at": datetime.now().astimezone().isoformat(),
        "source": "cpi_data_lastyear=100.csv actual sequence",
        "split_protocol": "train 2018-09..2022-05 (45), test 2022-06..2026-04 (47), no validation",
        "architecture": (
            "fixed 50D optical states -> shared Linear-ReLU-BatchNorm-Linear-ReLU-Linear "
            "backbone -> 32D embedding difference -> Linear-ReLU-Linear regression head"
        ),
        "configuration": asdict(CONFIG),
        "configuration_policy": (
            "architecture, optimizer and epochs adapted directly from the teacher example; "
            "three seeds declared as a fixed mean ensemble; no seed or hyperparameter selected "
            "from test metrics"
        ),
        "training_targets": 45,
        "derived_training_pairs": int(len(train_pairs)),
        "derived_pair_target_months": int(train_pairs["sample_i_id"].nunique()),
        "test_targets": 47,
        "test_references_per_target": CONFIG.k_references,
        "all_test_references_from_train45": True,
        "test_labels_used_for_reference_selection": False,
        "reservoir_parameters_trained": False,
        "pytorch_backbone_and_regression_head_trained": True,
        "state_standardization_fit_on_train45_only": True,
        "state_audit": state_audit,
        "per_seed_test_metrics": per_seed_metrics,
        "ensemble_test_metrics": ensemble_metrics,
        "single_optical_reservoir_test_metrics": single_metrics,
    }
    _write_json(OUTPUT_DIR / "experiment_manifest.json", manifest)

    mae_change = (ensemble_metrics["mae"] / single_metrics["mae"] - 1.0) * 100.0
    rmse_change = (ensemble_metrics["rmse"] / single_metrics["rmse"] - 1.0) * 100.0
    readme = f"""# 老师结构 PyTorch 孪生光储备池：同比 train45/test47\n\n本实验真正采用老师示例中的共享 `Linear-ReLU-BatchNorm` 骨干网络：50维 MATLAB 光储备池状态经过共享骨干得到32维嵌入，嵌入差值进入非线性回归头预测 CPI 同比差值，再加回训练参考月份的已知 CPI。\n\n- 训练目标：2018-09至2022-05，共45个；无验证集\n- 测试目标：2022-06至2026-04，共47个\n- 训练配对：{len(train_pairs)}条，仅由train45内部派生\n- 测试参考：每个目标5个，全部来自train45，选择不使用测试标签\n- 固定随机种子：{', '.join(str(seed) for seed in CONFIG.seeds)}；最终结果为三个种子的预测平均，不挑选最好种子\n\n## 测试结果\n\n- 单光储备池：MAE={single_metrics['mae']:.6f}，RMSE={single_metrics['rmse']:.6f}\n- 老师结构 PyTorch 孪生（三种子平均）：MAE={ensemble_metrics['mae']:.6f}，RMSE={ensemble_metrics['rmse']:.6f}\n- 相对单光储备池：MAE变化={mae_change:.2f}%，RMSE变化={rmse_change:.2f}%（负数为改善，正数为退化）\n\n该实验使用的是单路连续窗口 MATLAB 状态缓存，目标/参考分支在 PyTorch 中共享非线性骨干；不是 MATLAB 内同时仿真的显式双分支 Twin。\n"""
    (OUTPUT_DIR / "README.md").write_bytes(readme.encode("utf-8"))
    print("Per-seed test metrics:")
    print(per_seed_table.to_string(index=False))
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
