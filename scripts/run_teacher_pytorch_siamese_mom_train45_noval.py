"""Run the fixed teacher-style PyTorch Siamese model for MoM CPI."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_teacher_pytorch_siamese_yoy_train45_noval import (
    CONFIG,
    TrainOnlyStandardizer,
    _fit_one_seed,
    _predict_pairs,
)
from src.config import RESULTS_DIR, ROOT_DIR
from src.siamese_reservoir_regression import regression_metrics
from src.siamese_split_isolation import load_isolated_split
from src.teacher_shared_readout_pipeline import (
    _state_matrix,
    build_evaluation_candidates,
    build_train_pairs,
    select_references,
)
from src.twin_state_cache_contract import STATE_PROTOCOL, load_twin_state_splits


PROFILE_ROOT = ROOT_DIR / "matlab" / "optical_reservoir_cpi_mom_train45_noval_20260807"
DATA_DIR = PROFILE_ROOT / "data"
STATE_DIR = PROFILE_ROOT / "states_twin"
SINGLE_RESULT_DIR = RESULTS_DIR / "siamese_optical_mom_teacher_twin_train45_noval_20260807"
OUTPUT_DIR = RESULTS_DIR / "siamese_optical_mom_teacher_pytorch_train45_noval_20260807"


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _load_split(split: str) -> dict[str, object]:
    payload = load_isolated_split(DATA_DIR, split)
    expected = {
        "train": (45, "2018-09", "2022-05"),
        "test": (47, "2022-06", "2026-04"),
    }[split]
    index = payload["index"]
    if len(index) != expected[0]:
        raise ValueError(f"{split} has {len(index)} targets; expected {expected[0]}")
    if str(index["target_date"].iloc[0]) != expected[1] or str(index["target_date"].iloc[-1]) != expected[2]:
        raise ValueError(f"{split} target dates do not match the fixed protocol")
    return payload


def _load_single_baseline() -> tuple[pd.DataFrame, dict[str, float]]:
    metrics = pd.read_csv(SINGLE_RESULT_DIR / "tables" / "test_model_comparison.csv")
    row = metrics.loc[metrics["model"].eq("absolute_only_same_states")].iloc[0]
    predictions = pd.read_csv(SINGLE_RESULT_DIR / "tables" / "test_prediction_comparison.csv")
    result = predictions[
        ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted_absolute"]
    ].rename(columns={"cpi_predicted_absolute": "cpi_predicted_single"})
    if len(result) != 47:
        raise ValueError("single optical-reservoir baseline does not contain test47")
    return result, {"mae": float(row["test_mae"]), "rmse": float(row["test_rmse"])}


def _save_figures(
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    losses: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    axis.plot(
        predictions["target_date"], predictions["cpi_actual"],
        color="black", linewidth=2, label="Actual MoM CPI",
    )
    axis.plot(
        predictions["target_date"], predictions["cpi_predicted_single"],
        label="Single optical reservoir",
    )
    axis.plot(
        predictions["target_date"], predictions["cpi_predicted_siamese_ensemble"],
        label="Teacher PyTorch Siamese (3-seed mean)",
    )
    axis.set_xlabel("Test target month")
    axis.set_ylabel("CPI MoM index (previous month = 100)")
    axis.set_title("MoM CPI: single vs teacher-style PyTorch Siamese")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "mom_pytorch_siamese_prediction_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(comparison))
    width = 0.36
    axis.bar(x - width / 2, comparison["test_mae"], width, label="MAE")
    axis.bar(x + width / 2, comparison["test_rmse"], width, label="RMSE")
    axis.set_xticks(x, comparison["display_name"], rotation=10, ha="right")
    axis.set_ylabel("Test error")
    axis.set_title("MoM CPI test metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "mom_pytorch_siamese_metric_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    for seed, group in losses.groupby("seed", sort=True):
        axis.plot(group["epoch"], group["train_pair_mse"], label=f"seed={seed}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training pair MSE")
    axis.set_title("MoM teacher-style Siamese training curves")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "training_loss_three_seeds.png", dpi=180)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(1)
    train = _load_split("train")
    test = _load_split("test")
    state_lookup, state_audit = load_twin_state_splits(
        STATE_DIR, {"train": train, "test": test}
    )
    if state_audit.get("status") != "passed" or state_audit.get("state_protocol") != STATE_PROTOCOL:
        raise ValueError("explicit Twin state audit failed")

    train_ids = train["index"]["sample_id"].to_numpy(dtype=int)
    test_ids = test["index"]["sample_id"].to_numpy(dtype=int)
    train_states_raw = _state_matrix(train_ids, state_lookup)
    test_states_raw = _state_matrix(test_ids, state_lookup)
    standardizer = TrainOnlyStandardizer.fit(train_states_raw)
    train_states = standardizer.transform(train_states_raw)
    test_states = standardizer.transform(test_states_raw)
    standardized_lookup = {
        int(sample_id): state for sample_id, state in zip(train_ids, train_states)
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
    if not set(selected["sample_j_id"].astype(int)).issubset(set(train_ids.tolist())):
        raise ValueError("a test reference escaped train45")
    if not selected.groupby("sample_i_id").size().eq(CONFIG.k_references).all():
        raise ValueError("a test target lacks exactly K train references")

    table_dir = OUTPUT_DIR / "tables"
    model_dir = OUTPUT_DIR / "models"
    figure_dir = OUTPUT_DIR / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    loss_tables: list[pd.DataFrame] = []
    pair_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int]] = []
    for seed in CONFIG.seeds:
        model, losses = _fit_one_seed(seed, train_states, pair_i, pair_j, train_targets)
        pair_output, predictions = _predict_pairs(
            model, selected, standardized_lookup, seed
        )
        metrics = regression_metrics(
            predictions["cpi_actual"].to_numpy(dtype=float),
            predictions["cpi_predicted"].to_numpy(dtype=float),
        )
        metric_rows.append({"seed": seed, **metrics})
        loss_tables.append(losses)
        pair_tables.append(pair_output)
        prediction_tables.append(predictions)
        torch.save(
            {"state_dict": model.state_dict(), "config": asdict(CONFIG), "seed": seed},
            model_dir / f"teacher_pytorch_siamese_mom_seed{seed}.pt",
        )

    seed_predictions = pd.concat(prediction_tables, ignore_index=True)
    wide = seed_predictions.pivot(
        index=["sample_i_id", "target_date", "cpi_actual"],
        columns="seed", values="cpi_predicted",
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
    single_predictions, single_metrics = _load_single_baseline()
    final_predictions = single_predictions.merge(
        wide, on=["sample_i_id", "target_date", "cpi_actual"], validate="one_to_one"
    )
    for suffix in ("single", "siamese_ensemble"):
        final_predictions[f"residual_{suffix}"] = (
            final_predictions[f"cpi_predicted_{suffix}"] - final_predictions["cpi_actual"]
        )
        final_predictions[f"absolute_error_{suffix}"] = final_predictions[
            f"residual_{suffix}"
        ].abs()

    comparison = pd.DataFrame(
        [
            {
                "model": "single_optical_reservoir_same_states",
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

    losses = pd.concat(loss_tables, ignore_index=True)
    pair_predictions = pd.concat(pair_tables, ignore_index=True)
    per_seed = pd.DataFrame(metric_rows)
    train_pairs.to_csv(table_dir / "train_pair_relations.csv", index=False)
    selected.to_csv(table_dir / "selected_test_references.csv", index=False)
    pair_predictions.to_csv(
        table_dir / "selected_test_pair_predictions_all_seeds.csv", index=False
    )
    seed_predictions.to_csv(table_dir / "test_predictions_all_seeds.csv", index=False)
    final_predictions.to_csv(table_dir / "test_prediction_comparison.csv", index=False)
    per_seed.to_csv(table_dir / "per_seed_test_metrics.csv", index=False)
    comparison.to_csv(table_dir / "test_model_comparison.csv", index=False)
    losses.to_csv(table_dir / "training_loss_curves.csv", index=False)
    np.savez_compressed(
        model_dir / "train_only_state_standardizer.npz",
        mean=standardizer.mean, scale=standardizer.scale,
    )
    _save_figures(final_predictions, comparison, losses, figure_dir)

    manifest = {
        "experiment": "teacher-style PyTorch Siamese on MoM explicit-Twin optical states",
        "created_at": datetime.now().astimezone().isoformat(),
        "source": "cpi_data_lastmonth=100.csv actual MoM sequence",
        "split_protocol": "train 2018-09..2022-05 (45), test 2022-06..2026-04 (47), no validation",
        "architecture": (
            "fixed 50D optical states -> shared Linear-ReLU-BatchNorm-Linear-ReLU-Linear "
            "backbone -> 32D embedding difference -> Linear-ReLU-Linear regression head"
        ),
        "configuration": asdict(CONFIG),
        "configuration_policy": (
            "identical to the predeclared YoY teacher-style run; three seeds form a fixed mean "
            "ensemble; no MoM test metric used to choose a seed or hyperparameter"
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
        "explicit_twin_state_audit": state_audit,
        "per_seed_test_metrics": metric_rows,
        "ensemble_test_metrics": ensemble_metrics,
        "single_optical_reservoir_test_metrics": single_metrics,
    }
    _write_json(OUTPUT_DIR / "experiment_manifest.json", manifest)

    mae_change = (ensemble_metrics["mae"] / single_metrics["mae"] - 1.0) * 100.0
    rmse_change = (ensemble_metrics["rmse"] / single_metrics["rmse"] - 1.0) * 100.0
    readme = f"""# 老师结构 PyTorch 孪生光储备池：环比 train45/test47\n\n本实验使用审计通过的 MATLAB 显式 Twin 唯一窗口状态缓存。50维光储备池状态经过共享 `Linear-ReLU-BatchNorm` 骨干得到32维嵌入，嵌入差值进入非线性回归头预测 CPI 环比差值，再加回训练参考月份的已知 CPI。\n\n- 训练目标：2018-09至2022-05，共45个；无验证集\n- 测试目标：2022-06至2026-04，共47个\n- 训练配对：{len(train_pairs)}条，仅由train45内部派生\n- 测试参考：每个目标5个，全部来自train45，选择不使用测试标签\n- 固定随机种子：{', '.join(str(seed) for seed in CONFIG.seeds)}；正式结果为三种子预测平均\n\n## 测试结果\n\n- 单光储备池同状态基准：MAE={single_metrics['mae']:.6f}，RMSE={single_metrics['rmse']:.6f}\n- 老师结构 PyTorch 孪生（三种子平均）：MAE={ensemble_metrics['mae']:.6f}，RMSE={ensemble_metrics['rmse']:.6f}\n- 相对单光储备池：MAE变化={mae_change:.2f}%，RMSE变化={rmse_change:.2f}%（负数为改善，正数为退化）\n\n网络结构、训练轮数和随机种子均从同比实验原样迁移，没有根据环比测试结果调整。\n"""
    (OUTPUT_DIR / "README.md").write_bytes(readme.encode("utf-8"))
    print("Per-seed test metrics:")
    print(per_seed.to_string(index=False))
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
