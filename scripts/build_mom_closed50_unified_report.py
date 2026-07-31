"""Build the unified MoM closed50 comparison and Chinese experiment report."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR
from src.siamese_mom_closed50_pipeline import ORDINARY_DIR, OUTPUT_DIR as SIAMESE_DIR


CLASSICAL_DIR = RESULTS_DIR / "classical_models_mom_closed50_20260730"
OUTPUT_DIR = RESULTS_DIR / "mom_closed50_unified_comparison_20260730"


def _percent_change(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def build_report(
    output_dir: Path = OUTPUT_DIR,
    ordinary_dir: Path = ORDINARY_DIR,
    siamese_dir: Path = SIAMESE_DIR,
    classical_dir: Path = CLASSICAL_DIR,
) -> pd.DataFrame:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    siamese = pd.read_csv(siamese_dir / "tables" / "model_comparison.csv")
    siamese["family"] = "optical_reservoir"
    classical_long = pd.read_csv(classical_dir / "tables" / "classical_model_metrics.csv")
    classical_val = classical_long.loc[classical_long["split"].eq("val")].set_index("model")
    classical_test = classical_long.loc[classical_long["split"].eq("test")].set_index("model")
    classical = pd.DataFrame(
        {
            "model": classical_val.index,
            "optimized_by_ssa": False,
            "val_mae": classical_val["mae"],
            "val_rmse": classical_val["rmse"],
            "test_mae": classical_test.loc[classical_val.index, "mae"],
            "test_rmse": classical_test.loc[classical_val.index, "rmse"],
            "family": "classical",
        }
    ).reset_index(drop=True)
    unified = pd.concat([siamese, classical], ignore_index=True)
    ordinary = unified.loc[unified["model"].eq("ordinary_optical_reservoir")].iloc[0]
    unified["test_mae_change_percent_vs_ordinary"] = unified["test_mae"].map(
        lambda value: _percent_change(float(value), float(ordinary["test_mae"]))
    )
    unified["test_rmse_change_percent_vs_ordinary"] = unified["test_rmse"].map(
        lambda value: _percent_change(float(value), float(ordinary["test_rmse"]))
    )
    unified = unified.sort_values(["test_rmse", "test_mae"]).reset_index(drop=True)
    unified.insert(0, "test_rmse_rank", np.arange(1, len(unified) + 1))
    unified.to_csv(table_dir / "all_model_metrics.csv", index=False)

    optical_predictions = pd.read_csv(
        siamese_dir / "tables" / "test_prediction_comparison.csv"
    )
    classical_predictions = pd.read_csv(
        classical_dir / "tables" / "classical_model_predictions.csv"
    ).query("split == 'test'")
    wide = optical_predictions.copy()
    for model, rows in classical_predictions.groupby("model"):
        values = rows[["sample_id", "target_date", "actual", "predicted"]].rename(
            columns={
                "sample_id": "sample_i_id",
                "actual": "cpi_actual",
                "predicted": f"cpi_predicted_{model}",
            }
        )
        wide = wide.merge(
            values,
            on=["sample_i_id", "target_date", "cpi_actual"],
            validate="one_to_one",
        )
    wide.to_csv(table_dir / "all_model_test_predictions.csv", index=False)

    fig, axis = plt.subplots(figsize=(11, 5.2))
    ordered = unified.sort_values("test_rmse")
    x = np.arange(len(ordered))
    width = 0.38
    axis.bar(x - width / 2, ordered["test_mae"], width, label="MAE")
    axis.bar(x + width / 2, ordered["test_rmse"], width, label="RMSE")
    axis.set_xticks(x, ordered["model"], rotation=28, ha="right")
    axis.set_ylabel("Error")
    axis.set_title("Shared MoM closed50 test metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "all_model_test_metrics.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    axis.plot(wide["target_date"], wide["cpi_actual"], color="black", linewidth=2.2, label="Actual")
    axis.plot(wide["target_date"], wide["cpi_predicted_ordinary"], label="Ordinary optical")
    axis.plot(wide["target_date"], wide["cpi_predicted_ssa"], label="SSA Siamese")
    best_classical_name = classical.sort_values(["test_rmse", "test_mae"]).iloc[0]["model"]
    axis.plot(
        wide["target_date"],
        wide[f"cpi_predicted_{best_classical_name}"],
        label=f"Best classical: {best_classical_name}",
    )
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI MoM index (previous month=100)")
    axis.set_title("MoM closed50 test predictions: key models")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "key_model_test_predictions.png", dpi=180)
    plt.close(fig)

    ssa = unified.loc[unified["model"].eq("siamese_closed50_ssa")].iloc[0]
    baseline_siamese = unified.loc[
        unified["model"].eq("siamese_closed50_baseline")
    ].iloc[0]
    best = unified.iloc[0]
    manifest = json.loads(
        (siamese_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    config = manifest["selected_configuration"]
    report = f"""# 环比严格封闭50样本统一实验报告

## 实验口径

- 数据：`cpi_data_lastmonth=100.csv` 的 `actual` 环比序列。
- 输入：连续过去12个月，预测下一个月。
- 训练集：2014-07至2018-08，共50个目标。
- 验证集：2018-09至2022-05，共45个目标。
- 测试集：2022-06至2026-04，共47个目标。
- 单光与孪生光储备池共享完全相同的50维状态、随机掩码和仿真参数。
- 孪生模型只能访问50个训练窗口；验证和测试参考均来自固定训练参考池。
- 所有参数只用验证集选择，测试集在配置冻结后评估一次。

## SSA最终配置

- gap：{config['gap_months']}个月
- 参考窗口数K：{config['k_references']}
- 绝对水平权重β：{config['level_weight']}
- 距离加权指数p：{config['distance_power']}
- 每个差值区间最大配对数M：{config['max_pairs_per_bin']}
- Ridge alpha：{manifest['selected_alpha']}

## 核心结果

单光储备池测试集MAE/RMSE为{ordinary['test_mae']:.6f}/{ordinary['test_rmse']:.6f}。
未优化孪生模型为{baseline_siamese['test_mae']:.6f}/{baseline_siamese['test_rmse']:.6f}。
SSA孪生模型为{ssa['test_mae']:.6f}/{ssa['test_rmse']:.6f}。

相对单光储备池，SSA孪生模型测试MAE变化{ssa['test_mae_change_percent_vs_ordinary']:+.2f}%，
RMSE变化{ssa['test_rmse_change_percent_vs_ordinary']:+.2f}%。
因此本次实验中SSA孪生模型降低了平均绝对误差，但RMSE没有优于单光储备池，不能表述为全面提升。

按测试RMSE排序，全部模型中最优的是`{best['model']}`，RMSE为{best['test_rmse']:.6f}。
传统模型结果用于补充参照；项目的核心公平结论仍是共享状态下单光与孪生光储备池的直接比较。

## 结论

孪生结构在验证集上表现出明显改善，但优势没有稳定迁移到测试集，说明50个训练窗口下仍存在验证集适配和时序分布变化。
当前结果证明联动流程已经完整跑通、数据边界严格、公平性成立；同时也说明SSA不能保证测试集一定优于单光储备池。
后续若继续改进，应预先固定搜索空间或采用滚动验证，不能依据本次测试集结果继续调参。
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "target_scale": "CPI MoM index (previous month=100)",
        "num_models": len(unified),
        "best_test_rmse_model": str(best["model"]),
        "best_test_rmse": float(best["test_rmse"]),
        "ordinary_test_rmse": float(ordinary["test_rmse"]),
        "ssa_siamese_test_rmse": float(ssa["test_rmse"]),
        "test_used_for_selection": False,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return unified


def main() -> None:
    comparison = build_report()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
