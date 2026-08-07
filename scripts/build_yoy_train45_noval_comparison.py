"""Combine the single-reservoir and YoY baseline test results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR


OUT = ROOT_DIR / "results" / "yoy_train45_noval_unified_comparison_20260807"
RESERVOIR = ROOT_DIR / "results" / "optical_reservoir_yoy_train45_noval_20260807"
BASELINES = ROOT_DIR / "results" / "baselines_yoy_train45_noval_20260807"


def main() -> None:
    table_dir = OUT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    reservoir = pd.read_csv(RESERVOIR / "tables" / "optical_reservoir_metrics.csv")
    reservoir["model"] = reservoir["model"].replace({"single_optical_reservoir": "单网络光储备池"})
    baselines = pd.read_csv(BASELINES / "tables" / "baseline_metrics.csv")
    names = {
        "naive_last_value": "Naive（上期值）",
        "linear_regression": "线性回归",
        "ridge": "Ridge回归",
        "svr_rbf": "SVR（RBF）",
        "random_forest": "随机森林",
        "gradient_boosting": "梯度提升树",
    }
    baselines["model"] = baselines["model"].replace(names)
    metrics = pd.concat([reservoir, baselines], ignore_index=True)
    metrics["target_scale"] = "同比指数（上年同月=100）"
    metrics["window_protocol"] = "actual序列连续前12个月预测下1个月"
    metrics = metrics[["model", "split", "num_targets", "mae", "rmse", "target_scale", "window_protocol"]]
    metrics.to_csv(table_dir / "all_model_metrics.csv", index=False)
    test = metrics.loc[metrics["split"].eq("test")].sort_values(["rmse", "mae"]).reset_index(drop=True)
    test.insert(0, "rank_by_test_rmse", range(1, len(test) + 1))
    test.to_csv(table_dir / "test_ranking.csv", index=False)

    reservoir_pred = pd.read_csv(RESERVOIR / "tables" / "optical_reservoir_predictions.csv")
    baseline_pred = pd.read_csv(BASELINES / "tables" / "baseline_predictions.csv")
    predictions = pd.concat([reservoir_pred, baseline_pred], ignore_index=True)
    predictions["model"] = predictions["model"].replace({"single_optical_reservoir": "单网络光储备池", **names})
    predictions.to_csv(table_dir / "all_model_predictions.csv", index=False)

    manifest = {
        "experiment": "unified YoY train45/test47 comparison",
        "source": "cpi_data_lastyear=100.csv actual sequence",
        "window_protocol": "continuous previous 12 actual values; one-step-ahead target",
        "split_protocol": "train 2018-09..2022-05 (45), test 2022-06..2026-04 (47), no validation",
        "models": list(test["model"]),
        "test_used_for_selection": False,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 同比 train45/test47 单网络与基线统一对比\n\n"
        "数据来自 `cpi_data_lastyear=100.csv` 的 `actual` 序列；每个样本使用连续前12个月预测下1个月。\n"
        "训练目标为2018-09至2022-05（45个），测试目标为2022-06至2026-04（47个），不设置验证集。\n"
        "基线超参数和单网络读出均通过训练集内部时间序列交叉验证确定，测试集只用于最终评估。\n",
        encoding="utf-8",
    )
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
