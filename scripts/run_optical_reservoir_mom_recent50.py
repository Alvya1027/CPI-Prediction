"""Train the ordinary optical-reservoir readout on the isolated MoM states."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT_DIR
from src.optical_reservoir_regression import run_training


PROFILE_ROOT = ROOT_DIR / "matlab" / "optical_reservoir_cpi_mom_recent50_20260730"
STATE_DIR = PROFILE_ROOT / "states"
SAMPLE_INDEX = PROFILE_ROOT / "data" / "sample_index.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "optical_reservoir_mom_recent50_20260730"


def save_figures() -> None:
    table_dir = OUTPUT_DIR / "tables"
    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        predictions = pd.read_csv(table_dir / f"optical_reservoir_predictions_{split}.csv")
        fig, axis = plt.subplots(figsize=(10.5, 4.8))
        axis.plot(predictions["target_date"], predictions["cpi_actual"], marker="o", markersize=3, label="Actual MoM CPI")
        axis.plot(predictions["target_date"], predictions["cpi_predicted"], marker="o", markersize=3, label="Optical reservoir prediction")
        axis.set_title(f"Ordinary optical reservoir ({split}, MoM, train targets=50)")
        axis.set_xlabel("Target month")
        axis.set_ylabel("CPI (previous month=100)")
        axis.tick_params(axis="x", rotation=60)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"optical_reservoir_mom_recent50_{split}_predictions.png", dpi=180)
        plt.close(fig)


def main() -> None:
    for path in (STATE_DIR / "states_train.mat", STATE_DIR / "states_val.mat", STATE_DIR / "states_test.mat", SAMPLE_INDEX):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run the Python preparation and MATLAB simulation first.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = run_training(state_dir=STATE_DIR, output_dir=OUTPUT_DIR, sample_index_path=SAMPLE_INDEX)
    save_figures()
    summary_path = OUTPUT_DIR / "tables" / "optical_reservoir_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": "ordinary_optical_reservoir_mom_recent50",
            "created_at": datetime.now().astimezone().isoformat(),
            "target_scale": "CPI MoM index (previous month=100)",
            "split_definition": "train 2014-07..2018-08 (50), val 2018-09..2022-05 (45), test 2022-06..2026-04 (47)",
            "state_dir": str(STATE_DIR),
            "sample_index": str(SAMPLE_INDEX),
            "output_dir": str(OUTPUT_DIR),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(
        "# 环比单网络光储备池（近 50 个月训练）\n\n"
        "本目录是独立的环比实验结果，不覆盖同比实验或原有结果。\n"
        "输入为 `cpi_data_lastmonth=100.csv` 的 actual 序列，每个目标使用前 12 个月窗口。\n"
        "训练/验证/测试目标月份分别为 2014-07..2018-08、2018-09..2022-05、2022-06..2026-04。\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(f"独立环比结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
