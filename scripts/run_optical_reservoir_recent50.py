"""Run the ordinary optical-reservoir readout with a recent 50-target train split.

The original MATLAB reservoir states are kept untouched.  This script writes an
isolated state cache and result directory so the experiment cannot overwrite the
existing 212-target ordinary-reservoir outputs.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

# Allow direct execution via ``python scripts/run_optical_reservoir_recent50.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT_DIR
from src.optical_reservoir_regression import run_training


SOURCE_STATE_DIR = ROOT_DIR / "matlab" / "optical_reservoir_cpi" / "states"
OUTPUT_DIR = ROOT_DIR / "results" / "optical_reservoir_recent50_20260723"
ISOLATED_STATE_DIR = OUTPUT_DIR / "input_states_recent50"
N_RECENT_TRAIN = 50


def _save_isolated_state_cache() -> dict[str, object]:
    """Subset only the old training state file; copy validation/test unchanged."""
    ISOLATED_STATE_DIR.mkdir(parents=True, exist_ok=True)
    sample_index = pd.read_csv(ROOT_DIR / "data_processed" / "sample_index.csv")
    manifest: dict[str, object] = {
        "source_state_dir": str(SOURCE_STATE_DIR),
        "recent_train_targets": N_RECENT_TRAIN,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        source_path = SOURCE_STATE_DIR / f"states_{split}.mat"
        target_path = ISOLATED_STATE_DIR / source_path.name
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        if split != "train":
            shutil.copy2(source_path, target_path)
            data = loadmat(source_path)
            sample_ids = np.asarray(data["sample_id"]).reshape(-1).astype(int)
        else:
            data = loadmat(source_path)
            states = np.asarray(data["state_matrix"], dtype=float)
            sample_ids = np.asarray(data["sample_id"]).reshape(-1).astype(int)
            targets = np.asarray(data["target"]).reshape(-1).astype(float)
            if states.shape[0] != sample_ids.size or targets.size != sample_ids.size:
                raise ValueError("Training state cache has inconsistent row counts")
            if sample_ids.size < N_RECENT_TRAIN:
                raise ValueError("The original training state cache has fewer than 50 rows")

            # State rows are identified by global sample_id, not by an implicit
            # array offset.  Select the latest 50 target windows chronologically.
            order = np.argsort(sample_ids)
            keep = order[-N_RECENT_TRAIN:]
            selected_ids = sample_ids[keep]
            selected_targets = targets[keep]
            selected_states = states[keep]
            savemat(
                target_path,
                {
                    "state_matrix": selected_states,
                    "sample_id": selected_ids.reshape(-1, 1),
                    "target": selected_targets.reshape(-1, 1),
                },
                do_compression=True,
            )
            sample_ids = selected_ids

        metadata = sample_index.set_index("sample_id").loc[sample_ids]
        manifest["splits"][split] = {
            "num_targets": int(len(sample_ids)),
            "sample_id_min": int(sample_ids.min()),
            "sample_id_max": int(sample_ids.max()),
            "target_date_min": str(metadata["target_date"].min()),
            "target_date_max": str(metadata["target_date"].max()),
            "state_width": int(np.asarray(data["state_matrix"]).shape[1]),
        }

    return manifest


def _save_figures() -> None:
    table_dir = OUTPUT_DIR / "tables"
    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        predictions = pd.read_csv(
            table_dir / f"optical_reservoir_predictions_{split}.csv"
        )
        fig, axis = plt.subplots(figsize=(10.5, 4.8))
        axis.plot(
            predictions["target_date"],
            predictions["cpi_actual"],
            marker="o",
            markersize=3,
            label="Actual CPI",
        )
        axis.plot(
            predictions["target_date"],
            predictions["cpi_predicted"],
            marker="o",
            markersize=3,
            label="Recent-50 ordinary optical reservoir",
        )
        axis.set_title(f"Ordinary optical reservoir ({split}, train targets=50)")
        axis.set_xlabel("Target month")
        axis.set_ylabel("CPI")
        axis.tick_params(axis="x", rotation=60)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"optical_reservoir_recent50_{split}_predictions.png", dpi=180)
        plt.close(fig)

    test = pd.read_csv(table_dir / "optical_reservoir_predictions_test.csv")
    fig, axis = plt.subplots(figsize=(10.5, 4.5))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.bar(test["target_date"], test["error"], color="#4472C4")
    axis.set_title("Recent-50 ordinary optical reservoir test residuals")
    axis.set_xlabel("Target month")
    axis.set_ylabel("Prediction error")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "optical_reservoir_recent50_test_residuals.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _save_isolated_state_cache()
    metrics = run_training(
        state_dir=ISOLATED_STATE_DIR,
        output_dir=OUTPUT_DIR,
    )
    _save_figures()

    summary_path = OUTPUT_DIR / "tables" / "optical_reservoir_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": "recent_50_train_targets_only",
            "created_at": datetime.now().astimezone().isoformat(),
            "train_selection": "last 50 target windows from the original chronological training split",
            "validation_and_test": "unchanged original validation/test splits",
            "siamese_run": False,
            "isolated_state_dir": str(ISOLATED_STATE_DIR),
            "isolated_output_dir": str(OUTPUT_DIR),
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# 单网络光储备池：近50个月训练集实验",
                "",
                "本目录是独立实验结果，不覆盖 `results/tables/` 下原有212条训练集结果。",
                "训练集取原始时间顺序训练集的最后50个目标窗口；验证集45个目标、测试集47个目标保持不变。",
                "本次只运行普通/单网络光储备池读出，没有运行孪生模型。",
                "",
                "- `tables/optical_reservoir_metrics.csv`：各划分指标",
                "- `tables/optical_reservoir_predictions_*.csv`：预测与误差",
                "- `tables/optical_reservoir_alpha_selection.csv`：验证集选alpha记录",
                "- `tables/optical_reservoir_run_summary.json`：运行配置",
                "- `data_manifest.json`：实际使用的样本范围与状态维度",
                "- `input_states_recent50/`：本实验独立使用的状态缓存",
                "- `figures/`：验证集、测试集预测图和测试残差图",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(metrics.to_string(index=False))
    print(f"\n独立结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
