"""Run classical baselines on the shared MoM 50/45/47 window protocol."""

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
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_classical_models_recent50 import metrics, select_model
from src.config import RESULTS_DIR
from src.siamese_mom_closed50_pipeline import DATA_DIR, EXPECTED_SPLITS


OUTPUT_DIR = RESULTS_DIR / "classical_models_mom_closed50_20260730"
MODEL_NAMES = (
    "linear_regression",
    "ridge",
    "svr_rbf",
    "random_forest",
    "gradient_boosting",
)


def load_data(
    data_dir: Path = DATA_DIR,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    index = pd.read_csv(data_dir / "sample_index.csv")
    payload = loadmat(data_dir / "cpi_windows.mat")
    arrays: dict[str, np.ndarray] = {}
    for split, (count, first_date, last_date) in EXPECTED_SPLITS.items():
        rows = index.loc[index["split"].eq(split)].reset_index(drop=True)
        if (
            len(rows) != count
            or str(rows["target_date"].iloc[0]) != first_date
            or str(rows["target_date"].iloc[-1]) != last_date
        ):
            raise ValueError(f"{split} does not match the shared MoM protocol")
        arrays[f"X_{split}"] = np.asarray(payload[f"X_{split}"], dtype=float)
        arrays[f"y_{split}"] = np.asarray(payload[f"y_{split}"], dtype=float).reshape(-1)
        arrays[f"id_{split}"] = rows["sample_id"].to_numpy(dtype=int)
    return arrays, index


def run_classical_models(
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    arrays, index = load_data(data_dir)
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    validation_trials: list[pd.DataFrame] = []

    naive = {
        "train": X_train[:, -1],
        "val": X_val[:, -1],
        "test": X_test[:, -1],
    }
    for split, actual in (("train", y_train), ("val", y_val), ("test", y_test)):
        predicted = naive[split]
        rows.append(
            {
                "model": "naive_last_value",
                "split": split,
                "num_targets": len(actual),
                **metrics(actual, predicted),
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "model": "naive_last_value",
                    "split": split,
                    "sample_id": arrays[f"id_{split}"],
                    "target_date": index.loc[index["split"].eq(split), "target_date"].to_numpy(),
                    "actual": actual,
                    "predicted": predicted,
                    "error": predicted - actual,
                }
            )
        )

    for name in MODEL_NAMES:
        model, trials, train_pred, val_pred = select_model(
            name,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        validation_trials.append(pd.DataFrame(trials))
        best = min(trials, key=lambda row: (row["rmse"], row["mae"]))
        if name in {"ridge", "svr_rbf", "linear_regression"}:
            x_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
            test_pred = y_scaler.inverse_transform(
                model.predict(x_scaler.transform(X_test)).reshape(-1, 1)
            ).reshape(-1)
        else:
            test_pred = model.predict(X_test)
        for split, actual, predicted in (
            ("train", y_train, train_pred),
            ("val", y_val, val_pred),
            ("test", y_test, test_pred),
        ):
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "num_targets": len(actual),
                    **metrics(actual, predicted),
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "split": split,
                        "sample_id": arrays[f"id_{split}"],
                        "target_date": index.loc[index["split"].eq(split), "target_date"].to_numpy(),
                        "actual": actual,
                        "predicted": predicted,
                        "error": predicted - actual,
                    }
                )
            )
        (table_dir / f"{name}_selected_params.json").write_text(
            json.dumps(best, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    metrics_table = pd.DataFrame(rows)
    prediction_table = pd.concat(predictions, ignore_index=True)
    metrics_table.to_csv(table_dir / "classical_model_metrics.csv", index=False)
    prediction_table.to_csv(table_dir / "classical_model_predictions.csv", index=False)
    pd.concat(validation_trials, ignore_index=True).to_csv(
        table_dir / "validation_hyperparameter_trials.csv",
        index=False,
    )

    test = prediction_table.loc[prediction_table["split"].eq("test")]
    fig, axis = plt.subplots(figsize=(11.5, 5.2))
    actual = test.drop_duplicates("sample_id").sort_values("sample_id")
    axis.plot(actual["target_date"], actual["actual"], color="black", linewidth=2, label="Actual MoM CPI")
    for name, group in test.groupby("model"):
        group = group.sort_values("sample_id")
        axis.plot(group["target_date"], group["predicted"], linewidth=1, label=name)
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI MoM index (previous month=100)")
    axis.set_title("Classical baselines on the shared MoM closed50 test split")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "classical_test_predictions.png", dpi=180)
    plt.close(fig)

    manifest = {
        "experiment": "classical_models_mom_closed50",
        "created_at": datetime.now().astimezone().isoformat(),
        "target_scale": "CPI MoM index (previous month=100)",
        "input": "continuous previous 12 actual values",
        "split": {
            name: {
                "num_targets": values[0],
                "first_target_date": values[1],
                "last_target_date": values[2],
            }
            for name, values in EXPECTED_SPLITS.items()
        },
        "models": ["naive_last_value", *MODEL_NAMES],
        "selection_data": "validation only",
        "test_used_for_selection": False,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics_table


def main() -> None:
    table = run_classical_models()
    print(
        table.loc[table["split"].eq("test")]
        .sort_values(["rmse", "mae"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
