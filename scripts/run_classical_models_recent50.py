"""Benchmark compact classical regressors on the recent-50-target split.

This is a diagnostic comparison for the ordinary CPI sliding-window task.  The
last 50 targets of the original training split are used for fitting; the
original validation and test targets remain unchanged.  Hyperparameters are
selected on validation only, and the test split is evaluated once afterward.
"""

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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PROCESSED_DIR, ROOT_DIR


OUTPUT_DIR = ROOT_DIR / "results" / "classical_models_recent50_20260723"
N_RECENT_TRAIN = 50


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def load_recent50() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    sample_index = pd.read_csv(DATA_PROCESSED_DIR / "sample_index.csv")
    arrays = {
        name: np.load(DATA_PROCESSED_DIR / f"{name}.npy")
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    }
    train_index = sample_index.loc[sample_index["split"] == "train"].copy()
    keep = train_index.tail(N_RECENT_TRAIN)["sample_id"].to_numpy(dtype=int)
    # The default arrays are ordered exactly like the split rows, so the last
    # 50 array rows are the same target windows as the last 50 sample IDs.
    arrays["X_train"] = arrays["X_train"][-N_RECENT_TRAIN:]
    arrays["y_train"] = arrays["y_train"][-N_RECENT_TRAIN:]
    arrays["_train_sample_ids"] = keep
    arrays["_val_sample_ids"] = sample_index.loc[
        sample_index["split"] == "val", "sample_id"
    ].to_numpy(dtype=int)
    arrays["_test_sample_ids"] = sample_index.loc[
        sample_index["split"] == "test", "sample_id"
    ].to_numpy(dtype=int)
    return arrays, sample_index


def select_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[object, list[dict[str, object]], np.ndarray, np.ndarray]:
    """Fit a model family and select its small grid using validation RMSE."""
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train_scaled = x_scaler.transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()

    trials: list[dict[str, object]] = []
    candidates: list[tuple[dict[str, object], object, bool]] = []
    if name == "ridge":
        candidates = [
            ({"alpha": alpha}, Ridge(alpha=alpha), True)
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0)
        ]
    elif name == "svr_rbf":
        for C in (0.1, 1.0, 10.0, 100.0):
            for epsilon in (0.05, 0.1, 0.2):
                candidates.append(
                    (
                        {"C": C, "epsilon": epsilon},
                        SVR(kernel="rbf", C=C, epsilon=epsilon, gamma="scale"),
                        True,
                    )
                )
    elif name == "random_forest":
        for depth in (2, 3, None):
            for leaf in (1, 2, 4):
                candidates.append(
                    (
                        {"max_depth": depth, "min_samples_leaf": leaf},
                        RandomForestRegressor(
                            n_estimators=300,
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            max_features=1.0,
                            random_state=42,
                            n_jobs=1,
                        ),
                        False,
                    )
                )
    elif name == "gradient_boosting":
        for depth in (1, 2):
            for learning_rate in (0.03, 0.1):
                candidates.append(
                    (
                        {"max_depth": depth, "learning_rate": learning_rate},
                        GradientBoostingRegressor(
                            n_estimators=100,
                            max_depth=depth,
                            learning_rate=learning_rate,
                            loss="huber",
                            random_state=42,
                        ),
                        False,
                    )
                )
    elif name == "linear_regression":
        candidates = [({}, LinearRegression(), True)]
    else:
        raise ValueError(name)

    for params, model, scaled in candidates:
        if scaled:
            model.fit(X_train_scaled, y_train_scaled)
            val_scaled = model.predict(X_val_scaled)
            val_pred = y_scaler.inverse_transform(val_scaled.reshape(-1, 1)).ravel()
        else:
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
        row = {"model": name, **params, **metrics(y_val, val_pred)}
        trials.append(row)

    best = min(trials, key=lambda row: (row["rmse"], row["mae"]))
    best_model = next(
        model for params, model, _ in candidates if all(best.get(k) == v for k, v in params.items())
    )
    scaled = name in {"ridge", "svr_rbf", "linear_regression"}
    if scaled:
        best_model.fit(X_train_scaled, y_train_scaled)
        train_pred = y_scaler.inverse_transform(
            best_model.predict(X_train_scaled).reshape(-1, 1)
        ).ravel()
        val_pred = y_scaler.inverse_transform(
            best_model.predict(X_val_scaled).reshape(-1, 1)
        ).ravel()
    else:
        best_model.fit(X_train, y_train)
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
    return best_model, trials, train_pred, val_pred


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_dir = OUTPUT_DIR / "tables"
    figure_dir = OUTPUT_DIR / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    arrays, sample_index = load_recent50()
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    models: dict[str, object] = {}
    trial_tables: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []

    # Last-value persistence is an important low-variance baseline for CPI.
    naive_predictions = {
        "train": X_train[:, -1],
        "val": X_val[:, -1],
        "test": X_test[:, -1],
    }
    for split, actual, predicted, ids in (
        ("train", y_train, naive_predictions["train"], arrays["_train_sample_ids"]),
        ("val", y_val, naive_predictions["val"], arrays["_val_sample_ids"]),
        ("test", y_test, naive_predictions["test"], arrays["_test_sample_ids"]),
    ):
        rows.append({"model": "naive_last_value", "split": split, "num_targets": len(actual), **metrics(actual, predicted)})
        all_predictions.append(pd.DataFrame({"model": "naive_last_value", "split": split, "sample_id": ids, "actual": actual, "predicted": predicted, "error": predicted - actual}))

    model_names = ("linear_regression", "ridge", "svr_rbf", "random_forest", "gradient_boosting")
    for name in model_names:
        model, trials, train_pred, val_pred = select_model(name, X_train, y_train, X_val, y_val)
        models[name] = model
        trial_tables.append(pd.DataFrame(trials))
        best = min(trials, key=lambda row: (row["rmse"], row["mae"]))
        scaled = name in {"ridge", "svr_rbf", "linear_regression"}
        if scaled:
            x_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
            test_pred = y_scaler.inverse_transform(
                model.predict(x_scaler.transform(X_test)).reshape(-1, 1)
            ).ravel()
        else:
            test_pred = model.predict(X_test)
        for split, actual, predicted, ids in (
            ("train", y_train, train_pred, arrays["_train_sample_ids"]),
            ("val", y_val, val_pred, arrays["_val_sample_ids"]),
            ("test", y_test, test_pred, arrays["_test_sample_ids"]),
        ):
            rows.append({"model": name, "split": split, "num_targets": len(actual), **metrics(actual, predicted)})
            all_predictions.append(pd.DataFrame({"model": name, "split": split, "sample_id": ids, "actual": actual, "predicted": predicted, "error": predicted - actual}))
        (table_dir / f"{name}_selected_params.json").write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(table_dir / "classical_models_recent50_metrics.csv", index=False)
    pd.concat(trial_tables, ignore_index=True).to_csv(table_dir / "classical_models_recent50_validation_trials.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(table_dir / "classical_models_recent50_predictions.csv", index=False)

    test = pd.concat(all_predictions, ignore_index=True).query("split == 'test'")
    fig, axis = plt.subplots(figsize=(11.5, 5.2))
    actual = test.drop_duplicates("sample_id").sort_values("sample_id")
    axis.plot(actual["sample_id"], actual["actual"], marker="o", linewidth=2, label="Actual CPI")
    for name, group in test.groupby("model"):
        group = group.sort_values("sample_id")
        axis.plot(group["sample_id"], group["predicted"], marker="o", markersize=3, label=name)
    axis.set_title("Recent-50 training targets: classical model test predictions")
    axis.set_xlabel("Global sample ID")
    axis.set_ylabel("CPI")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "classical_models_recent50_test_predictions.png", dpi=180)
    plt.close(fig)

    train_index = sample_index.loc[sample_index["sample_id"].isin(arrays["_train_sample_ids"])]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "train_selection": "last 50 target windows from the original chronological training split",
        "train_target_dates": [str(train_index["target_date"].min()), str(train_index["target_date"].max())],
        "train_sample_ids": [int(arrays["_train_sample_ids"].min()), int(arrays["_train_sample_ids"].max())],
        "validation_targets": int(len(y_val)),
        "test_targets": int(len(y_test)),
        "models": ["naive_last_value", *model_names],
        "test_used_for_selection": False,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(
        "\n".join([
            "# 近50个月训练集：传统模型对照",
            "",
            "同一套CPI滑动窗口数据：训练集取原训练集最后50个目标，验证集45个、测试集47个保持不变。",
            "模型参数只使用验证集选择，测试集仅用于最终评估。该目录独立于单网络光储备池结果。",
            "",
            "- `tables/classical_models_recent50_metrics.csv`：指标",
            "- `tables/classical_models_recent50_validation_trials.csv`：验证集选参记录",
            "- `tables/classical_models_recent50_predictions.csv`：逐样本预测",
            "- `tables/*_selected_params.json`：各模型最终参数",
            "- `figures/classical_models_recent50_test_predictions.png`：测试预测对比图",
        ]) + "\n", encoding="utf-8"
    )
    print(metrics_table[metrics_table["split"] == "test"].sort_values("mae").to_string(index=False))
    print(f"\n独立结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
