"""Run the six compact baselines on the user's continuous YoY windows."""

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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR


PROFILE_ROOT = ROOT_DIR / "matlab" / "optical_reservoir_cpi_yoy_train45_noval_20260807"
DATA_DIR = PROFILE_ROOT / "data"
OUTPUT_DIR = ROOT_DIR / "results" / "baselines_yoy_train45_noval_20260807"


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def candidates(name: str) -> list[tuple[dict[str, object], object, bool]]:
    if name == "linear_regression":
        return [({}, LinearRegression(), True)]
    if name == "ridge":
        return [({"alpha": a}, Ridge(alpha=a), True) for a in (0.01, 0.1, 1.0, 10.0, 100.0)]
    if name == "svr_rbf":
        return [
            ({"C": C, "epsilon": eps}, SVR(kernel="rbf", C=C, epsilon=eps, gamma="scale"), True)
            for C in (0.1, 1.0, 10.0, 100.0)
            for eps in (0.05, 0.1, 0.2)
        ]
    if name == "random_forest":
        return [
            ({"max_depth": depth, "min_samples_leaf": leaf}, RandomForestRegressor(
                n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                max_features=1.0, random_state=42, n_jobs=1), False)
            for depth in (2, 3, None) for leaf in (1, 2, 4)
        ]
    if name == "gradient_boosting":
        return [
            ({"max_depth": depth, "learning_rate": rate}, GradientBoostingRegressor(
                n_estimators=100, max_depth=depth, learning_rate=rate,
                loss="huber", random_state=42), False)
            for depth in (1, 2) for rate in (0.03, 0.1)
        ]
    raise ValueError(name)


def select_by_train_cv(name: str, X: np.ndarray, y: np.ndarray) -> tuple[object, bool, dict[str, object], pd.DataFrame]:
    splitter = TimeSeriesSplit(n_splits=5)
    rows: list[dict[str, object]] = []
    for params, _, scaled in candidates(name):
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
            if scaled:
                x_scaler = StandardScaler().fit(X[train_idx])
                y_scaler = StandardScaler().fit(y[train_idx].reshape(-1, 1))
                model = next(m for p, m, s in candidates(name) if p == params and s == scaled)
                model.fit(x_scaler.transform(X[train_idx]), y_scaler.transform(y[train_idx].reshape(-1, 1)).ravel())
                prediction = y_scaler.inverse_transform(model.predict(x_scaler.transform(X[val_idx])).reshape(-1, 1)).ravel()
            else:
                model = next(m for p, m, s in candidates(name) if p == params and s == scaled)
                model.fit(X[train_idx], y[train_idx])
                prediction = model.predict(X[val_idx])
            fold_scores.append(metrics(y[val_idx], prediction))
        rows.append({"model": name, **params,
                     "cv_mae": float(np.mean([s["mae"] for s in fold_scores])),
                     "cv_rmse": float(np.mean([s["rmse"] for s in fold_scores]))})
    trials = pd.DataFrame(rows).sort_values(["cv_rmse", "cv_mae"])
    best = trials.iloc[0].to_dict()
    for params, model, scaled in candidates(name):
        if all(best.get(key) == value for key, value in params.items()):
            return model, scaled, params, trials
    raise RuntimeError(f"Could not resolve selected {name} parameters")


def main() -> None:
    index = pd.read_csv(DATA_DIR / "sample_index.csv")
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    train_ids = index.loc[index["split"].eq("train"), "sample_id"].to_numpy(int)
    test_ids = index.loc[index["split"].eq("test"), "sample_id"].to_numpy(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_dir = OUTPUT_DIR / "tables"
    figure_dir = OUTPUT_DIR / "figures"
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []

    def add_predictions(name: str, split: str, ids: np.ndarray, actual: np.ndarray, prediction: np.ndarray) -> None:
        meta = index.set_index("sample_id").loc[ids]
        rows.append({"model": name, "split": split, "num_targets": len(actual), **metrics(actual, prediction)})
        prediction_tables.append(pd.DataFrame({
            "model": name, "split": split, "sample_id": ids,
            "target_date": meta["target_date"].to_numpy(),
            "cpi_actual": actual, "cpi_predicted": prediction,
            "error": prediction - actual, "absolute_error": np.abs(prediction - actual),
        }))

    add_predictions("naive_last_value", "train", train_ids, y_train, X_train[:, -1])
    add_predictions("naive_last_value", "test", test_ids, y_test, X_test[:, -1])

    model_names = ("linear_regression", "ridge", "svr_rbf", "random_forest", "gradient_boosting")
    all_trials = []
    for name in model_names:
        model, scaled, params, trials = select_by_train_cv(name, X_train, y_train)
        all_trials.append(trials)
        (table_dir / f"{name}_selected_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if scaled:
            x_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
            model.fit(x_scaler.transform(X_train), y_scaler.transform(y_train.reshape(-1, 1)).ravel())
            train_prediction = y_scaler.inverse_transform(model.predict(x_scaler.transform(X_train)).reshape(-1, 1)).ravel()
            test_prediction = y_scaler.inverse_transform(model.predict(x_scaler.transform(X_test)).reshape(-1, 1)).ravel()
        else:
            model.fit(X_train, y_train)
            train_prediction = model.predict(X_train)
            test_prediction = model.predict(X_test)
        add_predictions(name, "train", train_ids, y_train, train_prediction)
        add_predictions(name, "test", test_ids, y_test, test_prediction)

    metric_table = pd.DataFrame(rows)
    prediction_table = pd.concat(prediction_tables, ignore_index=True)
    metric_table.to_csv(table_dir / "baseline_metrics.csv", index=False)
    prediction_table.to_csv(table_dir / "baseline_predictions.csv", index=False)
    pd.concat(all_trials, ignore_index=True).to_csv(table_dir / "train_time_series_cv_trials.csv", index=False)

    test = prediction_table.loc[prediction_table["split"].eq("test")]
    actual = test.loc[test["model"].eq("naive_last_value")].sort_values("sample_id")
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(actual["target_date"], actual["cpi_actual"], marker="o", linewidth=2, label="Actual YoY CPI")
    for name, group in test.groupby("model"):
        group = group.sort_values("sample_id")
        axis.plot(group["target_date"], group["cpi_predicted"], marker="o", markersize=3, label=name)
    axis.set_title("YoY train45/test47 classical baselines")
    axis.set_xlabel("Target month")
    axis.set_ylabel("CPI YoY index (previous year same month=100)")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_yoy_test_predictions.png", dpi=180)
    plt.close(fig)

    manifest = {
        "experiment": "classical_models_yoy_train45_noval",
        "created_at": datetime.now().astimezone().isoformat(),
        "source": "cpi_data_lastyear=100.csv actual sequence",
        "window_protocol": "continuous previous 12 actual YoY values; one-step-ahead target",
        "split_protocol": "train 2018-09..2022-05 (45), test 2022-06..2026-04 (47), no validation",
        "models": ["naive_last_value", *model_names],
        "hyperparameter_selection": "5-fold chronological TimeSeriesSplit on train45 only",
        "test_used_for_selection": False,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(metric_table.loc[metric_table["split"].eq("test")].sort_values(["rmse", "mae"]).to_string(index=False))
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
