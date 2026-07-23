import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import pi

# =========================
# 路径设置
# =========================
TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1. 加载数据
# =========================
final_test_df = pd.read_csv(TABLE_DIR / "final_test_predictions.csv")
baseline_df = pd.read_csv(TABLE_DIR / "baseline_predictions.csv")

print(f"[INFO] final_test_predictions.csv 行数: {len(final_test_df)}")
print(f"[INFO] baseline_predictions.csv 行数: {len(baseline_df)}")

# 定义列名
date_col = "target_date"
actual_col = "cpi_actual"
ordinary_full_col = "cpi_predicted_ordinary_full"
ordinary_matched_col = "cpi_predicted_ordinary_matched"
siamese_col = "cpi_predicted_siamese"
svr_col = "SVR"
xgb_col = "XGBoost"

# 储备池全部数据（47个样本）
actual_full = final_test_df[actual_col].values
dates_full = final_test_df[date_col].values
ordinary_full_pred = final_test_df[ordinary_full_col].values
ordinary_matched_pred = final_test_df[ordinary_matched_col].values
siamese_pred = final_test_df[siamese_col].values

# 共同样本数（用于SVR和XGBoost）
n_common = min(len(final_test_df), len(baseline_df))
print(f"[INFO] 共同样本数: {n_common}")

# 基线数据（取共同样本）
actual_common = final_test_df[actual_col].iloc[:n_common].values
dates_common = final_test_df[date_col].iloc[:n_common].values
svr_pred = baseline_df[svr_col].iloc[:n_common].values
xgb_pred = baseline_df[xgb_col].iloc[:n_common].values

# =========================
# 2. 定义评估函数
# =========================
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def directional_accuracy(y_true, y_pred):
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    mask = true_diff != 0
    if np.sum(mask) == 0:
        return np.nan
    correct = np.sign(true_diff[mask]) == np.sign(pred_diff[mask])
    return np.mean(correct) * 100

def max_error_point(y_true, y_pred, dates):
    errors = np.abs(y_true - y_pred)
    idx = np.argmax(errors)
    return dates[idx], errors[idx], y_true[idx], y_pred[idx]

def high_low_mae(y_true, y_pred, threshold_quantile=0.5):
    diff = np.abs(np.diff(y_true))
    if len(diff) == 0:
        return np.nan, np.nan
    threshold = np.quantile(diff, threshold_quantile)
    high_idx = np.where(diff > threshold)[0] + 1
    low_idx = np.where(diff <= threshold)[0] + 1
    high_idx = high_idx[high_idx < len(y_true)]
    low_idx = low_idx[low_idx < len(y_true)]
    high_mae = mae(y_true[high_idx], y_pred[high_idx]) if len(high_idx) > 0 else np.nan
    low_mae = mae(y_true[low_idx], y_pred[low_idx]) if len(low_idx) > 0 else np.nan
    return high_mae, low_mae

def compute_metrics(y_true, y_pred, dates, model_name, n_samples):
    high_mae, low_mae = high_low_mae(y_true, y_pred)
    max_date, max_err, true_val, pred_val = max_error_point(y_true, y_pred, dates)
    return {
        "model": model_name,
        "n_samples": n_samples,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "Corr": correlation(y_true, y_pred),
        "DirAcc": directional_accuracy(y_true, y_pred),
        "HighMAE": high_mae,
        "LowMAE": low_mae,
        "MaxErrDate": max_date,
        "MaxErr": max_err,
        "MaxTrue": true_val,
        "MaxPred": pred_val
    }

# =========================
# 3. 计算各模型指标
# =========================
metrics_list = []

# 储备池：使用全部47个样本
metrics_list.append(compute_metrics(actual_full, ordinary_full_pred, dates_full,
                                    "Ordinary_Full_212", len(actual_full)))
metrics_list.append(compute_metrics(actual_full, ordinary_matched_pred, dates_full,
                                    "Ordinary_Matched_189", len(actual_full)))
metrics_list.append(compute_metrics(actual_full, siamese_pred, dates_full,
                                    "Siamese_189", len(actual_full)))

# 基线模型：使用共同样本（30个）
if n_common > 0:
    metrics_list.append(compute_metrics(actual_common, svr_pred, dates_common,
                                        "SVR", n_common))
    metrics_list.append(compute_metrics(actual_common, xgb_pred, dates_common,
                                        "XGBoost", n_common))

metrics_df = pd.DataFrame(metrics_list)
print("\n===== 各模型指标（储备池全47样本，基线共同样本） =====")
print(metrics_df[["model", "n_samples", "MAE", "RMSE", "Corr", "DirAcc", "HighMAE", "LowMAE"]].to_string(index=False))

# 保存指标表
metrics_df.to_csv(TABLE_DIR / "advanced_metrics_final.csv", index=False, encoding="utf-8-sig")

# =========================
# 4. 生成图表
# =========================

# 4.1 方向预测准确率柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_df["model"], metrics_df["DirAcc"], color="skyblue")
plt.ylabel("Directional Accuracy (%)")
plt.title("Directional Prediction Accuracy")
plt.ylim(0, 100)
for bar, acc in zip(bars, metrics_df["DirAcc"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{acc:.1f}%", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(FIG_DIR / "directional_accuracy.png", dpi=300)
plt.show()

# 4.2 分段误差双柱对比图
plot_df_seg = metrics_df[["model", "HighMAE", "LowMAE"]].dropna()
if len(plot_df_seg) > 0:
    x = np.arange(len(plot_df_seg))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, plot_df_seg["HighMAE"], width, label="High Volatility", color="coral")
    bars2 = ax.bar(x + width/2, plot_df_seg["LowMAE"], width, label="Low Volatility", color="lightblue")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE")
    ax.set_title("Segment Error: High vs Low Volatility")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df_seg["model"])
    ax.legend()
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "segment_error_comparison.png", dpi=300)
    plt.show()

# 4.3 MAE/RMSE/Corr 表格图
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
table_data = metrics_df[["model", "n_samples", "MAE", "RMSE", "Corr"]].values
col_labels = ["Model", "N", "MAE", "RMSE", "Correlation"]
table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title("Model Performance with Sample Count", y=1.1)
plt.savefig(FIG_DIR / "metric_table_with_corr.png", dpi=300, bbox_inches="tight")
plt.show()

# 4.4 最大误差点案例分析（在预测曲线上标记）
# 构建绘图DataFrame（储备池全47，基线30，但只画共同样本部分，否则基线无法覆盖）
# 为了公平，这里只画共同样本（30）的数据，但储备池的曲线也截取前30个点
if n_common > 0:
    plot_df = pd.DataFrame({
        "date": dates_common,
        "actual": actual_common,
        "SVR": svr_pred,
        "XGBoost": xgb_pred,
        "Ordinary_Full_212": ordinary_full_pred[:n_common],
        "Ordinary_Matched_189": ordinary_matched_pred[:n_common],
        "Siamese_189": siamese_pred[:n_common]
    })
    model_names_plot = ["SVR", "XGBoost", "Ordinary_Full_212", "Ordinary_Matched_189", "Siamese_189"]
else:
    plot_df = pd.DataFrame({
        "date": dates_full,
        "actual": actual_full,
        "Ordinary_Full_212": ordinary_full_pred,
        "Ordinary_Matched_189": ordinary_matched_pred,
        "Siamese_189": siamese_pred
    })
    model_names_plot = ["Ordinary_Full_212", "Ordinary_Matched_189", "Siamese_189"]

n_models = len(model_names_plot)
cols = 2
rows = (n_models + 1) // 2
fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
axes = axes.flatten()
for idx, name in enumerate(model_names_plot):
    ax = axes[idx]
    y_true = plot_df["actual"].values
    y_pred = plot_df[name].values
    dates = plot_df["date"].values
    ax.plot(dates, y_true, label="Actual", color="black", linewidth=1.5)
    ax.plot(dates, y_pred, label=name, linestyle="--", linewidth=1.5)
    max_date, max_err, true_val, pred_val = max_error_point(y_true, y_pred, dates)
    ax.scatter(max_date, pred_val, color="red", s=100, edgecolors="darkred", zorder=5)
    ax.annotate(f"Max error: {max_err:.3f}\nDate: {max_date}",
                xy=(max_date, pred_val), xytext=(max_date, pred_val + 0.5),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=8, color="red")
    ax.set_title(name)
    ax.set_xlabel("Date")
    ax.set_ylabel("CPI")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
for i in range(n_models, len(axes)):
    axes[i].axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "max_error_points.png", dpi=300)
plt.show()

# 4.5 雷达图（综合对比）
def normalize_high_better(values):
    v_min, v_max = values.min(), values.max()
    if v_max == v_min:
        return np.ones_like(values)
    return (values - v_min) / (v_max - v_min)

radar_df = metrics_df[["model", "MAE", "RMSE", "DirAcc", "HighMAE", "LowMAE"]].dropna()
if len(radar_df) > 0:
    # 对误差指标取倒数（越大越好）
    radar_df["MAE_inv"] = 1 / radar_df["MAE"]
    radar_df["RMSE_inv"] = 1 / radar_df["RMSE"]
    radar_df["HighMAE_inv"] = 1 / radar_df["HighMAE"]
    radar_df["LowMAE_inv"] = 1 / radar_df["LowMAE"]

    # 归一化
    for col in ["MAE_inv", "RMSE_inv", "DirAcc", "HighMAE_inv", "LowMAE_inv"]:
        radar_df[col] = normalize_high_better(radar_df[col])

    categories = ['MAE', 'RMSE', 'DirAcc', 'HighMAE', 'LowMAE']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for idx, row in radar_df.iterrows():
        values = row[['MAE_inv', 'RMSE_inv', 'DirAcc', 'HighMAE_inv', 'LowMAE_inv']].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row['model'])
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Radar Chart: Comprehensive Model Comparison")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "radar_chart.png", dpi=300)
    plt.show()

print("\n所有高级分析图表已生成并保存至 results/figures/")
print("指标表已保存至 results/tables/advanced_metrics_final.csv")
print("注意：储备池指标基于全部47个测试样本，SVR/XGBoost基于共同样本（30个）。")