import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# 路径设置
# =========================
TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1. 读取基线预测 (SVR)
# =========================
baseline_path = TABLE_DIR / "baseline_predictions.csv"
baseline_df = pd.read_csv(baseline_path)

# 确定实际值列和SVR列
actual_col_baseline = "actual"  # 根据实际列名调整
svr_col = "SVR"                 # 根据 baseline_predictions.csv 中的列名

# =========================
# 2. 读取光储备池预测（包含三个版本）
# =========================
final_test_path = TABLE_DIR / "final_test_predictions.csv"
test_df = pd.read_csv(final_test_path)

# 这些列名应与 final_test_predictions.csv 一致
actual_col = "cpi_actual"
ordinary_full_col = "cpi_predicted_ordinary_full"        # 212 样本
ordinary_matched_col = "cpi_predicted_ordinary_matched"  # 189 样本
siamese_col = "cpi_predicted_siamese"                    # 189 样本

# 日期列（用于 x 轴）
date_col = "target_date"

# =========================
# 3. 对齐数据（假设所有文件行数一致，且顺序相同）
# =========================
n = min(len(baseline_df), len(test_df))
# 取 test_df 的前 n 行，baseline 的前 n 行
plot_df = pd.DataFrame({
    "date": test_df[date_col].iloc[:n],
    "actual": test_df[actual_col].iloc[:n],
    "SVR": baseline_df[svr_col].iloc[:n],
    "Ordinary_Full_212": test_df[ordinary_full_col].iloc[:n],
    "Ordinary_Matched_189": test_df[ordinary_matched_col].iloc[:n],
    "Siamese_189": test_df[siamese_col].iloc[:n]
})

# =========================
# 4. 计算各模型的 MAE 和 RMSE (测试集)
# =========================
metrics = []
for model in ["SVR", "Ordinary_Full_212", "Ordinary_Matched_189", "Siamese_189"]:
    pred = plot_df[model]
    true = plot_df["actual"]
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    metrics.append({"model": model, "MAE": mae, "RMSE": rmse})

metrics_df = pd.DataFrame(metrics)
print("\n===== 各模型测试集 MAE / RMSE =====")
print(metrics_df.to_string(index=False))

# 可选：保存指标表
metrics_path = TABLE_DIR / "four_model_metrics.csv"
metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
print(f"\n指标表已保存至: {metrics_path}")

# =========================
# 5. 绘图
# =========================
plt.figure(figsize=(16, 7))
plt.plot(plot_df["date"], plot_df["actual"], label="Actual CPI", linewidth=2.5, color="black")
plt.plot(plot_df["date"], plot_df["SVR"], label="SVR (Baseline)", linewidth=2, linestyle="--")
plt.plot(plot_df["date"], plot_df["Ordinary_Full_212"], label="Ordinary Reservoir (212 samples)", linewidth=2, linestyle=":")
plt.plot(plot_df["date"], plot_df["Ordinary_Matched_189"], label="Ordinary Reservoir (189 samples)", linewidth=2, linestyle="-.")
plt.plot(plot_df["date"], plot_df["Siamese_189"], label="Siamese Reservoir (189 samples)", linewidth=2)

plt.title("CPI Prediction: Four Models Comparison", fontsize=14)
plt.xlabel("Target Month", fontsize=12)
plt.ylabel("CPI (YoY, %)", fontsize=12)
plt.xticks(rotation=60)
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

fig_path = FIG_DIR / "four_model_prediction_compare.png"
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"\n对比图已保存至: {fig_path}")

# =========================
# 6. 可选：绘制残差图（偏分析）
# =========================
residual_fig, axes = plt.subplots(2, 2, figsize=(14, 10))
models_plot = ["SVR", "Ordinary_Full_212", "Ordinary_Matched_189", "Siamese_189"]
for ax, model in zip(axes.flatten(), models_plot):
    pred = plot_df[model]
    true = plot_df["actual"]
    residual = pred - true
    ax.scatter(true, residual, alpha=0.6)
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_title(f"{model} Residuals")
    ax.set_xlabel("Actual CPI")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)

residual_fig.tight_layout()
residual_path = FIG_DIR / "four_model_residuals.png"
plt.savefig(residual_path, dpi=300)
plt.show()
print(f"残差图已保存至: {residual_path}")