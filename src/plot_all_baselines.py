import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 路径设置
# =========================
TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 读取基线预测数据
# =========================
baseline_path = TABLE_DIR / "baseline_predictions.csv"
df = pd.read_csv(baseline_path)

# 定义要绘制的模型列（按顺序）
model_columns = [
    "actual",
    "Naive",
    "Ridge",
    "RandomForest",
    "SVR",
    "XGBoost",
    "SARIMAX"
]

# 确保所有列都存在
missing_cols = [col for col in model_columns if col not in df.columns]
if missing_cols:
    raise KeyError(f"以下列在 baseline_predictions.csv 中不存在: {missing_cols}")

# =========================
# 绘图
# =========================
plt.figure(figsize=(16, 7))

# 绘制实际值（黑色粗实线）
plt.plot(df.index, df["actual"], label="Actual CPI", linewidth=2.5, color="black")

# 定义其他模型的颜色和线型（区分度）
styles = [
    {"color": "blue", "linestyle": "--"},      # Naive
    {"color": "green", "linestyle": "-."},     # Ridge
    {"color": "orange", "linestyle": ":"},     # RandomForest
    {"color": "red", "linestyle": "-"},        # SVR
    {"color": "purple", "linestyle": "--"},    # XGBoost
    {"color": "brown", "linestyle": "-."}      # SARIMAX
]

# 依次绘制各模型（跳过 actual 已绘制）
for i, col in enumerate(model_columns[1:]):  # 跳过 'actual'
    plt.plot(
        df.index,
        df[col],
        label=col,
        linewidth=2,
        color=styles[i]["color"],
        linestyle=styles[i]["linestyle"]
    )

# =========================
# 图表装饰
# =========================
plt.title("CPI Prediction: All Baseline Models Comparison", fontsize=14)
plt.xlabel("Test Sample Index", fontsize=12)
plt.ylabel("CPI (YoY, %)", fontsize=12)
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# =========================
# 保存并显示
# =========================
fig_path = FIG_DIR / "baseline_models_comparison.png"
plt.savefig(fig_path, dpi=300)
plt.show()

print(f"对比图已保存至: {fig_path}")