import sys
import numpy as np
import pandas as pd

sys.path.append('.')
from src.data_utils import create_sliding_window, split_sequence
from src.config import DATA_PROCESSED_DIR

# 配置
HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# 读取数据
df = pd.read_csv(DATA_PROCESSED_DIR / "cpi_monthly.csv")
series = df["cpi_yoy"]  # 确认列名正确

# 要生成的窗口大小列表
window_sizes = [6, 12, 24]

for ws in window_sizes:
    print(f"\n--- 生成窗口大小 = {ws} 的数据集 ---")
    X, y = create_sliding_window(series, window_size=ws, horizon=HORIZON)
    X_train, y_train, X_val, y_val, X_test, y_test = split_sequence(
        X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO
    )

    # 保存到带窗口大小后缀的文件
    np.save(DATA_PROCESSED_DIR / f"X_train_ws{ws}.npy", X_train)
    np.save(DATA_PROCESSED_DIR / f"y_train_ws{ws}.npy", y_train)
    np.save(DATA_PROCESSED_DIR / f"X_val_ws{ws}.npy", X_val)
    np.save(DATA_PROCESSED_DIR / f"y_val_ws{ws}.npy", y_val)
    np.save(DATA_PROCESSED_DIR / f"X_test_ws{ws}.npy", X_test)
    np.save(DATA_PROCESSED_DIR / f"y_test_ws{ws}.npy", y_test)

    print(f"样本数: {len(X)} | 训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}")

print("\n所有窗口大小的数据集已生成并保存。")