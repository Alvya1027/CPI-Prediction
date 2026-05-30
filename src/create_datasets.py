import sys
import numpy as np
import pandas as pd

# 让 Python 能找到 src 模块
sys.path.append('.')

from src.data_utils import create_sliding_window, split_sequence
from src.config import DATA_PROCESSED_DIR

# ==================== 配置参数 ====================
WINDOW_SIZE = 12      # 用过去12个月预测
HORIZON = 1           # 预测未来1个月
TRAIN_RATIO = 0.7     # 训练集70%
VAL_RATIO = 0.15      # 验证集15% (测试集自动为15%)

print("=" * 50)
print("开始生成滑动窗口数据集...")

# 1. 读取清洗好的 CPI 数据 (使用正确的列名 cpi_yoy)
df = pd.read_csv(DATA_PROCESSED_DIR / "cpi_monthly.csv")
series = df["cpi_yoy"]   # 使用同比指数列
print(f"CPI 数据总行数: {len(series)}")

# 2. 生成特征 X 和标签 y
X, y = create_sliding_window(series, window_size=WINDOW_SIZE, horizon=HORIZON)
print(f"生成样本数: {len(X)}")
print(f"每个样本特征长度: {WINDOW_SIZE}")

# 3. 按时间顺序划分训练集、验证集、测试集
X_train, y_train, X_val, y_val, X_test, y_test = split_sequence(
    X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO
)

print(f"训练集样本数: {len(X_train)}")
print(f"验证集样本数: {len(X_val)}")
print(f"测试集样本数: {len(X_test)}")

# 4. 保存为 .npy 文件
np.save(DATA_PROCESSED_DIR / "X_train.npy", X_train)
np.save(DATA_PROCESSED_DIR / "y_train.npy", y_train)
np.save(DATA_PROCESSED_DIR / "X_val.npy", X_val)
np.save(DATA_PROCESSED_DIR / "y_val.npy", y_val)
np.save(DATA_PROCESSED_DIR / "X_test.npy", X_test)
np.save(DATA_PROCESSED_DIR / "y_test.npy", y_test)

print("\n所有数据集已保存至 data_processed/ 文件夹:")
print("  - X_train.npy, y_train.npy")
print("  - X_val.npy, y_val.npy")
print("  - X_test.npy, y_test.npy")
print("=" * 50)
print("步骤三完成！")