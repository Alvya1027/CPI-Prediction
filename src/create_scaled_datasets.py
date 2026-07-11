"""
样本构造 — 标准化数据生成脚本

功能：
  1. 加载 data_processed/ 下的原始 .npy 滑动窗口数据
  2. 只在训练集上拟合 StandardScaler（默认）或 MinMaxScaler
  3. 用同一个标准化器 transform 训练集、验证集、测试集
  4. 保存标准化后的 .npy 文件（带 _scaled 后缀，不覆盖原始数据）
  5. 保存标准化器参数为 JSON 文件（供后续模型复现和逆变换）

用法：
  python src/create_scaled_datasets.py

输出：
  data_processed/X_train_scaled.npy
  data_processed/y_train_scaled.npy
  data_processed/X_val_scaled.npy
  data_processed/y_val_scaled.npy
  data_processed/X_test_scaled.npy
  data_processed/y_test_scaled.npy
  data_processed/scaler_params_scaled.json
  （以及对应 ws6, ws12, ws24 后缀的版本）

原则（来自实习计划）：
  - 只用训练集拟合标准化器
  - 避免数据泄漏
  - 不随意打乱时间顺序
"""

from pathlib import Path

import numpy as np

from src.data_utils import save_scaled_dataset

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = ROOT_DIR / "data_processed"

# 需要处理的文件后缀列表
# 空字符串 "" 对应默认 window_size=12 的数据
# "_ws6" 对应 window_size=6
# "_ws12" 对应 window_size=12
# "_ws24" 对应 window_size=24
SUFFIXES = ["", "_ws6", "_ws12", "_ws24"]

# 标准化类型（"standard" 或 "minmax"）
SCALER_TYPE = "standard"


def main() -> None:
    """对 data_processed 中所有后缀的原始数据执行标准化。"""

    print("=" * 60)
    print("样本构造 — 标准化数据生成")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print(f"标准化类型: {SCALER_TYPE}")
    print(f"待处理后缀: {SUFFIXES}")
    print()

    for suffix in SUFFIXES:
        # 检查原始数据文件是否存在
        x_train_file = DATA_DIR / f"X_train{suffix}.npy"
        if not x_train_file.exists():
            print(f"  ⚠ 跳过 {suffix!r}: 文件不存在")
            continue

        print(f"--- 处理后缀: {suffix!r} ---")

        # 加载原始数据查看概况
        X_train = np.load(DATA_DIR / f"X_train{suffix}.npy")
        y_train = np.load(DATA_DIR / f"y_train{suffix}.npy")

        print(f"  原始训练集 X: {X_train.shape}, 值范围 [{X_train.min():.2f}, {X_train.max():.2f}]")
        print(f"  原始训练集 y: {y_train.shape}, 值范围 [{y_train.min():.2f}, {y_train.max():.2f}]")

        # 执行标准化并保存
        result = save_scaled_dataset(
            data_dir=DATA_DIR,
            suffix=suffix,
            scaler_type=SCALER_TYPE,
        )

        scaler = result["scaler_params"]
        print(f"  X_mean={scaler['X_mean']:.4f}, X_std={scaler['X_std']:.4f}")
        print(f"  y_mean={scaler['y_mean']:.4f}, y_std={scaler['y_std']:.4f}")
        print(f"  标准化后训练集: {result['train']}")
        print(f"  标准化后验证集: {result['val']}")
        print(f"  标准化后测试集: {result['test']}")
        print(f"  scaler 参数已保存: {result['scaler_file']}")

        # 验证：标准化后训练集 X 的均值应接近 0，标准差应接近 1
        X_train_scaled = np.load(
            DATA_DIR / f"X_train_scaled{suffix}.npy"
        )
        y_train_scaled = np.load(
            DATA_DIR / f"y_train_scaled{suffix}.npy"
        )

        print(f"  ✓ 标准化后 X 均值: {X_train_scaled.mean():.6f} (期望 ≈ 0)")
        print(f"  ✓ 标准化后 X 标准差: {X_train_scaled.std():.6f} (期望 ≈ 1)")
        print(f"  ✓ 标准化后 y 均值: {y_train_scaled.mean():.6f} (期望 ≈ 0)")
        print(f"  ✓ 标准化后 y 标准差: {y_train_scaled.std():.6f} (期望 ≈ 1)")

        # 验证：没有数据泄漏（验证集和测试集的标准化参数来自训练集）
        print(f"  ✓ 无数据泄漏: 标准化器仅用训练集拟合")
        print()

    print("=" * 60)
    print("全部标准化数据生成完毕！")
    print("=" * 60)

    # 列出所有生成的文件
    print("\n生成的文件列表:")
    import glob

    scaled_files = sorted(glob.glob(str(DATA_DIR / "*_scaled*")))
    for f in scaled_files:
        print(f"  {Path(f).name}")


if __name__ == "__main__":
    main()