"""
样本构造 — 同比/环比一维数据集生成脚本

功能：
  1. 读取 cpi_monthly.csv
  2. 同比：按月份分组（1月~12月），每组是各年同月 CPI 的一维序列
     按时间顺序切分训练集/验证集/测试集（70%/15%/15%）
  3. 环比：所有 CPI 按时间顺序排成一维序列
     按时间顺序切分训练集/验证集/测试集（70%/15%/15%）
  4. 保存原始数据和标准化后的数据（_scaled 后缀）

输出文件命名：
  同比：
    cpi_m01_train.npy  → 1月训练集
    cpi_m01_val.npy    → 1月验证集
    cpi_m01_test.npy   → 1月测试集
    ...（2月~12月同理）
  环比：
    cpi_seq_train.npy  → 环比训练集
    cpi_seq_val.npy    → 环比验证集
    cpi_seq_test.npy   → 环比测试集

  标准化版本：在文件名中插入 _scaled（如 cpi_m01_train_scaled.npy）
  标准化参数：scaler_params_m01.json, ..., scaler_params_seq.json

用法：
  python -m src.create_monthly_datasets
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data_processed"
CSV_PATH = DATA_DIR / "cpi_monthly.csv"

# 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def split_1d_sequence(
    arr: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple:
    """将一维数组按时间顺序切分为训练集、验证集、测试集。

    参数：
        arr: 一维 numpy 数组
        train_ratio: 训练集比例
        val_ratio: 验证集比例

    返回：
        (train, val, test) 三个一维数组
    """
    n = len(arr)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    return arr[:train_end], arr[train_end:val_end], arr[val_end:]


def fit_1d_scaler(train: np.ndarray) -> dict:
    """在一维训练集上拟合 StandardScaler。

    返回：
        dict: {"mean": float, "std": float}
    """
    return {
        "mean": float(np.mean(train)),
        "std": float(np.std(train)),
    }


def apply_1d_scaler(arr: np.ndarray, params: dict) -> np.ndarray:
    """用 scaler 参数 transform 一维数组。"""
    return (arr - params["mean"]) / params["std"]


def inverse_1d_scaler(arr: np.ndarray, params: dict) -> np.ndarray:
    """将标准化后的一维数组还原。"""
    return arr * params["std"] + params["mean"]


def main() -> None:
    """主函数：生成同比和环比数据集。"""

    print("=" * 60)
    print("同比 / 环比 一维数据集生成")
    print("=" * 60)
    print(f"数据源: {CSV_PATH}")
    print(f"划分比例: 训练 {TRAIN_RATIO:.0%} / 验证 {VAL_RATIO:.0%} / 测试 {1 - TRAIN_RATIO - VAL_RATIO:.0%}")
    print()

    # 读取数据
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # ============================================
    # 1. 同比：按月份分组
    # ============================================
    print("--- 同比（按月份分组）---")
    print()

    for month in range(1, 13):
        # 筛选该月所有数据，按时间排序
        monthly = df[df["month"] == month].sort_values("date")
        cpi_values = monthly["cpi"].values  # 一维数组

        # 切分
        train, val, test = split_1d_sequence(cpi_values)

        # 保存原始数据
        np.save(DATA_DIR / f"cpi_m{month:02d}_train.npy", train)
        np.save(DATA_DIR / f"cpi_m{month:02d}_val.npy", val)
        np.save(DATA_DIR / f"cpi_m{month:02d}_test.npy", test)

        # 标准化（只对训练集拟合）
        scaler_params = fit_1d_scaler(train)
        train_scaled = apply_1d_scaler(train, scaler_params)
        val_scaled = apply_1d_scaler(val, scaler_params)
        test_scaled = apply_1d_scaler(test, scaler_params)

        # 保存标准化数据
        np.save(DATA_DIR / f"cpi_m{month:02d}_train_scaled.npy", train_scaled)
        np.save(DATA_DIR / f"cpi_m{month:02d}_val_scaled.npy", val_scaled)
        np.save(DATA_DIR / f"cpi_m{month:02d}_test_scaled.npy", test_scaled)

        # 保存 scaler 参数
        import json

        scaler_file = DATA_DIR / f"scaler_params_m{month:02d}.json"
        with open(scaler_file, "w", encoding="utf-8") as f:
            json.dump(scaler_params, f, ensure_ascii=False, indent=2)

        # 打印信息
        years = monthly["year"].values
        print(
            f"  {month:2d}月: {len(cpi_values)} 个点 "
            f"({years[0]}~{years[-1]}) → "
            f"训练 {len(train)} / 验证 {len(val)} / 测试 {len(test)}"
        )
        print(
            f"       原始值范围 [{train.min():.2f}, {train.max():.2f}], "
            f"标准化后均值={train_scaled.mean():.6f}, 标准差={train_scaled.std():.6f}"
        )

    print()

    # ============================================
    # 2. 环比：所有数据按时间排成一维
    # ============================================
    print("--- 环比（时间序列）---")

    df_sorted = df.sort_values("date")
    cpi_seq = df_sorted["cpi"].values  # 一维数组

    train, val, test = split_1d_sequence(cpi_seq)

    # 保存原始数据
    np.save(DATA_DIR / "cpi_seq_train.npy", train)
    np.save(DATA_DIR / "cpi_seq_val.npy", val)
    np.save(DATA_DIR / "cpi_seq_test.npy", test)

    # 标准化
    scaler_params = fit_1d_scaler(train)
    train_scaled = apply_1d_scaler(train, scaler_params)
    val_scaled = apply_1d_scaler(val, scaler_params)
    test_scaled = apply_1d_scaler(test, scaler_params)

    np.save(DATA_DIR / "cpi_seq_train_scaled.npy", train_scaled)
    np.save(DATA_DIR / "cpi_seq_val_scaled.npy", val_scaled)
    np.save(DATA_DIR / "cpi_seq_test_scaled.npy", test_scaled)

    import json

    scaler_file = DATA_DIR / "scaler_params_seq.json"
    with open(scaler_file, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, ensure_ascii=False, indent=2)

    print(
        f"  环比: {len(cpi_seq)} 个点 "
        f"({df_sorted['date'].iloc[0].strftime('%Y-%m')}~"
        f"{df_sorted['date'].iloc[-1].strftime('%Y-%m')}) → "
        f"训练 {len(train)} / 验证 {len(val)} / 测试 {len(test)}"
    )
    print(
        f"       原始值范围 [{train.min():.2f}, {train.max():.2f}], "
        f"标准化后均值={train_scaled.mean():.6f}, 标准差={train_scaled.std():.6f}"
    )

    print()
    print("=" * 60)
    print("全部生成完毕！")
    print("=" * 60)

    # 列出所有生成的文件
    print("\n生成的文件列表:")
    for f in sorted(DATA_DIR.glob("cpi_m*_*.npy")):
        print(f"  {f.name}")
    for f in sorted(DATA_DIR.glob("cpi_seq_*.npy")):
        print(f"  {f.name}")
    for f in sorted(DATA_DIR.glob("scaler_params_m*.json")):
        print(f"  {f.name}")
    for f in sorted(DATA_DIR.glob("scaler_params_seq*.json")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()