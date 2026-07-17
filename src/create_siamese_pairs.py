"""
孪生回归样本对构造

功能：
  为孪生光储备池回归模型构造 (目标窗口 i, 参考窗口 j) 样本对。
  模型预测目标：
      delta_cpi_ij = cpi_i - cpi_j
  然后用已知的 cpi_j 还原目标值：
      y_i_hat = cpi_j + delta_cpi_ij_hat

关键约束：
  1. 参考窗口 j 必须早于目标窗口 i（开始月份差距 >= 12 个月），避免窗口重叠。
  2. j 的下一个月 CPI（cpi_j）在预测时必须是已知的，避免数据泄漏。
  3. 样本对按目标窗口 i 所属的数据集划分 train/val/test。
  4. 训练集按 delta_cpi 的小/中/大三档分层采样，避免训练对集中在单一区间。
  5. 验证集和测试集只按输入窗口距离选择参考，不能用真实目标 CPI 选参考。
  6. 保留相似性判断规则作为分析字段，但 0/1 标签不再作为最终训练目标。

输入：
  data_processed/X_train.npy / y_train.npy
  data_processed/X_val.npy   / y_val.npy
  data_processed/X_test.npy  / y_test.npy
  data_processed/sample_index.csv

输出：
  data_processed/pair_indices_train.csv
  data_processed/pair_indices_val.csv
  data_processed/pair_indices_test.csv

字段说明：
  pair_id          : 样本对编号
  sample_i_id      : 目标窗口在原始样本中的索引
  sample_j_id      : 参考窗口在原始样本中的索引
  x_i_start_date   : 目标窗口起始月份
  x_i_end_date     : 目标窗口结束月份
  target_i_date    : 目标 CPI 月份
  x_j_start_date   : 参考窗口起始月份
  x_j_end_date     : 参考窗口结束月份
  target_j_date    : 参考 CPI 月份
  cpi_i            : 目标下一个月 CPI
  cpi_j            : 参考下一个月 CPI
  delta_cpi        : cpi_i - cpi_j
  delta_bin        : delta_cpi 所属区间，small / medium / large
  window_distance  : 只根据两个输入窗口计算的距离
  selection_method : 参考窗口选择方法
  similar_label    : 可选相似标签（1=相似，0=不相似或含跳变）

用法：
  python -m src.create_siamese_pairs
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data_processed"

# 参数配置
WINDOW_SIZE = 12                    # 每个 CPI 窗口长度
MIN_GAP_MONTHS = WINDOW_SIZE        # 目标窗口与参考窗口至少间隔 12 个月，避免重叠
MAX_PAIRS_PER_BIN = 2               # 每个目标窗口在每个 delta_cpi 区间最多选几个 j
MAX_EVAL_REFERENCES = 5             # 验证/测试每个目标最多选择几个历史参考窗口
TREND_THRESHOLD = 0.05              # 趋势方向阈值
JUMP_ZSCORE_THRESHOLD = 2.0         # 一阶差分跳变检测阈值
RANDOM_SEED = 42                    # 随机种子，保证可复现


def _month_diff(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    """计算两个日期之间相差的月数（d1 - d2）。"""
    return (d1.year - d2.year) * 12 + (d1.month - d2.month)


def _extract_segment_features(segment: np.ndarray) -> Dict[str, float]:
    """提取一个 CPI 窗口的统计特征。

    参数：
        segment: 长度为 WINDOW_SIZE 的一维 CPI 序列

    返回：
        dict，包含 mean, std, trend, start_end_diff
    """
    x = np.asarray(segment, dtype=float)
    trend = float(np.polyfit(np.arange(len(x)), x, 1)[0])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "trend": trend,
        "start_end_diff": float(x[-1] - x[0]),
    }


def _detect_jump(segment: np.ndarray, z_thresh: float = JUMP_ZSCORE_THRESHOLD) -> bool:
    """检测一个 CPI 窗口是否包含异常跳变。

    步骤：
      1. 对片段内部做 z-score 标准化
      2. 计算标准化后序列的一阶差分
      3. 对一阶差分再做 z-score，若任一 |z| > z_thresh 则认为跳变

    参数：
        segment: 长度为 WINDOW_SIZE 的一维 CPI 序列
        z_thresh: 跳变判定阈值

    返回：
        True 表示含跳变，False 表示不含跳变
    """
    x = np.asarray(segment, dtype=float)
    # 片段内部标准化
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return False
    x_scaled = (x - mu) / sigma

    # 一阶差分
    diff = np.diff(x_scaled)
    if len(diff) == 0:
        return False

    # 差分序列的 z-score
    diff_mu = np.mean(diff)
    diff_sigma = np.std(diff)
    if diff_sigma == 0:
        return False
    z_scores = (diff - diff_mu) / diff_sigma

    return bool(np.any(np.abs(z_scores) > z_thresh))


def _window_distance(seg_i: np.ndarray, seg_j: np.ndarray) -> float:
    """计算两个窗口的形状距离，只使用预测时已知的输入值。"""
    x_i = np.asarray(seg_i, dtype=float)
    x_j = np.asarray(seg_j, dtype=float)

    def _zscore(x: np.ndarray) -> np.ndarray:
        std = float(np.std(x))
        if std == 0:
            return x - float(np.mean(x))
        return (x - float(np.mean(x))) / std

    return float(np.linalg.norm(_zscore(x_i) - _zscore(x_j)) / np.sqrt(len(x_i)))


def _trend_direction(trend: float) -> str:
    """根据线性斜率判断趋势方向。"""
    if trend > TREND_THRESHOLD:
        return "up"
    if trend < -TREND_THRESHOLD:
        return "down"
    return "flat"


def _compute_similar_label(
    seg_i: np.ndarray,
    seg_j: np.ndarray,
) -> int:
    """判断两个 CPI 窗口是否相似，返回 1/0。

    返回：
        1  : 相似（同方向且无跳变）
        0  : 趋势不同或任一片段含跳变
    """
    jump_i = _detect_jump(seg_i)
    jump_j = _detect_jump(seg_j)
    if jump_i or jump_j:
        return 0

    trend_i = _extract_segment_features(seg_i)["trend"]
    trend_j = _extract_segment_features(seg_j)["trend"]
    dir_i = _trend_direction(trend_i)
    dir_j = _trend_direction(trend_j)

    return 1 if dir_i == dir_j else 0


def _load_split_data() -> Tuple[Dict, Dict, Dict]:
    """加载训练集、验证集、测试集的窗口数据及索引。"""
    splits = {}
    for split in ["train", "val", "test"]:
        X = np.load(DATA_DIR / f"X_{split}.npy")
        y = np.load(DATA_DIR / f"y_{split}.npy")
        index = pd.read_csv(DATA_DIR / f"sample_index.csv")
        # sample_index.csv 包含所有 split 的样本，需要按 split 字段筛选
        index_split = index[index["split"] == split].copy().reset_index(drop=True)

        # 注意：X/y 的顺序与 index_split 的顺序一致
        splits[split] = {
            "X": X,
            "y": y,
            "index": index_split,
        }
    return splits["train"], splits["val"], splits["test"]


def _build_candidate_pairs(
    target_split: Dict[str, object],
    ref_split: Dict[str, object],
    min_gap_months: int = MIN_GAP_MONTHS,
) -> pd.DataFrame:
    """构造目标窗口与候选参考窗口的所有合法配对。

    约束：
      - 参考窗口必须早于目标窗口 min_gap_months 个月
      - 参考窗口的下一个月 CPI 必须已知（天然满足，因为 ref_split 在目标 split 之前或同 split 内）

    参数：
        target_split: 目标 split 的数据字典
        ref_split:    参考 split 的数据字典
        min_gap_months: 目标窗口起始月份与参考窗口结束月份的最小时间间隔

    返回：
        DataFrame，列包括 sample_i_id, sample_j_id, target_i_date, target_j_date,
                   cpi_i, cpi_j, delta_cpi
    """
    X_i = target_split["X"]
    y_i = target_split["y"]
    idx_i = target_split["index"]

    X_j = ref_split["X"]
    y_j = ref_split["y"]
    idx_j = ref_split["index"]

    start_i = pd.to_datetime(idx_i["x_start_date"])
    end_j = pd.to_datetime(idx_j["x_end_date"])
    start_i_month = (start_i.dt.year * 12 + start_i.dt.month).to_numpy()
    end_j_month = (end_j.dt.year * 12 + end_j.dt.month).to_numpy()

    gaps = start_i_month[:, None] - end_j_month[None, :]
    i_rows, j_rows = np.where(gaps >= min_gap_months)
    if len(i_rows) == 0:
        return pd.DataFrame()

    def _zscore_rows(values: np.ndarray) -> np.ndarray:
        means = values.mean(axis=1, keepdims=True)
        stds = values.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        return (values - means) / stds

    x_i_z = _zscore_rows(np.asarray(X_i, dtype=float))
    x_j_z = _zscore_rows(np.asarray(X_j, dtype=float))
    distances = np.linalg.norm(x_i_z[i_rows] - x_j_z[j_rows], axis=1) / np.sqrt(
        X_i.shape[1]
    )
    cpi_i = np.asarray(y_i, dtype=float).reshape(-1)[i_rows]
    cpi_j = np.asarray(y_j, dtype=float).reshape(-1)[j_rows]

    return pd.DataFrame({
        "sample_i_id": idx_i["sample_id"].to_numpy(dtype=int)[i_rows],
        "sample_j_id": idx_j["sample_id"].to_numpy(dtype=int)[j_rows],
        "x_i_start_date": idx_i["x_start_date"].to_numpy()[i_rows],
        "x_i_end_date": idx_i["x_end_date"].to_numpy()[i_rows],
        "target_i_date": idx_i["target_date"].to_numpy()[i_rows],
        "x_j_start_date": idx_j["x_start_date"].to_numpy()[j_rows],
        "x_j_end_date": idx_j["x_end_date"].to_numpy()[j_rows],
        "target_j_date": idx_j["target_date"].to_numpy()[j_rows],
        "cpi_i": np.round(cpi_i, 4),
        "cpi_j": np.round(cpi_j, 4),
        "delta_cpi": np.round(cpi_i - cpi_j, 4),
        "window_distance": np.round(distances, 6),
    })


def _compute_delta_bin_thresholds(
    pairs: pd.DataFrame,
    quantiles: List[float] = [0.33, 0.67],
) -> Tuple[float, float]:
    """在训练集候选对上计算 delta_cpi 的分位阈值。

    返回：
        (q1, q2)，用于后续 val/test 的区间划分
    """
    deltas = pairs["delta_cpi"].values
    q1, q2 = np.quantile(deltas, quantiles)
    return float(q1), float(q2)


def _assign_delta_bins(
    pairs: pd.DataFrame,
    thresholds: Tuple[float, float],
) -> pd.DataFrame:
    """用训练集阈值把 delta_cpi 划分为 small / medium / large 三档。

    参数：
        pairs: 候选配对 DataFrame
        thresholds: (q1, q2) 两个分位阈值

    返回：
        增加 delta_bin 列的 DataFrame
    """
    q1, q2 = thresholds

    bins = []
    for d in pairs["delta_cpi"].values:
        if d <= q1:
            bins.append("small")
        elif d <= q2:
            bins.append("medium")
        else:
            bins.append("large")

    pairs = pairs.copy()
    pairs["delta_bin"] = bins
    return pairs


def _sample_pairs_by_bin(
    pairs: pd.DataFrame,
    max_per_bin: int = MAX_PAIRS_PER_BIN,
) -> pd.DataFrame:
    """对每个目标窗口，在每个 delta_bin 中随机最多选 max_per_bin 个参考窗口。

    保证：
      - 样本对不过度集中在某个 delta_cpi 区间
      - 每个目标窗口有多个不同区间的参考窗口
    """
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_rows = []

    for sample_i_id, group in pairs.groupby("sample_i_id"):
        for bin_name, bin_group in group.groupby("delta_bin"):
            n = len(bin_group)
            k = min(max_per_bin, n)
            chosen = bin_group.sample(n=k, random_state=rng.integers(0, 2**31))
            sampled_rows.append(chosen)

    return pd.concat(sampled_rows, ignore_index=True)


def _add_similar_labels(
    pairs: pd.DataFrame,
    X_all: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """为每个样本对计算可选的 similar_label。

    参数：
        pairs: 样本对 DataFrame
        X_all: sample_id -> CPI 窗口的映射字典

    返回：
        增加 similar_label 列的 DataFrame
    """
    labels = []
    for _, row in pairs.iterrows():
        seg_i = X_all[int(row["sample_i_id"])]
        seg_j = X_all[int(row["sample_j_id"])]
        labels.append(_compute_similar_label(seg_i, seg_j))

    pairs = pairs.copy()
    pairs["similar_label"] = labels
    return pairs


def _build_all_pairs_for_split(
    split_name: str,
    train_split: Dict[str, object],
    val_split: Dict[str, object],
    test_split: Dict[str, object],
    train_thresholds: Tuple[float, float],
) -> pd.DataFrame:
    """为一个目标 split 构造样本对。

    规则：
      - 训练集的目标窗口 i：参考窗口 j 可以是训练集中早于 i 的任意窗口
      - 验证集的目标窗口 i：参考窗口 j 可以是训练集或验证集中早于 i 的窗口
      - 测试集的目标窗口 i：参考窗口 j 可以是训练集、验证集或测试集中早于 i 的窗口
        但测试集中 j 的下一个月 CPI 必须已知——在同一 split 内，只要 j < i 即可保证
    """
    if split_name == "train":
        target = train_split
        refs = [train_split]
    elif split_name == "val":
        target = val_split
        refs = [train_split, val_split]
    else:
        target = test_split
        refs = [train_split, val_split, test_split]

    # 收集所有候选配对
    candidates = []
    for ref in refs:
        cand = _build_candidate_pairs(target, ref, min_gap_months=MIN_GAP_MONTHS)
        candidates.append(cand)
    candidates = pd.concat(candidates, ignore_index=True)

    # 验证/测试选参考时不能使用 cpi_i 或 delta_cpi，否则会泄漏目标标签。
    # 这里只依据两个已知输入窗口的距离选择最近历史参考。
    sampled = (
        candidates.sort_values(
            ["sample_i_id", "window_distance", "target_j_date", "sample_j_id"]
        )
        .groupby("sample_i_id", as_index=False, group_keys=False)
        .head(MAX_EVAL_REFERENCES)
        .reset_index(drop=True)
    )
    sampled = _assign_delta_bins(sampled, train_thresholds)
    sampled["selection_method"] = "window_distance"

    return sampled


def main() -> None:
    """主函数：生成 train/val/test 三个样本对文件。"""

    print("=" * 60)
    print("孪生回归样本对构造")
    print("=" * 60)
    print(f"窗口大小: {WINDOW_SIZE}")
    print(f"最小时间间隔: {MIN_GAP_MONTHS} 个月")
    print(f"每区间最大参考数: {MAX_PAIRS_PER_BIN}")
    print()

    # 加载数据
    train_split, val_split, test_split = _load_split_data()

    # 构造训练集候选对，并计算分位数阈值（用于后续 val/test 的区间划分）
    print("--- 训练集候选对 ---")
    train_candidates = _build_candidate_pairs(
        train_split, train_split, min_gap_months=MIN_GAP_MONTHS
    )
    train_thresholds = _compute_delta_bin_thresholds(train_candidates)
    train_candidates = _assign_delta_bins(train_candidates, train_thresholds)
    print(f"训练集候选对数量: {len(train_candidates)}")
    print(f"delta_cpi 分位阈值: q1={train_thresholds[0]:.4f}, q2={train_thresholds[1]:.4f}")
    print(f"训练集 delta_cpi 区间分布:")
    print(train_candidates["delta_bin"].value_counts().sort_index())
    print()

    # 按目标窗口分层采样训练集
    train_pairs = _sample_pairs_by_bin(train_candidates, max_per_bin=MAX_PAIRS_PER_BIN)
    train_pairs["selection_method"] = "delta_stratified_train_only"

    # 构造验证集和测试集样本对
    print("--- 验证集样本对 ---")
    val_pairs = _build_all_pairs_for_split(
        "val", train_split, val_split, test_split, train_thresholds
    )
    print(f"验证集样本对数量: {len(val_pairs)}")
    print(val_pairs["delta_bin"].value_counts().sort_index())
    print()

    print("--- 测试集样本对 ---")
    test_pairs = _build_all_pairs_for_split(
        "test", train_split, val_split, test_split, train_thresholds
    )
    print(f"测试集样本对数量: {len(test_pairs)}")
    print(test_pairs["delta_bin"].value_counts().sort_index())
    print()

    # 构造全局 X 映射，用于计算 similar_label
    all_X = {}
    all_y = {}
    for split in [train_split, val_split, test_split]:
        idx_df = split["index"]
        for k, sid in enumerate(idx_df["sample_id"]):
            all_X[int(sid)] = split["X"][k]
            all_y[int(sid)] = split["y"][k]

    # 添加 similar_label（可选字段）
    train_pairs = _add_similar_labels(train_pairs, all_X)
    val_pairs = _add_similar_labels(val_pairs, all_X)
    test_pairs = _add_similar_labels(test_pairs, all_X)

    # 相似标签只用于分析，不能据此删除目标月份；每个目标都必须得到预测。
    train_pairs = train_pairs.reset_index(drop=True)
    val_pairs = val_pairs.reset_index(drop=True)
    test_pairs = test_pairs.reset_index(drop=True)

    # 添加 pair_id
    train_pairs.insert(0, "pair_id", range(len(train_pairs)))
    val_pairs.insert(0, "pair_id", range(len(val_pairs)))
    test_pairs.insert(0, "pair_id", range(len(test_pairs)))

    # 保存
    train_pairs.to_csv(DATA_DIR / "pair_indices_train.csv", index=False, encoding="utf-8-sig")
    val_pairs.to_csv(DATA_DIR / "pair_indices_val.csv", index=False, encoding="utf-8-sig")
    test_pairs.to_csv(DATA_DIR / "pair_indices_test.csv", index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("样本对生成完毕！")
    print("=" * 60)
    print(f"训练集: {len(train_pairs)} 对")
    print(f"验证集: {len(val_pairs)} 对")
    print(f"测试集: {len(test_pairs)} 对")
    print()
    print("输出文件:")
    print("  data_processed/pair_indices_train.csv")
    print("  data_processed/pair_indices_val.csv")
    print("  data_processed/pair_indices_test.csv")

    # 统计 similar_label 分布
    print()
    print("相似标签分布（1=相似，0=不相似）:")
    for name, df in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        counts = df["similar_label"].value_counts().to_dict()
        print(f"  {name}: 相似={counts.get(1, 0)}, 不相似={counts.get(0, 0)}")

    # 生成验证与统计报告
    _generate_report(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        thresholds=train_thresholds,
    )


def _generate_report(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    thresholds: Tuple[float, float],
) -> None:
    """生成样本对统计与验证报告 Markdown 文件。"""

    report_path = ROOT_DIR / "docs" / "siamese_pair_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    q1, q2 = thresholds

    lines = [
        "# 孪生回归样本对构造报告",
        "",
        f"生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 样本对总数",
        "",
        "| 数据集 | 样本对数量 |",
        "|--------|-----------:|",
        f"| 训练集 | {len(train_pairs)} |",
        f"| 验证集 | {len(val_pairs)} |",
        f"| 测试集 | {len(test_pairs)} |",
        f"| **合计** | **{len(train_pairs) + len(val_pairs) + len(test_pairs)}** |",
        "",
        "## 2. CPI 差值区间划分",
        "",
        f"训练集 delta_cpi 分位阈值：q1 = {q1:.4f}，q2 = {q2:.4f}",
        "",
        "| 区间 | 范围 | 训练集 | 验证集 | 测试集 |",
        "|------|------|-------:|-------:|-------:|",
    ]

    # 区间统计
    bins = ["small", "medium", "large"]
    bin_labels = {
        "small": f"delta_cpi ≤ {q1:.4f}",
        "medium": f"{q1:.4f} < delta_cpi ≤ {q2:.4f}",
        "large": f"delta_cpi > {q2:.4f}",
    }

    for b in bins:
        row = [b, bin_labels[b]]
        for df in [train_pairs, val_pairs, test_pairs]:
            row.append(str((df["delta_bin"] == b).sum()))
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")

    large_total = sum((df["delta_bin"] == "large").sum() for df in [train_pairs, val_pairs, test_pairs])
    lines.extend([
        "",
        f"**较大 CPI 差值区间（large）样本总数：{large_total} 对**",
        "",
        "## 3. 相似标签与目标覆盖",
        "",
        "| 数据集 | 相似（1） | 不相似或含跳变（0） | 已覆盖目标数 | 应覆盖目标数 |",
        "|--------|----------:|--------------------:|-------------:|-------------:|",
    ])

    sample_index = pd.read_csv(DATA_DIR / "sample_index.csv")
    for name, df in [("训练集", train_pairs), ("验证集", val_pairs), ("测试集", test_pairs)]:
        sim = (df["similar_label"] == 1).sum()
        dis = (df["similar_label"] == 0).sum()
        split_name = {"训练集": "train", "验证集": "val", "测试集": "test"}[name]
        expected = int((sample_index["split"] == split_name).sum())
        if split_name == "train":
            train_index = sample_index[sample_index["split"] == "train"].copy()
            starts = pd.to_datetime(train_index["x_start_date"])
            first_end = pd.to_datetime(train_index["x_end_date"]).min()
            gaps = (starts.dt.year - first_end.year) * 12 + (starts.dt.month - first_end.month)
            expected = int((gaps >= MIN_GAP_MONTHS).sum())
        covered = int(df["sample_i_id"].nunique())
        lines.append(f"| {name} | {sim} | {dis} | {covered} | {expected} |")

    # 时间泄漏检查
    def _check_leakage(df: pd.DataFrame) -> Tuple[bool, int]:
        df = df.copy()
        df["target_i_date"] = pd.to_datetime(df["target_i_date"])
        df["target_j_date"] = pd.to_datetime(df["target_j_date"])
        df["x_i_start_date"] = pd.to_datetime(df["x_i_start_date"])
        df["x_j_end_date"] = pd.to_datetime(df["x_j_end_date"])

        # 检查 1：target_j < target_i
        cond1 = (df["target_j_date"] < df["target_i_date"]).all()

        # 检查 2：窗口间隔 >= 12 个月
        gaps = (df["x_i_start_date"].dt.year - df["x_j_end_date"].dt.year) * 12 + \
               (df["x_i_start_date"].dt.month - df["x_j_end_date"].dt.month)
        cond2 = (gaps >= MIN_GAP_MONTHS).all()
        min_gap = int(gaps.min())

        return cond1 and cond2, min_gap

    lines.extend([
        "",
        "## 4. 时间泄漏检查",
        "",
        "| 检查项 | 训练集 | 验证集 | 测试集 |",
        "|--------|--------|--------|--------|",
    ])

    lines.append(f"| 无时间泄漏且窗口间隔≥{MIN_GAP_MONTHS}月 | {'通过' if _check_leakage(train_pairs)[0] else '未通过'} | {'通过' if _check_leakage(val_pairs)[0] else '未通过'} | {'通过' if _check_leakage(test_pairs)[0] else '未通过'} |")

    lines.extend([
        "",
        "说明：",
        "- target_j_date 必须早于 target_i_date，确保参考 CPI 在预测时已知。",
        f"- x_i_start_date 与 x_j_end_date 至少间隔 {MIN_GAP_MONTHS} 个月，确保两个 12 个月窗口不重叠。",
        "- 验证集和测试集按 window_distance 选择参考，不使用 cpi_i 或 delta_cpi。",
        "- delta_bin 在验证集和测试集中只用于结果分析，不参与参考选择。",
        "",
        "## 5. 样本对字段说明",
        "",
        "| 字段 | 说明 |",
        "|------|------|",
        "| pair_id | 样本对编号 |",
        "| sample_i_id | 目标窗口在原始样本中的索引 |",
        "| sample_j_id | 参考窗口在原始样本中的索引 |",
        "| x_i_start_date | 目标窗口起始月份 |",
        "| x_i_end_date | 目标窗口结束月份 |",
        "| target_i_date | 目标 CPI 月份 |",
        "| x_j_start_date | 参考窗口起始月份 |",
        "| x_j_end_date | 参考窗口结束月份 |",
        "| target_j_date | 参考 CPI 月份 |",
        "| cpi_i | 目标下一个月 CPI |",
        "| cpi_j | 参考下一个月 CPI |",
        "| delta_cpi | cpi_i - cpi_j（回归标签） |",
        "| delta_bin | delta_cpi 所属区间：small / medium / large |",
        "| window_distance | 由两个已知输入窗口计算的形状距离 |",
        "| selection_method | 训练集为差值分层；验证/测试为窗口距离 |",
        "| similar_label | 可选相似标签：1=相似，0=不相似或含跳变 |",
        "",
        "## 6. 相似性筛选规则",
        "",
        "### 6.1 跳变检测",
        "",
        "1. 对 12 个月 CPI 片段内部做 z-score 标准化。",
        "2. 计算标准化后序列的一阶差分。",
        "3. 对一阶差分再做 z-score，若任一 \\|z\\| > 2.0，则判定该片段含跳变。",
        "",
        "### 6.2 趋势方向判定",
        "",
        "| 方向 | 线性斜率条件 |",
        "|------|-------------|",
        f"| 上升 | trend > {TREND_THRESHOLD} |",
        f"| 下降 | trend < -{TREND_THRESHOLD} |",
        f"| 平稳 | -{TREND_THRESHOLD} ≤ trend ≤ {TREND_THRESHOLD} |",
        "",
        "### 6.3 标签规则",
        "",
        "- similar_label = 1：两个片段趋势方向相同，且都不含跳变。",
        "- similar_label = 0：两个片段趋势方向不同，或任一片段含跳变。",
        "",
        "注意：similar_label 仅用于分析，不删除目标月份，也不作为孪生回归网络的最终训练目标。最终训练目标为 delta_cpi。",
        "",
        "## 7. 生成脚本",
        "",
        "```bash",
        "python -m src.create_siamese_pairs",
        "```",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print()
    print(f"统计与验证报告已保存: {report_path}")


if __name__ == "__main__":
    main()
