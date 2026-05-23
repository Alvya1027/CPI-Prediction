import pandas as pd

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR


RAW_FILE = DATA_RAW_DIR / "cpi_raw.xlsx"
OUTPUT_FILE = DATA_PROCESSED_DIR / "cpi_monthly.csv"


def parse_month(x):
    """
    将不同格式的月份统一转换为 pandas datetime 格式。

    支持格式示例：
    202604
    2026-04
    2026/04
    2026年4月
    2026年04月
    """
    x = str(x).strip()
    x = x.replace(".0", "")
    x = x.replace("年", "-")
    x = x.replace("月", "")
    x = x.replace("/", "-")

    # 处理 202604 这种格式
    if len(x) == 6 and x.isdigit():
        return pd.to_datetime(x, format="%Y%m")

    return pd.to_datetime(x)


def main():
    # 1. 读取原始 Excel 数据
    df = pd.read_excel(RAW_FILE)

    print("原始数据前 5 行：")
    print(df.head())

    print("\n原始列名：")
    print(df.columns)

    # 2. 去掉列名前后的空格
    df.columns = df.columns.astype(str).str.strip()

    # 3. 统一列名
    # 如果你的 Excel 列名和下面不一致，就在这里补充
    rename_dict = {
        # 日期列
        "月份": "date",
        "日期": "date",
        "时间": "date",
        "date": "date",
        "Date": "date",

        # CPI 同比增长率列，例如 0.3、1.2、-0.5
        "CPI同比增长率": "cpi_yoy_growth",
        "CPI同比": "cpi_yoy_growth",
        "居民消费价格指数同比增长率": "cpi_yoy_growth",
        "同比增长率": "cpi_yoy_growth",
        "年增率": "cpi_yoy_growth",
        "中国-居民消费价格指数(年增率)": "cpi_yoy_growth",
        "cpi_yoy_growth": "cpi_yoy_growth",

        # CPI 指数列，例如 100.3、101.2、99.5
        "居民消费价格指数": "cpi_yoy",
        "居民消费价格指数(上年同月=100)": "cpi_yoy",
        "CPI指数": "cpi_yoy",
        "CPI": "cpi_yoy",
        "cpi_yoy": "cpi_yoy",
    }

    df = df.rename(columns=rename_dict)

    print("\n重命名后的列名：")
    print(df.columns)

    # 4. 检查是否存在 date 列
    if "date" not in df.columns:
        raise ValueError(
            "没有找到日期列。请查看上方打印的原始列名，"
            "然后在 rename_dict 中把你的日期列名对应到 'date'。"
        )

    # 5. 处理日期格式
    df["date"] = df["date"].apply(parse_month)

    # 6. 处理 CPI 数据
    # 优先使用 cpi_yoy_growth；如果没有，则使用 cpi_yoy 反推
    if "cpi_yoy_growth" in df.columns:
        df["cpi_yoy_growth"] = pd.to_numeric(
            df["cpi_yoy_growth"],
            errors="coerce"
        )
        df["cpi_yoy"] = 100 + df["cpi_yoy_growth"]

    elif "cpi_yoy" in df.columns:
        df["cpi_yoy"] = pd.to_numeric(
            df["cpi_yoy"],
            errors="coerce"
        )
        df["cpi_yoy_growth"] = df["cpi_yoy"] - 100

    else:
        raise ValueError(
            "没有找到 CPI 同比增长率列或 CPI 指数列。"
            "请查看上方打印的原始列名，"
            "然后在 rename_dict 中把对应列名改成 'cpi_yoy_growth' 或 'cpi_yoy'。"
        )

    # 7. 只保留项目需要的标准列
    df = df[["date", "cpi_yoy_growth", "cpi_yoy"]]

    # 8. 删除完全重复的月份，只保留第一次出现
    df = df.drop_duplicates(subset=["date"], keep="first")

    # 9. 按时间升序排列
    df = df.sort_values("date").reset_index(drop=True)

    # 10. 检查缺失值
    print("\n缺失值检查：")
    print(df.isna().sum())

    missing_rows = df[df.isna().any(axis=1)]
    if not missing_rows.empty:
        print("\n存在缺失值的行：")
        print(missing_rows)

    # 11. 检查月份是否连续
    expected_months = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="MS"
    )

    missing_months = expected_months.difference(df["date"])

    print("\n缺失月份：")
    if len(missing_months) == 0:
        print("没有缺失月份，时间序列连续。")
    else:
        print(missing_months)

    # 12. 日期统一保存为 YYYY-MM 格式
    df["date"] = df["date"].dt.strftime("%Y-%m")

    # 13. 保存为 CSV
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n清洗后的数据前 5 行：")
    print(df.head())

    print("\n清洗后的数据后 5 行：")
    print(df.tail())

    print(f"\n已保存清洗后的数据到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()