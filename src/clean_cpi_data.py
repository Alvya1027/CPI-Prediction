import sys

import pandas as pd

sys.path.append(".")
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR


RAW_FILE = DATA_RAW_DIR / "cpi_raw.xlsx"
OUTPUT_FILE = DATA_PROCESSED_DIR / "cpi_monthly.csv"


def parse_month(value):
    """Convert common monthly date formats to month-start timestamps."""
    text = str(value).strip().replace(".0", "")
    text = text.replace("年", "-").replace("月", "").replace("/", "-")

    if len(text) == 6 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m")

    parsed = pd.to_datetime(text)
    return pd.Timestamp(year=parsed.year, month=parsed.month, day=1)


def read_raw_cpi() -> pd.DataFrame:
    """Read the raw CPI sheet and normalize column names."""
    df = pd.read_excel(RAW_FILE, sheet_name="CPI数据")
    df.columns = df.columns.astype(str).str.strip()

    rename_dict = {
        "月份": "date",
        "日期": "date",
        "时间": "date",
        "date": "date",
        "Date": "date",
        "CPI同比增长率": "cpi_yoy_growth",
        "CPI同比增长率(%)": "cpi_yoy_growth",
        "CPI同比": "cpi_yoy_growth",
        "居民消费价格指数同比增长率": "cpi_yoy_growth",
        "同比增长率": "cpi_yoy_growth",
        "年增率": "cpi_yoy_growth",
        "中国-居民消费价格指数(年增率)": "cpi_yoy_growth",
        "cpi_yoy_growth": "cpi_yoy_growth",
        "居民消费价格指数": "cpi_yoy",
        "居民消费价格指数(上年同月=100)": "cpi_yoy",
        "CPI指数": "cpi_yoy",
        "CPI": "cpi_yoy",
        "cpi_yoy": "cpi_yoy",
    }
    return df.rename(columns=rename_dict)


def clean_cpi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a monthly CPI table ready for modeling."""
    if "date" not in df.columns:
        raise ValueError("No date column found in raw CPI data.")

    df = df.copy()
    df["date"] = df["date"].apply(parse_month)

    if "cpi_yoy_growth" in df.columns:
        df["cpi_yoy_growth"] = pd.to_numeric(df["cpi_yoy_growth"], errors="coerce")
        df["cpi_yoy"] = 100 + df["cpi_yoy_growth"]
    elif "cpi_yoy" in df.columns:
        df["cpi_yoy"] = pd.to_numeric(df["cpi_yoy"], errors="coerce")
        df["cpi_yoy_growth"] = df["cpi_yoy"] - 100
    else:
        raise ValueError("No CPI column found. Expected cpi_yoy_growth or cpi_yoy.")

    df = df[["date", "cpi_yoy_growth", "cpi_yoy"]]
    df = df.drop_duplicates(subset=["date"], keep="first")
    df = df.sort_values("date").reset_index(drop=True)

    expected_months = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    missing_months = expected_months.difference(df["date"])
    if len(missing_months) > 0:
        missing_text = ", ".join(m.strftime("%Y-%m") for m in missing_months)
        raise ValueError(f"Missing monthly observations: {missing_text}")

    missing_rows = df[df.isna().any(axis=1)]
    if not missing_rows.empty:
        raise ValueError(f"Missing CPI values after cleaning:\n{missing_rows}")

    df["cpi_yoy_growth"] = df["cpi_yoy_growth"].round(2)
    df["cpi_yoy"] = df["cpi_yoy"].round(2)
    df["cpi"] = df["cpi_yoy"]
    df["date"] = df["date"].dt.strftime("%Y-%m")

    return df[["date", "cpi", "cpi_yoy_growth", "cpi_yoy"]]


def main():
    raw = read_raw_cpi()
    cleaned = clean_cpi_data(raw)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved cleaned CPI data to: {OUTPUT_FILE}")
    print(f"Rows: {len(cleaned)}")
    print(f"Date range: {cleaned['date'].iloc[0]} to {cleaned['date'].iloc[-1]}")
    print(cleaned.head())
    print(cleaned.tail())


if __name__ == "__main__":
    main()
