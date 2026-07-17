"""Generate the month-to-previous-month ratio of the existing CPI series.

This ratio is calculated from the project's CPI year-on-year index. It is not
the official CPI month-on-month index published with "previous month = 100".
"""

import sys

import pandas as pd

sys.path.append(".")

from src.config import DATA_PROCESSED_DIR, DEFAULT_HORIZON, DEFAULT_WINDOW_SIZE
from src.data_utils import save_window_dataset


INPUT_FILE = DATA_PROCESSED_DIR / "cpi_monthly.csv"
OUTPUT_FILE = DATA_PROCESSED_DIR / "cpi_month_ratio.csv"
TARGET_COL = "cpi_month_ratio"
SUFFIX = "_month_ratio"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def main():
    df = pd.read_csv(INPUT_FILE, usecols=["date", "cpi"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df = df.sort_values("date").reset_index(drop=True)

    expected_previous = df["date"] - pd.offsets.MonthBegin(1)
    actual_previous = df["date"].shift(1)
    if not actual_previous.iloc[1:].reset_index(drop=True).equals(
        expected_previous.iloc[1:].reset_index(drop=True)
    ):
        raise ValueError("CPI data contains missing or non-consecutive months.")

    result = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m"),
            "cpi_yoy": df["cpi"],
            "previous_date": actual_previous.dt.strftime("%Y-%m"),
            "previous_cpi_yoy": df["cpi"].shift(1),
        }
    )
    result[TARGET_COL] = (result["cpi_yoy"] / result["previous_cpi_yoy"] * 100).round(4)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    model_df = result.dropna(subset=[TARGET_COL])[["date", TARGET_COL]]
    summary = save_window_dataset(
        df=model_df,
        output_dir=DATA_PROCESSED_DIR,
        window_size=DEFAULT_WINDOW_SIZE,
        horizon=DEFAULT_HORIZON,
        target_col=TARGET_COL,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        suffix=SUFFIX,
    )

    print(f"Saved {OUTPUT_FILE.name}: {len(result)} rows")
    print(f"Usable ratio values: {model_df.shape[0]}")
    print(summary)


if __name__ == "__main__":
    main()
