import sys

import pandas as pd

sys.path.append(".")

from src.config import DATA_PROCESSED_DIR
from src.data_utils import save_window_dataset


HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TARGET_COL = "cpi"
WINDOW_SIZES = [6, 12, 24]


def main():
    df = pd.read_csv(DATA_PROCESSED_DIR / "cpi_monthly.csv")

    for window_size in WINDOW_SIZES:
        suffix = f"_ws{window_size}"
        summary = save_window_dataset(
            df=df,
            output_dir=DATA_PROCESSED_DIR,
            window_size=window_size,
            horizon=HORIZON,
            target_col=TARGET_COL,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            suffix=suffix,
        )
        print(f"Saved dataset {suffix}: {summary}")


if __name__ == "__main__":
    main()
