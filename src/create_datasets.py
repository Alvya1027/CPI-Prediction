import sys

import pandas as pd

sys.path.append(".")

from src.config import DATA_PROCESSED_DIR, DEFAULT_HORIZON, DEFAULT_WINDOW_SIZE
from src.data_utils import save_window_dataset


TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TARGET_COL = "cpi"


def main():
    df = pd.read_csv(DATA_PROCESSED_DIR / "cpi_monthly.csv")
    summary = save_window_dataset(
        df=df,
        output_dir=DATA_PROCESSED_DIR,
        window_size=DEFAULT_WINDOW_SIZE,
        horizon=DEFAULT_HORIZON,
        target_col=TARGET_COL,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        suffix="",
    )

    print("Saved default CPI sliding-window dataset.")
    print(summary)
    print("Files:")
    print("  X_train.npy, y_train.npy")
    print("  X_val.npy, y_val.npy")
    print("  X_test.npy, y_test.npy")
    print("  sample_index.csv")


if __name__ == "__main__":
    main()
