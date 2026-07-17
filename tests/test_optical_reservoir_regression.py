from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from src.config import DATA_PROCESSED_DIR
from src.optical_reservoir_regression import run_training


class OpticalReservoirRegressionTest(unittest.TestCase):
    def test_end_to_end_outputs_cover_the_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state_dir = root / "states"
            output_dir = root / "results"
            state_dir.mkdir()
            rng = np.random.default_rng(42)
            mask = rng.normal(size=(12, 8))
            sample_index = pd.read_csv(DATA_PROCESSED_DIR / "sample_index.csv")

            for split in ("train", "val", "test"):
                inputs = np.load(DATA_PROCESSED_DIR / f"X_{split}_scaled.npy")
                rows = sample_index.loc[sample_index["split"] == split]
                savemat(
                    state_dir / f"states_{split}.mat",
                    {
                        "state_matrix": np.tanh(inputs @ mask),
                        "sample_id": rows["sample_id"].to_numpy().reshape(-1, 1),
                        "target": rows["y"].to_numpy().reshape(-1, 1),
                    },
                )

            metrics = run_training(state_dir=state_dir, output_dir=output_dir)
            self.assertEqual(metrics["num_targets"].tolist(), [212, 45, 47])
            self.assertTrue(np.isfinite(metrics[["mae", "rmse"]]).all().all())
            predictions = pd.read_csv(
                output_dir / "tables" / "optical_reservoir_predictions_test.csv"
            )
            self.assertEqual(len(predictions), 47)


if __name__ == "__main__":
    unittest.main()
