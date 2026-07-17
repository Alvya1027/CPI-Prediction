from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from src.config import DATA_PROCESSED_DIR
from src.siamese_reservoir_regression import (
    build_pair_features,
    load_state_lookup,
    run_training,
)


class SiameseReservoirRegressionTest(unittest.TestCase):
    def _write_mock_states(self, state_dir: Path) -> None:
        rng = np.random.default_rng(42)
        mask = rng.choice((-1.0, 1.0), size=(12, 50))
        sample_index = pd.read_csv(DATA_PROCESSED_DIR / "sample_index.csv")
        for split in ("train", "val", "test"):
            inputs = np.load(DATA_PROCESSED_DIR / f"X_{split}_scaled.npy")
            states = np.tanh(inputs @ mask / np.sqrt(inputs.shape[1]))
            sample_ids = sample_index.loc[
                sample_index["split"] == split, "sample_id"
            ].to_numpy(dtype=int)
            savemat(
                state_dir / f"states_{split}.mat",
                {"state_matrix": states, "sample_id": sample_ids.reshape(-1, 1)},
            )

    def test_signed_state_difference_has_expected_direction(self) -> None:
        pairs = pd.DataFrame({"sample_i_id": [1], "sample_j_id": [2]})
        states = {1: np.asarray([3.0, 1.0]), 2: np.asarray([1.0, 4.0])}
        features = build_pair_features(pairs, states, "signed_diff")
        np.testing.assert_allclose(features, [[2.0, -3.0]])

    def test_end_to_end_pipeline_covers_all_test_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state_dir = root / "states"
            output_dir = root / "results"
            state_dir.mkdir()
            self._write_mock_states(state_dir)

            lookup = load_state_lookup(state_dir)
            self.assertEqual(len(lookup), 304)
            metrics = run_training(state_dir=state_dir, output_dir=output_dir)

            self.assertEqual(metrics["split"].tolist(), ["train", "val", "test"])
            self.assertTrue(np.isfinite(metrics["cpi_rmse"]).all())
            test_predictions = pd.read_csv(
                output_dir / "tables" / "siamese_optical_predictions_test.csv"
            )
            self.assertEqual(len(test_predictions), 47)
            self.assertEqual(test_predictions["sample_i_id"].nunique(), 47)
            self.assertTrue(np.isfinite(test_predictions["cpi_predicted"]).all())


if __name__ == "__main__":
    unittest.main()
