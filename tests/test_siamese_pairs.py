from __future__ import annotations

import unittest

import pandas as pd

from src.config import DATA_PROCESSED_DIR


class SiamesePairContractTest(unittest.TestCase):
    def test_validation_and_test_cover_every_target_without_label_based_selection(self) -> None:
        sample_index = pd.read_csv(DATA_PROCESSED_DIR / "sample_index.csv")
        for split in ("val", "test"):
            pairs = pd.read_csv(DATA_PROCESSED_DIR / f"pair_indices_{split}.csv")
            expected_targets = int((sample_index["split"] == split).sum())

            self.assertEqual(pairs["sample_i_id"].nunique(), expected_targets)
            self.assertTrue((pairs.groupby("sample_i_id").size() == 5).all())
            self.assertEqual(set(pairs["selection_method"]), {"window_distance"})
            self.assertTrue((pairs["window_distance"] >= 0).all())

    def test_reference_dates_are_strictly_earlier(self) -> None:
        for split in ("train", "val", "test"):
            pairs = pd.read_csv(DATA_PROCESSED_DIR / f"pair_indices_{split}.csv")
            target_i = pd.to_datetime(pairs["target_i_date"])
            target_j = pd.to_datetime(pairs["target_j_date"])
            start_i = pd.to_datetime(pairs["x_i_start_date"])
            end_j = pd.to_datetime(pairs["x_j_end_date"])
            gaps = (
                (start_i.dt.year - end_j.dt.year) * 12
                + start_i.dt.month
                - end_j.dt.month
            )

            self.assertTrue((target_j < target_i).all())
            self.assertTrue((gaps >= 12).all())
            self.assertFalse(
                pairs.duplicated(subset=["sample_i_id", "sample_j_id"]).any()
            )


if __name__ == "__main__":
    unittest.main()
