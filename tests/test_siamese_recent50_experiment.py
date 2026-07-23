import unittest

import pandas as pd

from src.siamese_recent50_experiment import (
    filter_training_pairs,
    select_recent_training_targets,
)


class Recent50ExperimentTests(unittest.TestCase):
    def test_selects_last_50_chronological_train_targets(self):
        dates = pd.period_range("2000-01", periods=70, freq="M").astype(str)
        table = pd.DataFrame(
            {
                "sample_id": range(70),
                "split": ["train"] * 60 + ["val"] * 10,
                "target_date": dates,
            }
        )
        selected = select_recent_training_targets(table, count=50)
        self.assertEqual(len(selected), 50)
        self.assertEqual(int(selected["sample_id"].iloc[0]), 10)
        self.assertEqual(int(selected["sample_id"].iloc[-1]), 59)
        self.assertEqual(str(selected["target_date"].iloc[0]), "2000-11")

    def test_pair_filter_limits_targets_but_keeps_older_references(self):
        pairs = pd.DataFrame(
            {
                "sample_i_id": [10, 10, 11, 11, 12],
                "sample_j_id": [0, 9, 1, 10, 2],
                "delta_cpi": [1.0, 0.5, 0.8, 0.2, 0.7],
                "cpi_j": [100.0, 100.5, 100.1, 101.0, 100.2],
            }
        )
        selected = filter_training_pairs(pairs, [10, 11])
        self.assertEqual(set(selected["sample_i_id"]), {10, 11})
        self.assertIn(0, set(selected["sample_j_id"]))
        self.assertEqual(len(selected), 4)

    def test_pair_filter_requires_every_recent_target(self):
        pairs = pd.DataFrame(
            {
                "sample_i_id": [10],
                "sample_j_id": [0],
                "delta_cpi": [1.0],
                "cpi_j": [100.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "without frozen training pairs"):
            filter_training_pairs(pairs, [10, 11])


if __name__ == "__main__":
    unittest.main()
