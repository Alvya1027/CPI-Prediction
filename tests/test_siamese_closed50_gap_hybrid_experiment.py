import unittest

import numpy as np
import pandas as pd

from src.siamese_closed50_gap_hybrid_experiment import (
    add_hybrid_distances,
    build_gap_training_pairs,
    fit_hybrid_distance_calibration,
    select_hybrid_references,
)


def _split(sample_ids, start_date, level_shift=0.0):
    sample_ids = list(sample_ids)
    target_dates = pd.period_range(start_date, periods=len(sample_ids), freq="M")
    index = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "x_start_date": (target_dates - 12).astype(str),
            "x_end_date": (target_dates - 1).astype(str),
            "target_date": target_dates.astype(str),
        }
    )
    base = np.linspace(99.0, 101.0, 12)
    X = np.vstack(
        [base + level_shift + 0.1 * index for index in range(len(sample_ids))]
    )
    y = X[:, -1] + 0.1
    return {"X": X, "y": y, "index": index}


class Closed50GapHybridExperimentTests(unittest.TestCase):
    def test_gap_candidate_and_target_counts(self):
        pool = _split(range(50), "2000-01")
        expected = {
            1: (741, 38),
            3: (666, 36),
            6: (561, 33),
            12: (378, 27),
        }
        for gap, (candidate_count, target_count) in expected.items():
            _, candidates, _, _ = build_gap_training_pairs(pool, gap)
            self.assertEqual(len(candidates), candidate_count)
            self.assertEqual(candidates["sample_i_id"].nunique(), target_count)

    def test_level_shift_is_visible_to_hybrid_distance(self):
        pool = _split(range(50), "2000-01")
        _, candidates, _, calibration = build_gap_training_pairs(pool, 1)
        pair = pd.DataFrame(
            {
                "sample_i_id": [100],
                "sample_j_id": [101],
                "target_j_date": ["1999-01"],
                "window_distance": [0.0],
            }
        )
        windows = {
            100: np.linspace(100.0, 101.0, 12),
            101: np.linspace(105.0, 106.0, 12),
        }
        enriched = add_hybrid_distances(pair, windows, calibration)
        self.assertEqual(float(enriched.loc[0, "shape_distance"]), 0.0)
        self.assertGreater(float(enriched.loc[0, "level_distance"]), 0.0)
        self.assertGreater(float(enriched.loc[0, "hybrid_distance"]), 0.0)
        self.assertTrue(np.isfinite(candidates["hybrid_distance"]).all())

    def test_hybrid_selection_uses_smallest_distance(self):
        candidates = pd.DataFrame(
            {
                "sample_i_id": [10, 10, 10, 11, 11, 11],
                "sample_j_id": [1, 2, 3, 1, 2, 3],
                "target_j_date": [
                    "2000-01",
                    "2000-02",
                    "2000-03",
                    "2000-01",
                    "2000-02",
                    "2000-03",
                ],
                "hybrid_distance": [0.5, 0.1, 0.3, 0.2, 0.6, 0.4],
                "cpi_i": [999.0, -999.0, 0.0, -500.0, 500.0, 0.0],
            }
        )
        selected = select_hybrid_references(candidates, k=2)
        self.assertEqual(
            selected.groupby("sample_i_id")["sample_j_id"].apply(list).to_dict(),
            {10: [2, 3], 11: [1, 3]},
        )

    def test_calibration_uses_train_pool_statistics(self):
        pool = _split(range(50), "2000-01")
        _, candidates, _, calibration = build_gap_training_pairs(pool, 3)
        independently_fitted = fit_hybrid_distance_calibration(pool, candidates)
        self.assertAlmostEqual(
            calibration.mean_level_std,
            independently_fitted.mean_level_std,
        )
        self.assertGreater(calibration.shape_distance_median, 0.0)
        self.assertGreater(calibration.level_distance_median, 0.0)


if __name__ == "__main__":
    unittest.main()
