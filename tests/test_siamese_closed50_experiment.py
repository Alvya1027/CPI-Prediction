import unittest

import numpy as np
import pandas as pd

from src.siamese_closed50_experiment import (
    build_closed_training_pairs,
    build_fixed_bank_evaluation_pairs,
)


def _split(sample_ids, start_date):
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
    X = np.column_stack(
        [np.linspace(i, i + 1, 12) for i in range(len(sample_ids))]
    ).T
    y = np.linspace(100.0, 102.0, len(sample_ids))
    return {"X": X, "y": y, "index": index}


class Closed50ExperimentTests(unittest.TestCase):
    def test_training_targets_and_references_stay_inside_pool(self):
        pool = _split(range(50), "2000-01")
        selected, candidates, _ = build_closed_training_pairs(pool)
        allowed = set(range(50))
        self.assertTrue(set(selected["sample_i_id"]).issubset(allowed))
        self.assertTrue(set(selected["sample_j_id"]).issubset(allowed))
        self.assertTrue(set(candidates["sample_i_id"]).issubset(allowed))
        self.assertTrue(set(candidates["sample_j_id"]).issubset(allowed))
        self.assertLess(selected["sample_i_id"].nunique(), 50)

    def test_evaluation_uses_only_fixed_closed_reference_bank(self):
        pool = _split(range(50), "2000-01")
        evaluation = _split(range(100, 105), "2005-01")
        selected = build_fixed_bank_evaluation_pairs(
            pool,
            evaluation,
            split_name="val",
            strategy="window_distance",
            k=3,
        )
        self.assertEqual(selected["sample_i_id"].nunique(), 5)
        self.assertTrue(set(selected["sample_j_id"]).issubset(set(range(50))))
        self.assertTrue((selected.groupby("sample_i_id").size() == 3).all())


if __name__ == "__main__":
    unittest.main()
