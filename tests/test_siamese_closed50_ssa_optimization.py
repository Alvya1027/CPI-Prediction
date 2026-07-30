import unittest

import numpy as np
import pandas as pd

from src.siamese_closed50_ssa_optimization import (
    BASELINE_POSITION,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    aggregate_power_predictions,
    decode_position,
    select_search_references,
    sparrow_search,
)


class Closed50SSAOptimizationTests(unittest.TestCase):
    def test_decode_position_respects_mixed_parameter_bounds(self):
        config = decode_position(np.asarray([-10.0, 99.0, 0.1234, 9.0, -2.0]))
        self.assertEqual(config.gap_months, 1)
        self.assertEqual(config.k_references, 15)
        self.assertEqual(config.level_weight, 0.123)
        self.assertEqual(config.distance_power, 3.0)
        self.assertEqual(config.max_pairs_per_bin, 1)

    def test_reference_selection_does_not_use_target_labels(self):
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
                "shape_distance_normalized": [0.1, 0.4, 0.7, 0.8, 0.2, 0.6],
                "level_distance_normalized": [0.8, 0.2, 0.3, 0.1, 0.7, 0.4],
                "cpi_i": [1.0, 2.0, 3.0, 100.0, -100.0, 0.0],
                "delta_cpi": [5.0, -5.0, 0.0, 10.0, -10.0, 0.0],
            }
        )
        first = select_search_references(candidates, k=2, level_weight=0.25)
        perturbed = candidates.copy()
        perturbed["cpi_i"] = perturbed["cpi_i"].iloc[::-1].to_numpy()
        perturbed["delta_cpi"] *= -1000
        second = select_search_references(perturbed, k=2, level_weight=0.25)
        self.assertEqual(
            first[["sample_i_id", "sample_j_id"]].values.tolist(),
            second[["sample_i_id", "sample_j_id"]].values.tolist(),
        )

    def test_power_aggregation_favors_nearer_reference(self):
        pairs = pd.DataFrame(
            {
                "sample_i_id": [10, 10],
                "target_i_date": ["2020-01", "2020-01"],
                "cpi_i": [100.0, 100.0],
                "cpi_pred_pair": [99.0, 103.0],
                "search_distance": [1.0, 2.0],
            }
        )
        p1 = aggregate_power_predictions(pairs, distance_power=1.0)
        p2 = aggregate_power_predictions(pairs, distance_power=2.0)
        self.assertLess(
            abs(float(p2.loc[0, "cpi_predicted"]) - 99.0),
            abs(float(p1.loc[0, "cpi_predicted"]) - 99.0),
        )

    def test_ssa_keeps_known_baseline_and_respects_bounds(self):
        def objective(position):
            self.assertTrue(np.all(position >= LOWER_BOUNDS))
            self.assertTrue(np.all(position <= UPPER_BOUNDS))
            return float(np.sum((position - BASELINE_POSITION) ** 2))

        best, fitness, history = sparrow_search(
            objective,
            seed=7,
            population_size=6,
            iterations=3,
        )
        np.testing.assert_allclose(best, BASELINE_POSITION)
        self.assertEqual(fitness, 0.0)
        self.assertEqual(len(history), 4)


if __name__ == "__main__":
    unittest.main()
