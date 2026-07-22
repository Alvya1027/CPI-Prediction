from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from src.siamese_validation_search import (
    build_validation_candidates,
    load_validation_state_lookup,
    select_validation_references,
)


class SiameseValidationSearchTest(unittest.TestCase):
    def test_state_loader_does_not_require_a_test_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state_dir = Path(temp)
            for offset, split in enumerate(("train", "val")):
                savemat(
                    state_dir / f"states_{split}.mat",
                    {
                        "state_matrix": np.full((2, 3), float(offset + 1)),
                        "sample_id": np.asarray([offset * 2, offset * 2 + 1])
                        .reshape(-1, 1),
                    },
                )
            lookup = load_validation_state_lookup(state_dir)
            self.assertEqual(sorted(lookup), [0, 1, 2, 3])

    def test_real_validation_pool_is_time_safe_and_supports_k10(self) -> None:
        candidates = build_validation_candidates()
        target_i = pd.to_datetime(candidates["target_i_date"])
        target_j = pd.to_datetime(candidates["target_j_date"])
        self.assertTrue((target_j < target_i).all())
        self.assertEqual(candidates["sample_i_id"].nunique(), 45)

        for strategy in ("window_distance", "recent"):
            selected = select_validation_references(candidates, strategy, 10)
            self.assertEqual(selected["sample_i_id"].nunique(), 45)
            self.assertTrue((selected.groupby("sample_i_id").size() == 10).all())

    def test_reference_selection_is_independent_of_target_labels(self) -> None:
        candidates = build_validation_candidates()
        changed = candidates.copy()
        rng = np.random.default_rng(42)
        changed["cpi_i"] = rng.normal(size=len(changed))
        changed["cpi_j"] = rng.normal(size=len(changed))
        changed["delta_cpi"] = rng.normal(size=len(changed))

        for strategy in ("window_distance", "recent"):
            original = select_validation_references(candidates, strategy, 5)
            relabeled = select_validation_references(changed, strategy, 5)
            original_ids = original[["sample_i_id", "sample_j_id"]].to_numpy()
            relabeled_ids = relabeled[["sample_i_id", "sample_j_id"]].to_numpy()
            np.testing.assert_array_equal(original_ids, relabeled_ids)


if __name__ == "__main__":
    unittest.main()
