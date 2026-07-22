from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.siamese_final_test import (
    FINAL_SUMMARY_NAME,
    load_frozen_configuration,
    run_final_test,
    select_final_references,
)


class SiameseFinalTestContractTest(unittest.TestCase):
    def test_frozen_configuration_must_be_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            payload = {
                "status": "validation_selected_not_tested",
                "reference_strategy": "window_distance",
                "k": 10,
                "feature_mode": "signed_diff",
                "aggregation": "inverse_distance",
                "alpha": 10.0,
                "validation_mae": 0.4,
                "validation_rmse": 0.5,
                "test_evaluated": False,
            }
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_frozen_configuration(config_path)
            self.assertFalse(loaded["test_evaluated"])

            payload["test_evaluated"] = True
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frozen_configuration(config_path)

    def test_final_reference_selection_is_label_independent(self) -> None:
        candidates = pd.DataFrame(
            {
                "sample_i_id": [10, 10, 10, 11, 11, 11],
                "sample_j_id": [1, 2, 3, 1, 2, 3],
                "target_j_date": [
                    "2020-01",
                    "2020-02",
                    "2020-03",
                    "2020-01",
                    "2020-02",
                    "2020-03",
                ],
                "window_distance": [0.3, 0.1, 0.2, 0.2, 0.3, 0.1],
                "cpi_i": [100.0] * 6,
                "cpi_j": [99.0] * 6,
                "delta_cpi": [1.0] * 6,
            }
        )
        changed = candidates.copy()
        changed[["cpi_i", "cpi_j", "delta_cpi"]] = np.arange(18).reshape(6, 3)
        for strategy in ("window_distance", "recent"):
            original = select_final_references(candidates, strategy, 2)
            relabeled = select_final_references(changed, strategy, 2)
            np.testing.assert_array_equal(
                original[["sample_i_id", "sample_j_id"]].to_numpy(),
                relabeled[["sample_i_id", "sample_j_id"]].to_numpy(),
            )

    def test_existing_final_summary_blocks_a_rerun_before_any_input_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            table_dir = output_dir / "tables"
            table_dir.mkdir()
            (table_dir / FINAL_SUMMARY_NAME).write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_final_test(
                    state_dir=output_dir / "missing_states",
                    data_dir=output_dir / "missing_data",
                    output_dir=output_dir,
                    config_path=output_dir / "missing_config.json",
                )


if __name__ == "__main__":
    unittest.main()
