from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.siamese_mom_ensemble_residual import (
    apply_residual_correction,
    build_equal_weight_ensemble,
    load_seed_winner_configs,
    select_residual_strength,
)


def _member(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_i_id": [1, 2, 3],
            "target_date": ["2020-01", "2020-02", "2020-03"],
            "cpi_actual": [100.0, 101.0, 99.0],
            "cpi_predicted": values,
        }
    )


def test_equal_weight_ensemble_aligns_members() -> None:
    second = _member([100.2, 100.8, 99.2]).iloc[::-1].reset_index(drop=True)
    result = build_equal_weight_ensemble(
        [
            ("seed_1", _member([99.8, 101.2, 98.8])),
            ("seed_2", second),
        ]
    )

    assert np.allclose(result["cpi_predicted_ensemble"], [100.0, 101.0, 99.0])
    assert np.allclose(result["member_prediction_std"], [0.2, 0.2, 0.2])


def test_residual_strength_is_convex_and_validation_selected() -> None:
    actual = np.asarray([1.0, 2.0, 3.0])
    ordinary = np.asarray([0.0, 1.0, 2.0])
    siamese = actual.copy()

    strength, trials = select_residual_strength(
        actual, ordinary, siamese, strengths=[0.0, 0.5, 1.0]
    )

    assert strength == 1.0
    assert trials.loc[0, "rmse"] == 0.0
    assert np.allclose(
        apply_residual_correction(ordinary, siamese, 0.5),
        [0.5, 1.5, 2.5],
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_residual_correction(ordinary, siamese, 1.1)


def test_seed_winner_config_loader_deduplicates(tmp_path: Path) -> None:
    table = pd.DataFrame(
        [
            {
                "seed": 1,
                "gap_months": 1,
                "k_references": 5,
                "level_weight": 0.1,
                "distance_power": 1.0,
                "max_pairs_per_bin": 1,
            },
            {
                "seed": 2,
                "gap_months": 12,
                "k_references": 8,
                "level_weight": 0.2,
                "distance_power": 2.0,
                "max_pairs_per_bin": 2,
            },
            {
                "seed": 3,
                "gap_months": 1,
                "k_references": 5,
                "level_weight": 0.1,
                "distance_power": 1.0,
                "max_pairs_per_bin": 1,
            },
        ]
    )
    path = tmp_path / "summary.csv"
    table.to_csv(path, index=False)

    winners = load_seed_winner_configs(path)

    assert [seed for seed, _ in winners] == [1, 2]
