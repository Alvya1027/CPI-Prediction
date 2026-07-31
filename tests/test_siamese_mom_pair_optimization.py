from __future__ import annotations

import numpy as np
import pandas as pd

from src.siamese_mom_pair_optimization import (
    add_balanced_pair_weights,
    aggregate_pair_predictions,
    build_target_gap_candidates,
    mirror_pairs,
)


def _split() -> dict[str, object]:
    dates = pd.date_range("2018-01-01", periods=4, freq="MS")
    index = pd.DataFrame(
        {
            "sample_id": np.arange(4),
            "x_start_date": (dates - pd.DateOffset(months=12)).strftime("%Y-%m"),
            "x_end_date": (dates - pd.DateOffset(months=1)).strftime("%Y-%m"),
            "target_date": dates.strftime("%Y-%m"),
        }
    )
    return {
        "X": np.arange(48, dtype=float).reshape(4, 12),
        "y": np.asarray([99.8, 100.0, 100.3, 100.1]),
        "index": index,
    }


def test_target_gap_uses_target_months_and_allows_overlapping_windows() -> None:
    split = _split()
    gap1 = build_target_gap_candidates(split, split, 1)
    gap2 = build_target_gap_candidates(split, split, 2)

    assert len(gap1) == 6
    assert len(gap2) == 3
    assert gap1["sample_i_id"].nunique() == 3
    assert (gap1["sample_i_id"] > gap1["sample_j_id"]).all()


def test_balanced_weights_give_every_target_equal_total_weight() -> None:
    pairs = pd.DataFrame(
        {
            "sample_i_id": [1, 1, 1, 2, 2],
            "delta_bin": ["small", "small", "large", "medium", "large"],
        }
    )
    weighted = add_balanced_pair_weights(pairs)

    totals = weighted.groupby("sample_i_id")["sample_weight"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)
    target1 = weighted.loc[weighted["sample_i_id"].eq(1)]
    assert np.isclose(target1.loc[target1["delta_bin"].eq("small"), "sample_weight"].sum(), 0.5)
    assert np.isclose(target1.loc[target1["delta_bin"].eq("large"), "sample_weight"].sum(), 0.5)


def test_mirror_pairs_reverse_ids_labels_and_preserve_total_weight() -> None:
    pairs = pd.DataFrame(
        {
            "sample_i_id": [2],
            "sample_j_id": [1],
            "x_i_start_date": ["2017-03"],
            "x_i_end_date": ["2018-02"],
            "target_i_date": ["2018-03"],
            "x_j_start_date": ["2017-02"],
            "x_j_end_date": ["2018-01"],
            "target_j_date": ["2018-02"],
            "cpi_i": [100.3],
            "cpi_j": [100.0],
            "delta_cpi": [0.3],
            "delta_bin": ["large"],
            "sample_weight": [1.0],
            "selection_method": ["original"],
        }
    )
    mirrored = mirror_pairs(pairs)

    assert len(mirrored) == 2
    assert mirrored["sample_weight"].sum() == 1.0
    assert mirrored.iloc[1]["sample_i_id"] == 1
    assert mirrored.iloc[1]["sample_j_id"] == 2
    assert np.isclose(mirrored.iloc[1]["delta_cpi"], -0.3)


def test_trimmed_mean_removes_one_high_and_one_low_reference() -> None:
    pairs = pd.DataFrame(
        {
            "sample_i_id": [10] * 5,
            "target_i_date": ["2020-01"] * 5,
            "cpi_i": [100.0] * 5,
            "cpi_pred_pair": [90.0, 99.8, 100.0, 100.2, 110.0],
            "search_distance": [1.0] * 5,
        }
    )
    result = aggregate_pair_predictions(pairs, "trimmed_mean", 1.0)

    assert len(result) == 1
    assert np.isclose(result.iloc[0]["cpi_predicted"], 100.0)
