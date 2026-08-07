from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.prepare_optical_reservoir_mom_train45_noval import build_dataset
from src.teacher_shared_readout_pipeline import build_train_pairs


ROOT = Path(__file__).resolve().parents[1]


def test_train45_test47_has_no_validation_split() -> None:
    index, arrays = build_dataset()
    assert set(index["split"]) == {"train", "test"}
    train = index.loc[index["split"].eq("train")].reset_index(drop=True)
    test = index.loc[index["split"].eq("test")].reset_index(drop=True)
    assert (len(train), train.iloc[0]["target_date"], train.iloc[-1]["target_date"]) == (
        45,
        "2018-09",
        "2022-05",
    )
    assert (len(test), test.iloc[0]["target_date"], test.iloc[-1]["target_date"]) == (
        47,
        "2022-06",
        "2026-04",
    )
    assert arrays["X_train"].shape == (45, 12)
    assert arrays["X_test"].shape == (47, 12)
    assert not any("val" in key for key in arrays)


def test_train45_gap1_relations_stay_inside_train() -> None:
    index, arrays = build_dataset()
    train_index = index.loc[index["split"].eq("train")].reset_index(drop=True)
    train = {
        "index": train_index,
        "X": arrays["X_train"],
        "y": arrays["y_train"],
    }
    pairs, pair_i, pair_j = build_train_pairs(train, min_gap_months=1)
    assert len(pairs) == len(pair_i) == len(pair_j) == 561
    assert pairs["sample_i_id"].nunique() == 33
    train_ids = set(train_index["sample_id"].astype(int))
    assert set(pairs["sample_i_id"].astype(int)).issubset(train_ids)
    assert set(pairs["sample_j_id"].astype(int)).issubset(train_ids)


def test_matlab_train45_entrypoints_do_not_generate_validation() -> None:
    matlab_dir = ROOT / "matlab" / "optical_reservoir_cpi"
    train_source = (matlab_dir / "run_teacher_twin_train45_noval.m").read_text(
        encoding="utf-8"
    )
    config_source = (
        matlab_dir / "config_twin_cpi_rc_train45_noval.m"
    ).read_text(encoding="utf-8")
    assert "prepare_twin_window_cache('train'" in train_source
    assert "run_twin_state_cache('train'" in train_source
    assert "'val'" not in train_source
    assert "config.train_count = 45" in config_source
    assert "config.valid_splits = {'train', 'test'}" in config_source
    assert "config.no_validation_split = true" in config_source


def test_python_train45_runner_declares_fixed_pretest_configuration() -> None:
    source = (
        ROOT / "scripts" / "run_teacher_explicit_twin_mom_train45_noval.py"
    ).read_text(encoding="utf-8")
    assert 'FREEZE_STATUS = "train_only_fixed_not_tested"' in source
    assert 'AUTHORIZATION_STATUS = "train_only_fixed_authorized_for_test_state_generation"' in source
    assert '"validation_targets": 0' in source
    assert "_load_expected_split(data_dir, \"train\")" in source
    assert "_load_expected_split(data_dir, \"val\")" not in source
    assert '"test_labels_accessed": False' in source
