from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.twin_state_cache_contract import (
    SCHEMA_VERSION,
    STATE_PROTOCOL,
    load_twin_state_cache,
    load_twin_state_splits,
    sha256_file,
)


def _hash(character: str) -> str:
    return character * 64


def _write_contract_fixture(
    root: Path,
    *,
    audit_failure: bool = False,
    duplicate_sample_id: bool = False,
    unequal_branch_parameters: bool = False,
) -> tuple[Path, Path]:
    state_dir = root / "states_twin"
    audit_dir = root / "audits_twin"
    state_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)

    a = np.linspace(0.1, 1.0, 50)
    b = np.linspace(-0.7, 0.2, 50)
    a_swapped = a.copy()
    if audit_failure:
        a_swapped[12] += 0.01
    state_file_placeholder = _hash("7")
    audit_payload = {
        "schema_version": SCHEMA_VERSION,
        "state_protocol": STATE_PROTOCOL,
        "simulation_protocol_sha256": _hash("1"),
        "shared_branch_model_sha256": _hash("b"),
        "twin_model_sha256": _hash("c"),
        "h_a_ab_branch_a": a,
        "h_b_ab_branch_b": b,
        "h_b_ba_branch_a": b,
        "h_a_ba_branch_b": a_swapped,
        "h_a_aa_branch_a": a,
        "h_a_aa_branch_b": a,
        "h_a_repeat_branch_a": a,
        "h_a_repeat_branch_b": a,
        "h_a_cache": a,
        "h_b_cache": b,
        "reservoir_parameter_sha256": _hash("e"),
        "mask_sha256": _hash("f"),
        "input_transform_sha256": _hash("5"),
        "state_file_sha256": state_file_placeholder,
    }
    audit_mat = audit_dir / "audit_twin_equivalence.mat"
    savemat(audit_mat, audit_payload)
    audit_json_payload = {
        "schema_version": SCHEMA_VERSION,
        "state_protocol": STATE_PROTOCOL,
        "audit_mat_sha256": sha256_file(audit_mat),
        "simulation_protocol_sha256": _hash("1"),
        "shared_branch_model_sha256": _hash("b"),
        "twin_model_sha256": _hash("c"),
        "audit_atol": 1e-10,
        "audit_rtol": 1e-8,
        "audit_passed": True,
        "reservoir_parameter_sha256": _hash("e"),
        "mask_sha256": _hash("f"),
        "input_transform_sha256": _hash("5"),
        "state_file_sha256": state_file_placeholder,
    }
    audit_json = audit_dir / "audit_twin_equivalence.json"
    audit_json.write_text(json.dumps(audit_json_payload), encoding="utf-8")

    sample_ids = np.asarray([10, 11, 12])
    if duplicate_sample_id:
        sample_ids[-1] = 11
    rng = np.random.default_rng(123)
    states = rng.normal(size=(3, 50))
    windows = rng.normal(size=(3, 12))
    state_payload = {
        "schema_version": SCHEMA_VERSION,
        "state_protocol": STATE_PROTOCOL,
        "split": "train",
        "state_matrix": states,
        "sample_id": sample_ids,
        "target_date": np.asarray(["2014-07", "2014-08", "2014-09"], dtype=object),
        "input_window_raw": windows,
        "input_window_scaled": (windows - windows.mean()) / windows.std(),
        "cache_record_id": np.asarray(["cache-10", "cache-11", "cache-12"], dtype=object),
        "branch_id": np.asarray(["target", "reference", "target"], dtype=object),
        "partner_sample_id": np.asarray([11, 10, 12]),
        "generation_run_id": np.asarray([1, 1, 2]),
        "state_mode": "isolated_repeated_window",
        "repeat_count": 4,
        "capture_cycle": 4,
        "sample_times_seconds": 4e-6 + np.arange(151, 201) * 4e-11,
        "input_transform_sha256": _hash("5"),
        "simulation_protocol_sha256": _hash("1"),
        "shared_branch_model_sha256": _hash("b"),
        "twin_model_sha256": _hash("c"),
    }
    state_file = state_dir / "state_cache_train.mat"
    savemat(state_file, state_payload)

    # The immutable audit is tied to the exact training cache.  Rebuild its
    # two files now that the synthetic cache hash is known.
    state_hash = sha256_file(state_file)
    audit_payload["state_file_sha256"] = state_hash
    savemat(audit_mat, audit_payload)
    audit_json_payload["audit_mat_sha256"] = sha256_file(audit_mat)
    audit_json_payload["state_file_sha256"] = state_hash
    audit_json.write_text(json.dumps(audit_json_payload), encoding="utf-8")

    parameter_b = _hash("e") if not unequal_branch_parameters else _hash("9")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state_protocol": STATE_PROTOCOL,
        "split": "train",
        "state_file": state_file.name,
        "state_file_sha256": sha256_file(state_file),
        "audit_mat_file": audit_mat.name,
        "audit_mat_sha256": sha256_file(audit_mat),
        "audit_json_file": audit_json.name,
        "audit_json_sha256": sha256_file(audit_json),
        "source_reservoir_model_sha256": _hash("a"),
        "shared_branch_model_sha256": _hash("b"),
        "twin_model_sha256": _hash("c"),
        "branch_a_model_sha256": _hash("b"),
        "branch_b_model_sha256": _hash("b"),
        "branch_a_parameter_sha256": _hash("e"),
        "branch_b_parameter_sha256": parameter_b,
        "branch_a_mask_sha256": _hash("f"),
        "branch_b_mask_sha256": _hash("f"),
        "input_transform_sha256": _hash("5"),
        "simulation_protocol_sha256": _hash("1"),
        "generator_script_sha256": _hash("2"),
        "input_file_sha256": _hash("3"),
        "initial_condition_inventory_sha256": _hash("4"),
        "cache_reuse_declared": True,
        "cache_generated_by_explicit_twin_model": True,
        "semantic_pairs_simulated_simultaneously": False,
        "pair_states_resolved_by_sample_id": True,
        "derived_pair_states_from_cache": True,
        "same_model_reference_for_both_branches": True,
        "no_cross_branch_reservoir_coupling": True,
        "reset_between_runs": True,
        "state_mode": "isolated_repeated_window",
        "sample_phase": "node_end",
        "num_records": 3,
        "state_width": 50,
        "window_size": 12,
        "repeat_count": 4,
        "capture_cycle": 4,
        "theta_seconds": 4e-11,
        "feedback_delay_seconds": 2.04e-9,
        "input_transport_delay_seconds": 4e-6,
        "noise_seed": 1,
        "solver": "ode4",
        "fixed_step_seconds": 1e-12,
        "matlab_release": "R2025b",
        "simulink_version": "25.2",
    }
    manifest_path = state_dir / "state_cache_train.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return state_dir, audit_dir


def test_valid_explicit_twin_cache_returns_lookup_and_provenance(tmp_path: Path) -> None:
    state_dir, audit_dir = _write_contract_fixture(tmp_path)

    loaded = load_twin_state_cache(
        state_dir,
        "train",
        audit_dir=audit_dir,
        expected_sample_ids=[10, 11, 12],
        expected_target_dates=["2014-07", "2014-08", "2014-09"],
    )

    assert set(loaded.state_lookup) == {10, 11, 12}
    assert loaded.state_lookup[10].shape == (50,)
    assert not loaded.state_lookup[10].flags.writeable
    assert loaded.provenance["cache_record_id"].tolist() == [
        "cache-10",
        "cache-11",
        "cache-12",
    ]
    assert loaded.provenance["generation_run_id"].tolist() == ["1", "1", "2"]
    assert not loaded.provenance["semantic_pairs_simulated_simultaneously"].any()
    assert all(metrics["passed"] is True for metrics in loaded.audit_metrics.values())


def test_pipeline_adapter_returns_lookup_and_hash_audit(tmp_path: Path) -> None:
    import pandas as pd

    state_dir, _audit_dir = _write_contract_fixture(tmp_path)
    fixture = loadmat_for_test(state_dir / "state_cache_train.mat")
    payload = {
        "X": np.asarray(fixture["input_window_raw"], dtype=float),
        # A deliberately present label vector is not needed or inspected by
        # the state loader; supervision remains a downstream Python concern.
        "y": np.asarray([100.0, 100.1, 99.9]),
        "index": pd.DataFrame(
            {
                "sample_id": [10, 11, 12],
                "target_date": ["2014-07", "2014-08", "2014-09"],
            }
        ),
    }

    lookup, audit = load_twin_state_splits(state_dir, {"train": payload})

    assert set(lookup) == {10, 11, 12}
    assert audit["state_protocol"] == STATE_PROTOCOL
    assert len(audit["protocol_identity_sha256"]) == 64
    assert audit["splits"]["train"]["sha256"] == sha256_file(
        state_dir / "state_cache_train.mat"
    )
    assert audit["audit_mat_sha256"] == sha256_file(
        tmp_path / "audits_twin" / "audit_twin_equivalence.mat"
    )
    assert audit["audit_json_sha256"] == sha256_file(
        tmp_path / "audits_twin" / "audit_twin_equivalence.json"
    )
    assert audit["cache_records"][0] == {
        "split": "train",
        "sample_id": 10,
        "cache_record_id": "cache-10",
        "branch_id": "target",
        "partner_sample_id": 11,
        "generation_run_id": "1",
        "state_file_sha256": sha256_file(state_dir / "state_cache_train.mat"),
    }
    json.dumps(audit)
    assert audit["semantic_pairs_simulated_simultaneously"] is False


def test_legacy_or_label_bearing_state_file_is_rejected(tmp_path: Path) -> None:
    state_dir = tmp_path / "states_twin"
    state_dir.mkdir()
    savemat(
        state_dir / "state_cache_train.mat",
        {
            "state_matrix": np.ones((3, 50)),
            "sample_id": np.arange(3),
            "target": np.asarray([100.0, 100.1, 99.9]),
        },
    )
    # A legacy artifact also has no formal manifest.  The missing manifest is
    # enough to reject it before any old cache can enter a formal experiment.
    with pytest.raises(FileNotFoundError, match="state manifest"):
        load_twin_state_cache(state_dir, "train")

    # When a forged manifest is added around a label-bearing MAT, the field
    # audit still rejects the embedded supervision.
    valid_state_dir, audit_dir = _write_contract_fixture(tmp_path / "forged")
    state_file = valid_state_dir / "state_cache_train.mat"
    payload = loadmat_for_test(state_file)
    payload["target"] = np.asarray([100.0, 100.1, 99.9])
    savemat(state_file, payload)
    manifest_path = valid_state_dir / "state_cache_train.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state_file_sha256"] = sha256_file(state_file)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden target/label"):
        load_twin_state_cache(valid_state_dir, "train", audit_dir=audit_dir)


def loadmat_for_test(path: Path) -> dict[str, object]:
    """Read user fields only so a synthetic MAT can be rewritten."""
    from scipy.io import loadmat

    return {
        key: value
        for key, value in loadmat(path).items()
        if not key.startswith("__")
    }


def test_state_file_hash_tampering_is_rejected(tmp_path: Path) -> None:
    state_dir, audit_dir = _write_contract_fixture(tmp_path)
    state_file = state_dir / "state_cache_train.mat"
    with state_file.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_twin_state_cache(state_dir, "train", audit_dir=audit_dir)


def test_numerically_failed_twin_audit_is_rejected(tmp_path: Path) -> None:
    state_dir, audit_dir = _write_contract_fixture(tmp_path, audit_failure=True)

    with pytest.raises(ValueError, match="numerical equivalence audit failed"):
        load_twin_state_cache(state_dir, "train", audit_dir=audit_dir)


@pytest.mark.parametrize(
    ("fixture_option", "message"),
    [
        ({"duplicate_sample_id": True}, "sample_id values must be unique"),
        (
            {"unequal_branch_parameters": True},
            "do not share one parameter hash",
        ),
    ],
)
def test_id_or_shared_parameter_contract_errors_are_rejected(
    tmp_path: Path,
    fixture_option: dict[str, bool],
    message: str,
) -> None:
    state_dir, audit_dir = _write_contract_fixture(tmp_path, **fixture_option)

    with pytest.raises(ValueError, match=message):
        load_twin_state_cache(state_dir, "train", audit_dir=audit_dir)
