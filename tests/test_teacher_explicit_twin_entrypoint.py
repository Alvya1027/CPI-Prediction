from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_teacher_explicit_twin_mom_closed50 import (
    validate_test_generation_authorization,
    write_test_generation_authorization,
)
from src.teacher_shared_readout_pipeline import (
    attach_cache_provenance,
    load_state_splits,
    run_frozen_test,
    run_validation_search,
)
from src.twin_state_cache_contract import STATE_PROTOCOL


def _hash(character: str) -> str:
    return character * 64


def _valid_frozen_payload() -> dict[str, object]:
    return {
        "status": "validation_frozen_not_tested",
        "state_protocol": STATE_PROTOCOL,
        "configuration": {
            "alpha": 0.2,
            "pair_weight": 0.3,
            "k_references": 3,
            "aggregation": "mean",
            "min_gap_months": 1,
            "absolute_alpha": 2.0,
        },
        "accessible_original_train_months": 50,
        "validation_targets": 45,
        "test_data_loaded": False,
        "test_state_loaded": False,
        "test_evaluated": False,
        "reservoir_parameters_trained": False,
        "only_output_weights_trained": True,
        "data_manifest_sha256": _hash("d"),
        "validation_metrics": {
            "absolute_only": {"mae": 0.4, "rmse": 0.5},
            "teacher_pair": {"mae": 0.3, "rmse": 0.45},
        },
        "train_validation_state_audit": {
            "status": "passed",
            "state_protocol": STATE_PROTOCOL,
            "protocol_identity_sha256": _hash("1"),
            "audit_mat_sha256": _hash("2"),
            "audit_json_sha256": _hash("3"),
            "shared_branch_model_sha256": _hash("4"),
            "twin_model_sha256": _hash("5"),
            "simulation_protocol_sha256": _hash("6"),
            "cache_reuse_declared": True,
            "cache_generated_by_explicit_twin_model": True,
            "semantic_pairs_simulated_simultaneously": False,
            "pair_states_resolved_by_sample_id": True,
            "loaded_splits": ["train", "val"],
            "splits": {
                "train": {"sha256": _hash("a"), "shape": [50, 50]},
                "val": {"sha256": _hash("b"), "shape": [45, 50]},
            },
            "cache_records": [
                {"split": "train", "sample_id": 10},
                {"split": "val", "sample_id": 20},
            ],
        },
    }


def test_authorization_is_written_only_for_a_valid_untested_freeze(
    tmp_path: Path,
) -> None:
    frozen_path = tmp_path / "selected_configuration.json"
    frozen_path.write_text(
        json.dumps(_valid_frozen_payload()), encoding="utf-8"
    )
    authorization_path = tmp_path / "inputs_twin" / "authorization.json"

    written = write_test_generation_authorization(
        frozen_path,
        authorization_path,
        test_state_path=tmp_path / "states_twin" / "state_cache_test.mat",
    )

    assert written == authorization_path
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    assert authorization["allowed_split"] == "test"
    assert authorization["protocol_identity_sha256"] == _hash("1")
    assert authorization["frozen_config_sha256"]
    assert authorization["test_labels_accessed"] is False
    assert authorization["test_state_generated"] is False


@pytest.mark.parametrize("invalid_case", ["status", "test_access", "test_in_audit"])
def test_invalid_freeze_cannot_create_test_authorization(
    tmp_path: Path,
    invalid_case: str,
) -> None:
    frozen = copy.deepcopy(_valid_frozen_payload())
    if invalid_case == "status":
        frozen["status"] = "draft"
    elif invalid_case == "test_access":
        frozen["test_state_loaded"] = True
    else:
        audit = frozen["train_validation_state_audit"]
        assert isinstance(audit, dict)
        audit["loaded_splits"] = ["train", "val", "test"]
    frozen_path = tmp_path / f"{invalid_case}.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    authorization_path = tmp_path / "inputs_twin" / "authorization.json"

    with pytest.raises(ValueError):
        write_test_generation_authorization(frozen_path, authorization_path)

    assert not authorization_path.exists()


def test_test_state_manifest_is_bound_to_the_exact_freeze_authorization(
    tmp_path: Path,
) -> None:
    frozen_path = tmp_path / "selected_configuration.json"
    frozen_path.write_text(json.dumps(_valid_frozen_payload()), encoding="utf-8")
    authorization_path = tmp_path / "inputs_twin" / "authorization.json"
    write_test_generation_authorization(
        frozen_path,
        authorization_path,
        test_state_path=tmp_path / "states_twin" / "state_cache_test.mat",
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / "states_twin" / "state_cache_test.manifest.json"
    manifest_path.parent.mkdir(parents=True)
    from src.twin_state_cache_contract import sha256_file

    manifest_path.write_text(
        json.dumps(
            {
                "split": "test",
                "state_protocol": STATE_PROTOCOL,
                "test_generation_authorization_file": authorization_path.name,
                "test_generation_authorization_sha256": sha256_file(
                    authorization_path
                ),
                "test_generation_authorization_status": authorization["status"],
                "protocol_identity_sha256": authorization[
                    "protocol_identity_sha256"
                ],
            }
        ),
        encoding="utf-8",
    )

    validated = validate_test_generation_authorization(
        frozen_path, authorization_path, manifest_path
    )

    assert validated["frozen_config_sha256"] == authorization[
        "frozen_config_sha256"
    ]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_identity_sha256"] = _hash("9")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        validate_test_generation_authorization(
            frozen_path, authorization_path, manifest_path
        )


def test_pair_tables_keep_both_cache_origins_without_a_fake_pair_run() -> None:
    pairs = pd.DataFrame(
        {
            "sample_i_id": [20],
            "sample_j_id": [10],
            "target_i_date": ["2020-01"],
            "target_j_date": ["2017-01"],
        }
    )
    state_audit = {
        "cache_reuse_declared": True,
        "semantic_pairs_simulated_simultaneously": False,
        "cache_records": [
            {
                "split": "train",
                "sample_id": 10,
                "cache_record_id": "train:10",
                "branch_id": "target",
                "partner_sample_id": 11,
                "generation_run_id": "train-run-1",
                "state_file_sha256": _hash("a"),
            },
            {
                "split": "val",
                "sample_id": 20,
                "cache_record_id": "val:20",
                "branch_id": "reference",
                "partner_sample_id": 21,
                "generation_run_id": "val-run-2",
                "state_file_sha256": _hash("b"),
            },
        ],
    }

    result = attach_cache_provenance(pairs, state_audit)

    assert result.loc[0, "semantic_pair_id"] == "val:20__train:10"
    assert result.loc[0, "i_cache_record_id"] == "val:20"
    assert result.loc[0, "j_cache_record_id"] == "train:10"
    assert result.loc[0, "i_generation_run_id"] == "val-run-2"
    assert result.loc[0, "j_generation_run_id"] == "train-run-1"
    assert result.loc[0, "semantic_pair_simulated_simultaneously"] == False
    assert "simultaneous_run_id" not in result.columns
    assert "pair_simulation_run_id" not in result.columns


def test_default_serial_pipeline_keeps_legacy_loader_and_pair_shape() -> None:
    validation_default = inspect.signature(run_validation_search).parameters[
        "state_loader"
    ].default
    test_default = inspect.signature(run_frozen_test).parameters["state_loader"].default
    assert validation_default is load_state_splits
    assert test_default is load_state_splits

    legacy_pairs = pd.DataFrame({"sample_i_id": [2], "sample_j_id": [1]})
    returned = attach_cache_provenance(
        legacy_pairs, {"state_protocol": "continuous_serial_shared_fixed_reservoir"}
    )
    assert returned is legacy_pairs
    pd.testing.assert_frame_equal(returned, legacy_pairs)
