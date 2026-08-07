"""Run the explicit Twin train45/test47 experiment without a validation split.

The readout/reference configuration is declared in this file before any test
state or label is opened.  The first invocation loads only train45, fits and
freezes the output layer contract, and authorizes MATLAB test-state generation.
The ``--frozen-test`` invocation evaluates the unchanged test47 exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR, ROOT_DIR
from src.siamese_split_isolation import load_isolated_split
from src.teacher_shared_readout import fit_absolute_ridge, fit_joint_shared_readout
from src.teacher_shared_readout_pipeline import (
    FrozenTeacherConfig,
    _metrics,
    _save_test_figures,
    _state_matrix,
    attach_cache_provenance,
    build_evaluation_candidates,
    build_train_pairs,
    predict_direct_split,
    predict_reference_split,
    select_references,
)
from src.twin_state_cache_contract import STATE_PROTOCOL, load_twin_state_splits


PROFILE_ROOT = (
    ROOT_DIR / "matlab" / "optical_reservoir_cpi_mom_train45_noval_20260807"
)
DATA_DIR = PROFILE_ROOT / "data"
STATE_DIR = PROFILE_ROOT / "states_twin"
INPUT_DIR = PROFILE_ROOT / "inputs_twin"
OUTPUT_DIR = RESULTS_DIR / "siamese_optical_mom_teacher_twin_train45_noval_20260807"
AUTHORIZATION_PATH = INPUT_DIR / "test_generation_authorization.json"

# With no independent validation set, these values are immutable protocol
# constants.  Changing them after viewing test output starts a new experiment.
FIXED_CONFIG = FrozenTeacherConfig(
    alpha=100.0,
    pair_weight=0.1,
    k_references=5,
    aggregation="mean",
    min_gap_months=1,
    absolute_alpha=2.0,
)
TRAIN_COUNT = 45
TEST_COUNT = 47
TRAIN_FIRST = "2018-09"
TRAIN_LAST = "2022-05"
TEST_FIRST = "2022-06"
TEST_LAST = "2026-04"
FREEZE_STATUS = "train_only_fixed_not_tested"
AUTHORIZATION_STATUS = "train_only_fixed_authorized_for_test_state_generation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    """Write audited JSON with stable UTF-8/LF bytes on every platform."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_bytes(serialized.encode("utf-8"))


def _load_expected_split(data_dir: Path, split: str) -> dict[str, object]:
    payload = load_isolated_split(data_dir, split)
    expected = {
        "train": (TRAIN_COUNT, TRAIN_FIRST, TRAIN_LAST),
        "test": (TEST_COUNT, TEST_FIRST, TEST_LAST),
    }[split]
    index = payload["index"]
    if len(index) != expected[0]:
        raise ValueError(f"{split} has {len(index)} rows, expected {expected[0]}")
    if (
        str(index["target_date"].iloc[0]) != expected[1]
        or str(index["target_date"].iloc[-1]) != expected[2]
    ):
        raise ValueError(f"{split} target dates do not match the frozen boundary")
    return payload


def _validate_train_audit(audit: object) -> dict[str, object]:
    if not isinstance(audit, dict) or audit.get("status") != "passed":
        raise ValueError("train-only explicit Twin audit is missing or failed")
    if audit.get("state_protocol") != STATE_PROTOCOL:
        raise ValueError("train-only audit uses the wrong state protocol")
    if audit.get("loaded_splits") != ["train"]:
        raise ValueError("pre-test freeze may load only the train state cache")
    splits = audit.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train"}:
        raise ValueError("pre-test audit must contain only train states")
    if splits["train"].get("shape") != [TRAIN_COUNT, 50]:
        raise ValueError("train state cache must have shape 45 x 50")
    for field in (
        "protocol_identity_sha256",
        "audit_mat_sha256",
        "audit_json_sha256",
        "shared_branch_model_sha256",
        "twin_model_sha256",
        "simulation_protocol_sha256",
    ):
        value = str(audit.get(field, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"train audit lacks a valid {field}")
    return audit


def freeze_train_only(
    *, data_dir: Path = DATA_DIR, state_dir: Path = STATE_DIR, output_dir: Path = OUTPUT_DIR
) -> Path:
    """Fit the predeclared configuration without opening test data or states."""
    train = _load_expected_split(data_dir, "train")
    state_lookup, raw_audit = load_twin_state_splits(state_dir, {"train": train})
    audit = _validate_train_audit(raw_audit)
    train_ids = train["index"]["sample_id"].to_numpy(dtype=int)
    train_states = _state_matrix(train_ids, state_lookup)
    train_targets = np.asarray(train["y"], dtype=np.float64).reshape(-1)
    train_pairs, pair_i, pair_j = build_train_pairs(
        train, min_gap_months=FIXED_CONFIG.min_gap_months
    )
    train_pairs = attach_cache_provenance(train_pairs, audit)
    if len(train_pairs) != 561:
        raise ValueError(f"train45 gap=1 must produce 561 relations, got {len(train_pairs)}")

    absolute_model = fit_absolute_ridge(
        train_states, train_targets, alpha=FIXED_CONFIG.absolute_alpha
    )
    teacher_model = fit_joint_shared_readout(
        train_states,
        train_targets,
        pair_i,
        pair_j,
        alpha=FIXED_CONFIG.alpha,
        pair_weight=FIXED_CONFIG.pair_weight,
        absolute_weight=1.0,
    )
    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_pairs.to_csv(table_dir / "train_pair_relations.csv", index=False)
    np.savez_compressed(
        model_dir / "teacher_shared_readout_train45.npz",
        **teacher_model.to_npz_dict(),
    )
    np.savez_compressed(
        model_dir / "absolute_readout_train45.npz", **absolute_model.to_npz_dict()
    )

    frozen = {
        "schema_version": "1.0",
        "reporting_status": "teacher_explicit_twin_train45_no_validation_frozen",
        "status": FREEZE_STATUS,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": "teacher explicit Twin train45/test47 without validation split",
        "state_protocol": STATE_PROTOCOL,
        "split_protocol": "train45_no_validation_test47_20260807",
        "configuration": asdict(FIXED_CONFIG),
        "configuration_policy": (
            "predeclared constants; no test data, test state, or test metric selected them"
        ),
        "training_targets": TRAIN_COUNT,
        "training_target_range": [TRAIN_FIRST, TRAIN_LAST],
        "validation_targets": 0,
        "test_targets_declared": TEST_COUNT,
        "test_target_range": [TEST_FIRST, TEST_LAST],
        "derived_train_pair_relations": int(len(train_pairs)),
        "derived_train_pair_target_months": int(
            train_pairs["sample_i_id"].nunique()
        ),
        "test_data_loaded": False,
        "test_state_loaded": False,
        "test_evaluated": False,
        "reservoir_parameters_trained": False,
        "only_output_weights_trained": True,
        "train_state_audit": audit,
        "data_manifest_sha256": _sha256(data_dir / "isolated_split_manifest.json"),
    }
    frozen_path = table_dir / "fixed_configuration.json"
    _write_json(frozen_path, frozen)
    _write_json(output_dir / "experiment_manifest.json", frozen)
    return frozen_path


def _load_frozen(path: Path) -> dict[str, object]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("status") != FREEZE_STATUS:
        raise ValueError("configuration is not an untested train-only freeze")
    if frozen.get("validation_targets") != 0 or frozen.get("training_targets") != 45:
        raise ValueError("configuration does not describe train45 with no validation")
    if frozen.get("state_protocol") != STATE_PROTOCOL:
        raise ValueError("configuration uses the wrong state protocol")
    if any(frozen.get(field) is not False for field in (
        "test_data_loaded", "test_state_loaded", "test_evaluated"
    )):
        raise ValueError("frozen configuration records premature test access")
    if FrozenTeacherConfig(**frozen["configuration"]) != FIXED_CONFIG:
        raise ValueError("frozen configuration differs from the predeclared constants")
    _validate_train_audit(frozen.get("train_state_audit"))
    return frozen


def write_test_authorization(
    frozen_path: Path, authorization_path: Path = AUTHORIZATION_PATH
) -> Path:
    frozen_path = frozen_path.resolve()
    frozen = _load_frozen(frozen_path)
    audit = frozen["train_state_audit"]
    test_state_path = STATE_DIR / "state_cache_test.mat"
    if test_state_path.exists():
        raise FileExistsError(f"test state already exists before authorization: {test_state_path}")
    authorization = {
        "schema_version": "1.0",
        "status": AUTHORIZATION_STATUS,
        "created_at": datetime.now().astimezone().isoformat(),
        "allowed_split": "test",
        "state_protocol": STATE_PROTOCOL,
        "split_protocol": "train45_no_validation_test47_20260807",
        "protocol_identity_sha256": audit["protocol_identity_sha256"],
        "frozen_config_path": str(frozen_path),
        "frozen_config_sha256": _sha256(frozen_path),
        "train_state_sha256": audit["splits"]["train"]["sha256"],
        "audit_mat_sha256": audit["audit_mat_sha256"],
        "audit_json_sha256": audit["audit_json_sha256"],
        "shared_branch_model_sha256": audit["shared_branch_model_sha256"],
        "twin_model_sha256": audit["twin_model_sha256"],
        "simulation_protocol_sha256": audit["simulation_protocol_sha256"],
        "data_manifest_sha256": frozen["data_manifest_sha256"],
        "frozen_configuration": frozen["configuration"],
        "validation_split_used": False,
        "test_labels_accessed": False,
        "test_state_generated": False,
    }
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(authorization_path, authorization)
    return authorization_path


def _validate_authorization(
    frozen_path: Path, authorization_path: Path, test_manifest_path: Path
) -> dict[str, object]:
    frozen = _load_frozen(frozen_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    audit = frozen["train_state_audit"]
    expected = {
        "status": AUTHORIZATION_STATUS,
        "allowed_split": "test",
        "state_protocol": STATE_PROTOCOL,
        "protocol_identity_sha256": audit["protocol_identity_sha256"],
        "frozen_config_sha256": _sha256(frozen_path),
        "train_state_sha256": audit["splits"]["train"]["sha256"],
        "audit_mat_sha256": audit["audit_mat_sha256"],
        "audit_json_sha256": audit["audit_json_sha256"],
        "shared_branch_model_sha256": audit["shared_branch_model_sha256"],
        "twin_model_sha256": audit["twin_model_sha256"],
        "simulation_protocol_sha256": audit["simulation_protocol_sha256"],
        "data_manifest_sha256": frozen["data_manifest_sha256"],
        "frozen_configuration": frozen["configuration"],
        "validation_split_used": False,
        "test_labels_accessed": False,
    }
    changed = [key for key, value in expected.items() if authorization.get(key) != value]
    if changed:
        raise ValueError("test authorization changed at: " + ", ".join(changed))
    manifest_expected = {
        "split": "test",
        "state_protocol": STATE_PROTOCOL,
        "test_generation_authorization_file": authorization_path.name,
        "test_generation_authorization_sha256": _sha256(authorization_path),
        "test_generation_authorization_status": AUTHORIZATION_STATUS,
        "protocol_identity_sha256": audit["protocol_identity_sha256"],
    }
    changed = [
        key for key, value in manifest_expected.items() if manifest.get(key) != value
    ]
    if changed:
        raise ValueError("test state manifest changed at: " + ", ".join(changed))
    return frozen


def evaluate_frozen_test(
    frozen_path: Path,
    *,
    authorization_path: Path = AUTHORIZATION_PATH,
    data_dir: Path = DATA_DIR,
    state_dir: Path = STATE_DIR,
) -> pd.DataFrame:
    frozen_path = frozen_path.resolve()
    frozen = _validate_authorization(
        frozen_path,
        authorization_path.resolve(),
        state_dir / "state_cache_test.manifest.json",
    )
    output_dir = frozen_path.parent.parent
    completion_path = output_dir / "test_evaluation_manifest.json"
    if completion_path.exists():
        raise FileExistsError("this frozen configuration was already tested")
    if frozen["data_manifest_sha256"] != _sha256(
        data_dir / "isolated_split_manifest.json"
    ):
        raise ValueError("isolated data changed after train-only freeze")

    train = _load_expected_split(data_dir, "train")
    test = _load_expected_split(data_dir, "test")
    state_lookup, state_audit = load_twin_state_splits(
        state_dir, {"train": train, "test": test}
    )
    frozen_audit = frozen["train_state_audit"]
    if (
        state_audit["splits"]["train"]["sha256"]
        != frozen_audit["splits"]["train"]["sha256"]
        or state_audit["protocol_identity_sha256"]
        != frozen_audit["protocol_identity_sha256"]
    ):
        raise ValueError("train states or Twin protocol changed after freeze")

    config = FrozenTeacherConfig(**frozen["configuration"])
    train_ids = train["index"]["sample_id"].to_numpy(dtype=int)
    train_states = _state_matrix(train_ids, state_lookup)
    train_targets = np.asarray(train["y"], dtype=np.float64).reshape(-1)
    train_pairs, pair_i, pair_j = build_train_pairs(
        train, min_gap_months=config.min_gap_months
    )
    absolute_model = fit_absolute_ridge(
        train_states, train_targets, alpha=config.absolute_alpha
    )
    teacher_model = fit_joint_shared_readout(
        train_states,
        train_targets,
        pair_i,
        pair_j,
        alpha=config.alpha,
        pair_weight=config.pair_weight,
        absolute_weight=1.0,
    )
    absolute_predictions = predict_direct_split(absolute_model, test, state_lookup)
    joint_direct_predictions = predict_direct_split(teacher_model, test, state_lookup)
    candidates = build_evaluation_candidates(test, train, config.min_gap_months)
    selected = attach_cache_provenance(
        select_references(candidates, config.k_references), state_audit
    )
    pair_predictions, teacher_predictions = predict_reference_split(
        teacher_model, selected, state_lookup, config.aggregation
    )
    if len(teacher_predictions) != TEST_COUNT:
        raise ValueError("test prediction table must contain 47 targets")

    absolute_metrics = _metrics(absolute_predictions)
    direct_metrics = _metrics(joint_direct_predictions)
    teacher_metrics = _metrics(teacher_predictions)
    comparison = pd.DataFrame(
        [
            {"model": "absolute_only_same_states", **{f"test_{k}": v for k, v in absolute_metrics.items()}},
            {"model": "joint_shared_readout_direct_diagnostic", **{f"test_{k}": v for k, v in direct_metrics.items()}},
            {"model": "teacher_shared_readout_pair", **{f"test_{k}": v for k, v in teacher_metrics.items()}},
        ]
    )
    unified = absolute_predictions[
        ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
    ].rename(columns={"cpi_predicted": "cpi_predicted_absolute"})
    unified = unified.merge(
        joint_direct_predictions[
            ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
        ].rename(columns={"cpi_predicted": "cpi_predicted_joint_direct"}),
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    ).merge(
        teacher_predictions[
            ["sample_i_id", "target_date", "cpi_actual", "cpi_predicted"]
        ].rename(columns={"cpi_predicted": "cpi_predicted_teacher_pair"}),
        on=["sample_i_id", "target_date", "cpi_actual"],
        validate="one_to_one",
    )
    for suffix in ("absolute", "joint_direct", "teacher_pair"):
        unified[f"residual_{suffix}"] = (
            unified[f"cpi_predicted_{suffix}"] - unified["cpi_actual"]
        )
        unified[f"absolute_error_{suffix}"] = unified[f"residual_{suffix}"].abs()

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(table_dir / "test_model_comparison.csv", index=False)
    unified.to_csv(table_dir / "test_prediction_comparison.csv", index=False)
    selected.to_csv(table_dir / "selected_test_references.csv", index=False)
    pair_predictions.to_csv(table_dir / "selected_test_pair_predictions.csv", index=False)
    np.savez_compressed(
        model_dir / "teacher_shared_readout_final.npz", **teacher_model.to_npz_dict()
    )
    np.savez_compressed(
        model_dir / "absolute_readout_final.npz", **absolute_model.to_npz_dict()
    )
    _save_test_figures(unified, comparison, figure_dir)

    completion = {
        **frozen,
        "status": "train_only_fixed_then_test_evaluated_once",
        "test_evaluated_at": datetime.now().astimezone().isoformat(),
        "test_data_loaded": True,
        "test_state_loaded": True,
        "test_evaluated": True,
        "test_state_audit": state_audit,
        "test_metrics": {
            "absolute_only": absolute_metrics,
            "joint_direct_diagnostic": direct_metrics,
            "teacher_pair": teacher_metrics,
        },
        "all_test_references_from_train45": True,
        "test_labels_used_for_reference_selection": False,
    }
    _write_json(completion_path, completion)
    _write_json(output_dir / "experiment_manifest.json", completion)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--authorization-path", type=Path, default=AUTHORIZATION_PATH)
    parser.add_argument("--frozen-test", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frozen_test is None:
        frozen_path = freeze_train_only(
            data_dir=args.data_dir, state_dir=args.state_dir, output_dir=args.output_dir
        )
        authorization = write_test_authorization(
            frozen_path, args.authorization_path
        )
        print(f"Train-only configuration frozen: {frozen_path}")
        print(f"MATLAB test-state authorization: {authorization}")
    else:
        comparison = evaluate_frozen_test(
            args.frozen_test,
            authorization_path=args.authorization_path,
            data_dir=args.data_dir,
            state_dir=args.state_dir,
        )
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
