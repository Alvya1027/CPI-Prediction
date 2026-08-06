"""Run the teacher-final explicit MATLAB Twin optical-reservoir experiment.

Validation consumes only the label-isolated train/validation data and the
audited MATLAB Twin state caches.  It freezes every readout/reference choice
and then writes a narrow authorization file that permits MATLAB to generate
the test state cache.  The test command can be run only once afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.teacher_shared_readout_pipeline import (
    DATA_DIR,
    FrozenTeacherConfig,
    MIN_GAP_MONTHS,
    PROFILE_ROOT,
    run_frozen_test,
    run_validation_search,
)
from src.twin_state_cache_contract import (
    STATE_PROTOCOL,
    load_twin_state_splits,
)
from src.config import RESULTS_DIR


TWIN_STATE_DIR = PROFILE_ROOT / "states_twin"
TWIN_INPUT_DIR = PROFILE_ROOT / "inputs_twin"
TWIN_OUTPUT_DIR = (
    RESULTS_DIR / "siamese_optical_mom_teacher_explicit_twin_20260802"
)
TEST_AUTHORIZATION_PATH = TWIN_INPUT_DIR / "test_generation_authorization.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    result = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{name} is not a SHA-256 digest")
    return result


def _validate_frozen_for_test_authorization(
    frozen: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Reject anything except an untouched explicit-Twin validation freeze."""
    if not isinstance(frozen, dict):
        raise ValueError("frozen configuration must contain one JSON object")
    if frozen.get("status") != "validation_frozen_not_tested":
        raise ValueError("configuration is not validation-frozen")
    for field in ("test_data_loaded", "test_state_loaded", "test_evaluated"):
        if frozen.get(field) is not False:
            raise ValueError(f"a valid freeze must declare {field}=false")
    if frozen.get("state_protocol") != STATE_PROTOCOL:
        raise ValueError("configuration is not from the explicit Twin protocol")
    if frozen.get("reservoir_parameters_trained") is not False:
        raise ValueError("the frozen experiment must keep reservoir parameters fixed")
    if frozen.get("only_output_weights_trained") is not True:
        raise ValueError("the frozen experiment must train only output weights")
    if frozen.get("accessible_original_train_months") != 50:
        raise ValueError("the explicit Twin freeze must use exactly 50 train months")
    if frozen.get("validation_targets") != 45:
        raise ValueError("the explicit Twin freeze must contain 45 validation targets")
    _require_sha256(frozen.get("data_manifest_sha256"), "data_manifest_sha256")

    raw_config = frozen.get("configuration")
    if not isinstance(raw_config, dict):
        raise ValueError("frozen configuration lacks the selected readout settings")
    try:
        config = FrozenTeacherConfig(**raw_config)
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen readout settings are incomplete or invalid") from exc
    try:
        alpha = float(config.alpha)
        pair_weight = float(config.pair_weight)
        absolute_alpha = float(config.absolute_alpha)
        k_references = int(config.k_references)
        min_gap_months = int(config.min_gap_months)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("frozen readout settings are not numeric") from exc
    if (
        not math.isfinite(alpha)
        or alpha < 0
        or not math.isfinite(pair_weight)
        or pair_weight <= 0
        or not math.isfinite(absolute_alpha)
        or absolute_alpha < 0
        or k_references <= 0
        or min_gap_months != MIN_GAP_MONTHS
        or config.aggregation not in {"mean", "inverse_distance"}
    ):
        raise ValueError("frozen readout settings violate the formal search contract")

    validation_metrics = frozen.get("validation_metrics")
    if not isinstance(validation_metrics, dict):
        raise ValueError("frozen configuration lacks validation metrics")
    for model in ("absolute_only", "teacher_pair"):
        metrics = validation_metrics.get(model)
        if not isinstance(metrics, dict):
            raise ValueError(f"frozen configuration lacks {model} validation metrics")
        for metric in ("mae", "rmse"):
            try:
                value = float(metrics[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid validation metric: {model}.{metric}") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid validation metric: {model}.{metric}")

    audit = frozen.get("train_validation_state_audit")
    if not isinstance(audit, dict) or audit.get("status") != "passed":
        raise ValueError("frozen Twin audit is missing or did not pass")
    if audit.get("state_protocol") != STATE_PROTOCOL:
        raise ValueError("frozen Twin audit uses a different state protocol")
    if audit.get("cache_reuse_declared") is not True:
        raise ValueError("frozen Twin audit does not declare cache reuse")
    if audit.get("cache_generated_by_explicit_twin_model") is not True:
        raise ValueError("frozen states were not generated by the explicit Twin model")
    if audit.get("semantic_pairs_simulated_simultaneously") is not False:
        raise ValueError("frozen audit makes a false simultaneous-pair claim")
    if audit.get("pair_states_resolved_by_sample_id") is not True:
        raise ValueError("frozen audit does not resolve pair states by sample ID")
    loaded_splits = audit.get("loaded_splits")
    if not isinstance(loaded_splits, list) or set(loaded_splits) != {"train", "val"}:
        raise ValueError("authorization requires an audit of train and validation only")
    splits = audit.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "val"}:
        raise ValueError("frozen audit must contain only train/validation state files")
    expected_rows = {"train": 50, "val": 45}
    for split, expected_count in expected_rows.items():
        split_audit = splits.get(split)
        if not isinstance(split_audit, dict):
            raise ValueError(f"frozen audit lacks split={split}")
        _require_sha256(split_audit.get("sha256"), f"{split} state sha256")
        shape = split_audit.get("shape")
        if shape != [expected_count, 50]:
            raise ValueError(f"frozen {split} state shape is not {[expected_count, 50]}")
    protocol_identity = _require_sha256(
        audit.get("protocol_identity_sha256"), "protocol_identity_sha256"
    )
    for field in (
        "audit_mat_sha256",
        "audit_json_sha256",
        "shared_branch_model_sha256",
        "twin_model_sha256",
        "simulation_protocol_sha256",
    ):
        _require_sha256(audit.get(field), field)
    cache_records = audit.get("cache_records")
    if not isinstance(cache_records, list) or not cache_records:
        raise ValueError("frozen Twin audit lacks cache-record provenance")
    if any(
        not isinstance(record, dict)
        or record.get("split") not in {"train", "val"}
        for record in cache_records
    ):
        raise ValueError("frozen cache provenance contains a test or invalid record")
    return frozen, audit


def write_test_generation_authorization(
    frozen_config_path: Path,
    authorization_path: Path = TEST_AUTHORIZATION_PATH,
    *,
    test_state_path: Path | None = None,
) -> Path:
    """Authorize only test-state generation after validation is frozen."""
    frozen_config_path = frozen_config_path.resolve()
    if not frozen_config_path.is_file():
        raise FileNotFoundError(f"missing frozen configuration: {frozen_config_path}")
    try:
        raw_frozen = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen configuration is not valid UTF-8 JSON") from exc
    frozen, audit = _validate_frozen_for_test_authorization(raw_frozen)
    protocol_identity = str(audit["protocol_identity_sha256"])
    if test_state_path is None:
        test_state_path = TWIN_STATE_DIR / "state_cache_test.mat"
    if test_state_path is not None and Path(test_state_path).exists():
        raise FileExistsError(
            "test state cache already exists before authorization: "
            f"{Path(test_state_path)}"
        )
    authorization = {
        "schema_version": "1.0",
        "status": "validation_frozen_authorized_for_test_state_generation",
        "created_at": datetime.now().astimezone().isoformat(),
        "allowed_split": "test",
        "state_protocol": STATE_PROTOCOL,
        "protocol_identity_sha256": protocol_identity,
        "frozen_config_path": str(frozen_config_path),
        "frozen_config_sha256": _sha256(frozen_config_path),
        "train_state_sha256": audit["splits"]["train"]["sha256"],
        "validation_state_sha256": audit["splits"]["val"]["sha256"],
        "audit_mat_sha256": audit["audit_mat_sha256"],
        "audit_json_sha256": audit["audit_json_sha256"],
        "shared_branch_model_sha256": audit["shared_branch_model_sha256"],
        "twin_model_sha256": audit["twin_model_sha256"],
        "simulation_protocol_sha256": audit["simulation_protocol_sha256"],
        "data_manifest_sha256": frozen["data_manifest_sha256"],
        "frozen_configuration": frozen["configuration"],
        "test_labels_accessed": False,
        "test_state_generated": False,
    }
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    if authorization_path.exists():
        try:
            existing = json.loads(authorization_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("existing test authorization is invalid") from exc
        immutable = {key: value for key, value in authorization.items() if key != "created_at"}
        existing_immutable = {
            key: existing.get(key) for key in immutable
        }
        if existing_immutable != immutable:
            raise FileExistsError(
                "a different or modified validation freeze owns the test authorization: "
                f"{authorization_path}"
            )
        return authorization_path
    authorization_path.write_text(
        json.dumps(authorization, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return authorization_path


def validate_test_generation_authorization(
    frozen_config_path: Path,
    authorization_path: Path,
    test_state_manifest_path: Path,
) -> dict[str, object]:
    """Bind a MATLAB test cache to this exact validation freeze."""
    frozen_config_path = Path(frozen_config_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    test_state_manifest_path = Path(test_state_manifest_path).resolve()
    try:
        frozen_raw = json.loads(frozen_config_path.read_text(encoding="utf-8"))
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        state_manifest = json.loads(
            test_state_manifest_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("freeze, authorization, and test manifest must be valid JSON") from exc
    frozen, audit = _validate_frozen_for_test_authorization(frozen_raw)
    if not isinstance(authorization, dict) or not isinstance(state_manifest, dict):
        raise ValueError("authorization and test state manifest must be JSON objects")

    expected_authorization = {
        "status": "validation_frozen_authorized_for_test_state_generation",
        "allowed_split": "test",
        "state_protocol": STATE_PROTOCOL,
        "protocol_identity_sha256": audit["protocol_identity_sha256"],
        "frozen_config_sha256": _sha256(frozen_config_path),
        "train_state_sha256": audit["splits"]["train"]["sha256"],
        "validation_state_sha256": audit["splits"]["val"]["sha256"],
        "audit_mat_sha256": audit["audit_mat_sha256"],
        "audit_json_sha256": audit["audit_json_sha256"],
        "shared_branch_model_sha256": audit["shared_branch_model_sha256"],
        "twin_model_sha256": audit["twin_model_sha256"],
        "simulation_protocol_sha256": audit["simulation_protocol_sha256"],
        "data_manifest_sha256": frozen["data_manifest_sha256"],
        "frozen_configuration": frozen["configuration"],
        "test_labels_accessed": False,
    }
    changed = [
        field
        for field, expected in expected_authorization.items()
        if authorization.get(field) != expected
    ]
    if changed:
        raise ValueError(
            "test-generation authorization disagrees with the frozen validation at: "
            + ", ".join(sorted(changed))
        )

    expected_manifest = {
        "split": "test",
        "state_protocol": STATE_PROTOCOL,
        "test_generation_authorization_file": authorization_path.name,
        "test_generation_authorization_sha256": _sha256(authorization_path),
        "test_generation_authorization_status": (
            "validation_frozen_authorized_for_test_state_generation"
        ),
        "protocol_identity_sha256": audit["protocol_identity_sha256"],
    }
    changed_manifest = [
        field
        for field, expected in expected_manifest.items()
        if state_manifest.get(field) != expected
    ]
    if changed_manifest:
        raise ValueError(
            "MATLAB test cache is not bound to this authorization at: "
            + ", ".join(sorted(changed_manifest))
        )
    return authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--state-dir", type=Path, default=TWIN_STATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=TWIN_OUTPUT_DIR)
    parser.add_argument(
        "--authorization-path",
        type=Path,
        default=TEST_AUTHORIZATION_PATH,
        help="Validation-freeze authorization consumed by MATLAB test generation.",
    )
    parser.add_argument(
        "--frozen-test",
        type=Path,
        help="Evaluate test once from a validation-frozen JSON config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frozen_test is None:
        config = run_validation_search(
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            state_dir=args.state_dir,
            state_loader=load_twin_state_splits,
            state_protocol=STATE_PROTOCOL,
            experiment_name=(
                "teacher-final explicit MATLAB Twin optical reservoir with "
                "shared linear output weights"
            ),
        )
        frozen_path = args.output_dir / "tables" / "selected_configuration.json"
        authorization = write_test_generation_authorization(
            frozen_path,
            args.authorization_path,
            test_state_path=args.state_dir / "state_cache_test.mat",
        )
        print(f"Validation configuration frozen: {config}")
        print(frozen_path)
        print(f"MATLAB test-state authorization: {authorization}")
    else:
        validate_test_generation_authorization(
            args.frozen_test,
            args.authorization_path,
            args.state_dir / "state_cache_test.manifest.json",
        )
        comparison = run_frozen_test(
            args.frozen_test,
            data_dir=args.data_dir,
            state_dir=args.state_dir,
            state_loader=load_twin_state_splits,
            expected_state_protocol=STATE_PROTOCOL,
        )
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
