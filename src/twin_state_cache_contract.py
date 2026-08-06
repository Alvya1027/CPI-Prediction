"""Strict contract for explicit-twin MATLAB optical-reservoir state caches.

This module deliberately does not accept the legacy ``states_<split>.mat``
files.  A formal twin cache must have been produced by the explicit two-branch
Simulink model, must declare that unique-window states are reused by ID, and
must be accompanied by cryptographic provenance and a numerical branch-
equivalence audit.

The accepted protocol is honest about the implementation boundary: MATLAB
executes two shared-model branches, but semantic target/reference relations
are assembled later from one cached state per unique window.  It therefore
must *not* claim that every semantic pair was simulated in the same run.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat


STATE_PROTOCOL = "explicit_twin_audited_unique_window_cache_v1"
SCHEMA_VERSION = "1.0"
VALID_SPLITS = ("train", "val", "test")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_FORBIDDEN_EXACT_FIELDS = {
    "actual",
    "cpi_actual",
    "cpi_i",
    "cpi_j",
    "cpi_target",
    "delta_cpi",
    "label",
    "labels",
    "target",
    "target_scaled",
    "target_value",
    "y",
    "y_i",
    "y_j",
    "y_scaled",
}

_REQUIRED_STATE_FIELDS = {
    "schema_version",
    "state_protocol",
    "split",
    "state_matrix",
    "sample_id",
    "target_date",
    "input_window_raw",
    "input_window_scaled",
    "cache_record_id",
    "branch_id",
    "partner_sample_id",
    "generation_run_id",
    "state_mode",
    "repeat_count",
    "capture_cycle",
    "sample_times_seconds",
    "input_transform_sha256",
    "simulation_protocol_sha256",
    "shared_branch_model_sha256",
    "twin_model_sha256",
}

_REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "state_protocol",
    "split",
    "state_file",
    "state_file_sha256",
    "audit_mat_file",
    "audit_mat_sha256",
    "audit_json_file",
    "audit_json_sha256",
    "source_reservoir_model_sha256",
    "shared_branch_model_sha256",
    "twin_model_sha256",
    "branch_a_model_sha256",
    "branch_b_model_sha256",
    "branch_a_parameter_sha256",
    "branch_b_parameter_sha256",
    "branch_a_mask_sha256",
    "branch_b_mask_sha256",
    "input_transform_sha256",
    "simulation_protocol_sha256",
    "generator_script_sha256",
    "input_file_sha256",
    "initial_condition_inventory_sha256",
    "cache_reuse_declared",
    "cache_generated_by_explicit_twin_model",
    "semantic_pairs_simulated_simultaneously",
    "pair_states_resolved_by_sample_id",
    "derived_pair_states_from_cache",
    "same_model_reference_for_both_branches",
    "no_cross_branch_reservoir_coupling",
    "reset_between_runs",
    "state_mode",
    "sample_phase",
    "num_records",
    "state_width",
    "window_size",
    "repeat_count",
    "capture_cycle",
    "theta_seconds",
    "feedback_delay_seconds",
    "input_transport_delay_seconds",
    "noise_seed",
    "solver",
    "fixed_step_seconds",
    "matlab_release",
    "simulink_version",
}

_REQUIRED_AUDIT_FIELDS = {
    "schema_version",
    "state_protocol",
    "simulation_protocol_sha256",
    "shared_branch_model_sha256",
    "twin_model_sha256",
    "h_a_ab_branch_a",
    "h_b_ab_branch_b",
    "h_b_ba_branch_a",
    "h_a_ba_branch_b",
    "h_a_aa_branch_a",
    "h_a_aa_branch_b",
    "h_a_repeat_branch_a",
    "h_a_repeat_branch_b",
    "h_a_cache",
    "h_b_cache",
    "reservoir_parameter_sha256",
    "mask_sha256",
    "input_transform_sha256",
    "state_file_sha256",
}

_REQUIRED_AUDIT_JSON_FIELDS = {
    "schema_version",
    "state_protocol",
    "audit_mat_sha256",
    "simulation_protocol_sha256",
    "shared_branch_model_sha256",
    "twin_model_sha256",
    "audit_atol",
    "audit_rtol",
    "audit_passed",
    "reservoir_parameter_sha256",
    "mask_sha256",
    "input_transform_sha256",
    "state_file_sha256",
}

_PINNED_MANIFEST_FIELDS = (
    "state_protocol",
    "audit_mat_sha256",
    "audit_json_sha256",
    "source_reservoir_model_sha256",
    "shared_branch_model_sha256",
    "twin_model_sha256",
    "branch_a_model_sha256",
    "branch_b_model_sha256",
    "branch_a_parameter_sha256",
    "branch_b_parameter_sha256",
    "branch_a_mask_sha256",
    "branch_b_mask_sha256",
    "input_transform_sha256",
    "simulation_protocol_sha256",
    "initial_condition_inventory_sha256",
    "generator_script_sha256",
    "state_mode",
    "sample_phase",
    "repeat_count",
    "capture_cycle",
    "theta_seconds",
    "feedback_delay_seconds",
    "input_transport_delay_seconds",
    "noise_seed",
    "solver",
    "fixed_step_seconds",
    "matlab_release",
    "simulink_version",
    "state_width",
    "window_size",
)


@dataclass(frozen=True)
class TwinStateCache:
    """A validated state lookup plus row-level generation provenance."""

    split: str
    state_lookup: Mapping[int, np.ndarray]
    provenance: pd.DataFrame
    manifest: Mapping[str, object]
    audit_metrics: Mapping[str, Mapping[str, float | bool]]
    state_file: Path
    state_file_sha256: str
    audit_mat_file: Path
    audit_json_file: Path

    @property
    def sample_ids(self) -> np.ndarray:
        return self.provenance["sample_id"].to_numpy(dtype=np.int64)

    @property
    def state_width(self) -> int:
        return int(self.manifest["state_width"])


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for an existing file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _require_fields(payload: Mapping[str, object], required: set[str], label: str) -> None:
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{label} lacks required fields: {sorted(missing)}")


def _scalar_string(value: object, name: str) -> str:
    raw = np.asarray(value)
    if raw.size != 1:
        raise ValueError(f"{name} must be a scalar string")
    item = raw.reshape(-1)[0]
    while isinstance(item, np.ndarray):
        if item.size != 1:
            raise ValueError(f"{name} must be a scalar string")
        item = item.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    result = str(item).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _string_vector(value: object, name: str, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"U", "S"} and raw.ndim == 2 and raw.shape[0] == length:
        values = ["".join(str(part) for part in row).strip() for row in raw]
    else:
        flat = raw.reshape(-1)
        values = []
        for row, item in enumerate(flat):
            try:
                values.append(_scalar_string(item, f"{name}[{row}]") )
            except ValueError as exc:
                raise ValueError(f"invalid {name}") from exc
    if len(values) != length or any(not value for value in values):
        raise ValueError(f"{name} must contain {length} non-empty strings")
    return np.asarray(values, dtype=object)


def _identifier_vector(value: object, name: str, length: int) -> np.ndarray:
    """Normalize numeric or string run/cache identifiers to non-empty strings."""
    raw = np.asarray(value).reshape(-1)
    if len(raw) != length:
        raise ValueError(f"{name} must contain {length} values")
    result: list[str] = []
    for row, item in enumerate(raw):
        nested = np.asarray(item)
        if nested.size != 1:
            raise ValueError(f"{name}[{row}] must be scalar")
        scalar = nested.reshape(-1)[0]
        if isinstance(scalar, bytes):
            scalar = scalar.decode("utf-8")
        if isinstance(scalar, (float, np.floating)):
            if not np.isfinite(scalar) or scalar != np.rint(scalar):
                raise ValueError(f"{name}[{row}] is not a finite integer/string")
            text = str(int(scalar))
        elif isinstance(scalar, (int, np.integer)):
            text = str(int(scalar))
        else:
            text = str(scalar).strip()
        if not text:
            raise ValueError(f"{name}[{row}] must not be empty")
        result.append(text)
    return np.asarray(result, dtype=object)


def _integer_vector(value: object, name: str, length: int | None = None) -> np.ndarray:
    raw = np.asarray(value).reshape(-1)
    if length is not None and len(raw) != length:
        raise ValueError(f"{name} must contain {length} values")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
        raise ValueError(f"{name} must contain finite integers")
    return numeric.astype(np.int64)


def _matrix(value: object, name: str, rows: int, columns: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1 and rows == 1 and len(result) == columns:
        result = result.reshape(1, columns)
    if result.shape != (rows, columns):
        raise ValueError(f"{name} has shape {result.shape}; expected {(rows, columns)}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def _positive_float(value: object, name: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _positive_integer(value: object, name: str) -> int:
    numeric = _positive_float(value, name)
    if numeric != np.rint(numeric):
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _require_sha256(payload: Mapping[str, object], field: str, label: str) -> str:
    value = str(payload[field]).strip().lower()
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label}.{field} is not a SHA-256 digest")
    return value


def _forbidden_label_fields(mat_payload: Mapping[str, object]) -> list[str]:
    forbidden: list[str] = []
    for field in mat_payload:
        if field.startswith("__"):
            continue
        normalized = field.strip().lower()
        if normalized in _FORBIDDEN_EXACT_FIELDS:
            forbidden.append(field)
        elif normalized.startswith("y_"):
            forbidden.append(field)
        elif normalized.startswith("label"):
            forbidden.append(field)
        elif normalized.startswith("cpi_"):
            forbidden.append(field)
        elif normalized.startswith("target_") and normalized != "target_date":
            forbidden.append(field)
    return sorted(set(forbidden))


def _validate_manifest_shape(manifest: Mapping[str, object], split: str) -> None:
    _require_fields(manifest, _REQUIRED_MANIFEST_FIELDS, "state manifest")
    if str(manifest["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unsupported state manifest schema_version")
    if manifest["state_protocol"] != STATE_PROTOCOL:
        raise ValueError("state manifest does not use the explicit twin cache protocol")
    if manifest["split"] != split:
        raise ValueError("state manifest split does not match the requested split")
    if manifest["state_mode"] != "isolated_repeated_window":
        raise ValueError("formal twin cache state_mode must be isolated_repeated_window")
    if manifest["sample_phase"] != "node_end":
        raise ValueError("formal twin cache sample_phase must be node_end")

    truth_requirements = {
        "cache_reuse_declared": True,
        "cache_generated_by_explicit_twin_model": True,
        "semantic_pairs_simulated_simultaneously": False,
        "pair_states_resolved_by_sample_id": True,
        "derived_pair_states_from_cache": True,
        "same_model_reference_for_both_branches": True,
        "no_cross_branch_reservoir_coupling": True,
        "reset_between_runs": True,
    }
    for field, expected in truth_requirements.items():
        if manifest[field] is not expected:
            raise ValueError(f"state manifest must declare {field}={expected}")

    hash_fields = [
        field for field in _REQUIRED_MANIFEST_FIELDS if field.endswith("_sha256")
    ]
    for field in hash_fields:
        _require_sha256(manifest, field, "state manifest")

    shared_model_hash = str(manifest["shared_branch_model_sha256"]).lower()
    if (
        str(manifest["branch_a_model_sha256"]).lower() != shared_model_hash
        or str(manifest["branch_b_model_sha256"]).lower() != shared_model_hash
    ):
        raise ValueError("the two branches do not reference one shared model hash")
    if manifest["branch_a_parameter_sha256"] != manifest["branch_b_parameter_sha256"]:
        raise ValueError("the two branches do not share one parameter hash")
    if manifest["branch_a_mask_sha256"] != manifest["branch_b_mask_sha256"]:
        raise ValueError("the two branches do not share one mask hash")

    if _positive_integer(manifest["state_width"], "state_width") != 50:
        raise ValueError("formal twin state width must be 50")
    if _positive_integer(manifest["window_size"], "window_size") != 12:
        raise ValueError("formal CPI input window size must be 12")
    repeats = _positive_integer(manifest["repeat_count"], "repeat_count")
    capture = _positive_integer(manifest["capture_cycle"], "capture_cycle")
    if capture > repeats:
        raise ValueError("capture_cycle cannot exceed repeat_count")
    _positive_float(manifest["theta_seconds"], "theta_seconds")
    _positive_float(manifest["feedback_delay_seconds"], "feedback_delay_seconds")
    _positive_float(
        manifest["input_transport_delay_seconds"],
        "input_transport_delay_seconds",
        allow_zero=True,
    )
    _positive_float(manifest["fixed_step_seconds"], "fixed_step_seconds")
    _positive_integer(manifest["num_records"], "num_records")
    if not str(manifest["solver"]).strip():
        raise ValueError("state manifest solver must not be empty")
    if not str(manifest["matlab_release"]).strip() or not str(
        manifest["simulink_version"]
    ).strip():
        raise ValueError("MATLAB and Simulink versions must be recorded")


def _validate_file_hashes(
    manifest: Mapping[str, object],
    state_file: Path,
    audit_mat_file: Path,
    audit_json_file: Path,
) -> None:
    expected_files = {
        "state_file": (state_file, "state_file_sha256"),
        "audit_mat_file": (audit_mat_file, "audit_mat_sha256"),
        "audit_json_file": (audit_json_file, "audit_json_sha256"),
    }
    for field, (path, hash_field) in expected_files.items():
        if Path(str(manifest[field])).name != path.name:
            raise ValueError(f"state manifest {field} does not name {path.name}")
        if not path.is_file():
            raise FileNotFoundError(f"missing contract artifact: {path}")
        expected_hash = str(manifest[hash_field]).lower()
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {path.name}")


def _audit_vector(payload: Mapping[str, object], field: str, width: int) -> np.ndarray:
    vector = np.asarray(payload[field], dtype=np.float64).reshape(-1)
    if len(vector) != width or not np.isfinite(vector).all():
        raise ValueError(f"audit field {field} must contain {width} finite states")
    return vector


def _relative_max_error(left: np.ndarray, right: np.ndarray, atol: float) -> float:
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), atol)
    return float(np.max(np.abs(left - right) / denominator))


def _validate_audit(
    audit_payload: Mapping[str, object],
    audit_json: Mapping[str, object],
    manifest: Mapping[str, object],
    audit_mat_file: Path,
    width: int,
) -> dict[str, dict[str, float | bool]]:
    _require_fields(audit_payload, _REQUIRED_AUDIT_FIELDS, "twin audit MAT")
    _require_fields(audit_json, _REQUIRED_AUDIT_JSON_FIELDS, "twin audit JSON")
    for label, payload in (("twin audit MAT", audit_payload), ("twin audit JSON", audit_json)):
        if _scalar_string(payload["schema_version"], f"{label}.schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported {label} schema_version")
        if _scalar_string(payload["state_protocol"], f"{label}.state_protocol") != STATE_PROTOCOL:
            raise ValueError(f"{label} uses the wrong state protocol")
        for field in (
            "simulation_protocol_sha256",
            "shared_branch_model_sha256",
            "twin_model_sha256",
        ):
            actual = _scalar_string(payload[field], f"{label}.{field}").lower()
            if actual != str(manifest[field]).lower():
                raise ValueError(f"{label}.{field} disagrees with the state manifest")
        linked_hashes = {
            "reservoir_parameter_sha256": "branch_a_parameter_sha256",
            "mask_sha256": "branch_a_mask_sha256",
            "input_transform_sha256": "input_transform_sha256",
        }
        for audit_field, manifest_field in linked_hashes.items():
            actual = _scalar_string(
                payload[audit_field], f"{label}.{audit_field}"
            ).lower()
            if actual != str(manifest[manifest_field]).lower():
                raise ValueError(
                    f"{label}.{audit_field} disagrees with the state manifest"
                )

    if _require_sha256(audit_json, "audit_mat_sha256", "twin audit JSON") != sha256_file(
        audit_mat_file
    ):
        raise ValueError("twin audit JSON does not hash the supplied audit MAT")
    if audit_json["audit_passed"] is not True:
        raise ValueError("twin audit JSON does not declare a passing audit")
    audit_state_hash = _require_sha256(
        audit_json, "state_file_sha256", "twin audit JSON"
    )
    if manifest["split"] == "train" and audit_state_hash != str(
        manifest["state_file_sha256"]
    ).lower():
        raise ValueError("twin audit was not computed from this train state cache")
    atol = _positive_float(audit_json["audit_atol"], "audit_atol")
    rtol = _positive_float(audit_json["audit_rtol"], "audit_rtol")

    vectors = {
        field: _audit_vector(audit_payload, field, width)
        for field in _REQUIRED_AUDIT_FIELDS
        if field.startswith("h_")
    }
    comparisons = {
        "a_swap": (vectors["h_a_ab_branch_a"], vectors["h_a_ba_branch_b"]),
        "b_swap": (vectors["h_b_ab_branch_b"], vectors["h_b_ba_branch_a"]),
        "aa_same_input": (
            vectors["h_a_aa_branch_a"],
            vectors["h_a_aa_branch_b"],
        ),
        "a_partner_independence_branch_a": (
            vectors["h_a_ab_branch_a"],
            vectors["h_a_aa_branch_a"],
        ),
        "a_partner_independence_branch_b": (
            vectors["h_a_ba_branch_b"],
            vectors["h_a_aa_branch_b"],
        ),
        "repeat_branch_a": (
            vectors["h_a_aa_branch_a"],
            vectors["h_a_repeat_branch_a"],
        ),
        "repeat_branch_b": (
            vectors["h_a_aa_branch_b"],
            vectors["h_a_repeat_branch_b"],
        ),
        "cache_a": (vectors["h_a_cache"], vectors["h_a_ab_branch_a"]),
        "cache_b": (vectors["h_b_cache"], vectors["h_b_ab_branch_b"]),
    }
    metrics: dict[str, dict[str, float | bool]] = {}
    failures: list[str] = []
    for name, (left, right) in comparisons.items():
        max_abs = float(np.max(np.abs(left - right)))
        max_rel = _relative_max_error(left, right, atol)
        passed = bool(np.allclose(left, right, atol=atol, rtol=rtol))
        metrics[name] = {
            "max_abs": max_abs,
            "max_rel": max_rel,
            "passed": passed,
        }
        if not passed:
            failures.append(name)
    if failures:
        raise ValueError(
            "explicit twin numerical equivalence audit failed: " + ", ".join(failures)
        )
    return metrics


def _validate_generation_provenance(
    sample_ids: np.ndarray,
    partner_ids: np.ndarray,
    branch_ids: np.ndarray,
    run_ids: np.ndarray,
) -> None:
    allowed_branch_sets = ({"a", "b"}, {"target", "reference"})
    normalized_branches = np.asarray(
        [str(value).strip().lower() for value in branch_ids], dtype=object
    )
    if not set(normalized_branches).issubset(set.union(*allowed_branch_sets)):
        raise ValueError("branch_id must use A/B or target/reference")
    if set(normalized_branches).intersection({"a", "b"}) and set(
        normalized_branches
    ).intersection({"target", "reference"}):
        raise ValueError("branch_id mixes two naming conventions")

    known_ids = set(sample_ids.tolist())
    if not set(partner_ids.tolist()).issubset(known_ids):
        raise ValueError("partner_sample_id references a window outside this cache")
    for run_id in pd.unique(run_ids):
        rows = np.flatnonzero(run_ids == run_id)
        if len(rows) not in {1, 2}:
            raise ValueError("each generation_run_id must produce one or two cache records")
        if len(rows) == 1:
            row = int(rows[0])
            if partner_ids[row] != sample_ids[row]:
                raise ValueError("a singleton generation run must be an explicit self-pair")
            continue
        if len(set(normalized_branches[rows].tolist())) != 2:
            raise ValueError("a two-record generation run must use both twin branches")
        left, right = map(int, rows)
        if (
            partner_ids[left] != sample_ids[right]
            or partner_ids[right] != sample_ids[left]
        ):
            raise ValueError("generation partners are not reciprocal within their run")


def load_twin_state_cache(
    state_dir: Path,
    split: str,
    *,
    audit_dir: Path | None = None,
    expected_sample_ids: Sequence[int] | np.ndarray | None = None,
    expected_target_dates: Sequence[str] | np.ndarray | None = None,
    expected_input_windows_raw: object | None = None,
) -> TwinStateCache:
    """Load and fully validate one explicit-twin unique-window state cache.

    The returned lookup is safe for both the single-reservoir absolute readout
    and the Siamese shared readout.  Both models must use this same lookup for
    a fair comparison.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split: {split}")
    state_dir = Path(state_dir)
    audit_dir = state_dir if audit_dir is None else Path(audit_dir)
    state_file = state_dir / f"state_cache_{split}.mat"
    manifest_file = state_dir / f"state_cache_{split}.manifest.json"
    audit_mat_file = audit_dir / "audit_twin_equivalence.mat"
    audit_json_file = audit_dir / "audit_twin_equivalence.json"
    if not state_file.is_file():
        raise FileNotFoundError(f"missing explicit twin state cache: {state_file}")

    manifest = _load_json(manifest_file, "state manifest")
    _validate_manifest_shape(manifest, split)
    _validate_file_hashes(manifest, state_file, audit_mat_file, audit_json_file)

    state_payload = loadmat(state_file, squeeze_me=True, struct_as_record=False)
    forbidden = _forbidden_label_fields(state_payload)
    if forbidden:
        raise ValueError(
            "explicit twin state cache contains forbidden target/label fields: "
            + ", ".join(forbidden)
        )
    _require_fields(state_payload, _REQUIRED_STATE_FIELDS, "twin state MAT")
    if _scalar_string(state_payload["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported twin state MAT schema_version")
    if _scalar_string(state_payload["state_protocol"], "state_protocol") != STATE_PROTOCOL:
        raise ValueError("twin state MAT uses the wrong state protocol")
    if _scalar_string(state_payload["split"], "split") != split:
        raise ValueError("twin state MAT split does not match the requested split")
    for field in (
        "input_transform_sha256",
        "simulation_protocol_sha256",
        "shared_branch_model_sha256",
        "twin_model_sha256",
    ):
        value = _scalar_string(state_payload[field], field).lower()
        if value != str(manifest[field]).lower():
            raise ValueError(f"twin state MAT {field} disagrees with its manifest")
    if _scalar_string(state_payload["state_mode"], "state_mode") != manifest["state_mode"]:
        raise ValueError("twin state MAT state_mode disagrees with its manifest")
    if _positive_integer(state_payload["repeat_count"], "repeat_count") != int(
        manifest["repeat_count"]
    ):
        raise ValueError("twin state MAT repeat_count disagrees with its manifest")
    if _positive_integer(state_payload["capture_cycle"], "capture_cycle") != int(
        manifest["capture_cycle"]
    ):
        raise ValueError("twin state MAT capture_cycle disagrees with its manifest")
    sample_times = np.asarray(
        state_payload["sample_times_seconds"], dtype=np.float64
    ).reshape(-1)
    if (
        len(sample_times) != int(manifest["state_width"])
        or not np.isfinite(sample_times).all()
        or not np.all(np.diff(sample_times) > 0)
    ):
        raise ValueError("sample_times_seconds must contain 50 increasing finite times")
    expected_sample_times = float(manifest["input_transport_delay_seconds"]) + (
        (int(manifest["capture_cycle"]) - 1) * int(manifest["state_width"])
        + np.arange(1, int(manifest["state_width"]) + 1)
    ) * float(manifest["theta_seconds"])
    if not np.allclose(
        sample_times,
        expected_sample_times,
        rtol=0.0,
        atol=max(1e-18, 32 * np.finfo(np.float64).eps * sample_times[-1]),
    ):
        raise ValueError(
            "sample_times_seconds do not match node-end capture-cycle timing"
        )

    states_raw = np.asarray(state_payload["state_matrix"], dtype=np.float64)
    if states_raw.ndim == 1:
        states_raw = states_raw.reshape(1, -1)
    if states_raw.ndim != 2:
        raise ValueError("state_matrix must be a 2D matrix")
    n_records, state_width = states_raw.shape
    if n_records != int(manifest["num_records"]):
        raise ValueError("state_matrix row count disagrees with its manifest")
    states = _matrix(states_raw, "state_matrix", n_records, int(manifest["state_width"]))
    if state_width != 50:
        raise ValueError("formal twin state_matrix must have 50 columns")

    sample_ids = _integer_vector(state_payload["sample_id"], "sample_id", n_records)
    if len(np.unique(sample_ids)) != n_records:
        raise ValueError("sample_id values must be unique")
    if not np.array_equal(sample_ids, np.sort(sample_ids, kind="stable")):
        raise ValueError("sample_id values must be chronological/increasing")
    target_dates = _string_vector(state_payload["target_date"], "target_date", n_records)
    if any(not _MONTH_RE.fullmatch(str(value)) for value in target_dates):
        raise ValueError("target_date values must use YYYY-MM format")
    if len(set(target_dates.tolist())) != n_records:
        raise ValueError("target_date values must be unique")
    raw_windows = _matrix(
        state_payload["input_window_raw"],
        "input_window_raw",
        n_records,
        int(manifest["window_size"]),
    )
    _matrix(
        state_payload["input_window_scaled"],
        "input_window_scaled",
        n_records,
        int(manifest["window_size"]),
    )
    cache_record_ids = _identifier_vector(
        state_payload["cache_record_id"], "cache_record_id", n_records
    )
    if len(set(cache_record_ids.tolist())) != n_records:
        raise ValueError("cache_record_id values must be unique")
    branch_ids = _string_vector(state_payload["branch_id"], "branch_id", n_records)
    partner_ids = _integer_vector(
        state_payload["partner_sample_id"], "partner_sample_id", n_records
    )
    run_ids = _identifier_vector(
        state_payload["generation_run_id"], "generation_run_id", n_records
    )
    _validate_generation_provenance(sample_ids, partner_ids, branch_ids, run_ids)

    if expected_sample_ids is not None:
        expected = _integer_vector(expected_sample_ids, "expected_sample_ids")
        if not np.array_equal(sample_ids, expected):
            raise ValueError("state cache sample IDs disagree with the isolated split")
    if expected_target_dates is not None:
        expected_dates = np.asarray(expected_target_dates, dtype=str).reshape(-1)
        if not np.array_equal(target_dates.astype(str), expected_dates):
            raise ValueError("state cache target dates disagree with the isolated split")
    if expected_input_windows_raw is not None:
        expected_windows = np.asarray(expected_input_windows_raw, dtype=np.float64)
        if expected_windows.shape != raw_windows.shape or not np.allclose(
            raw_windows, expected_windows, atol=0.0, rtol=0.0
        ):
            raise ValueError("state cache input windows disagree with the isolated split")

    audit_payload = loadmat(audit_mat_file, squeeze_me=True, struct_as_record=False)
    audit_json = _load_json(audit_json_file, "twin audit JSON")
    audit_metrics = _validate_audit(
        audit_payload,
        audit_json,
        manifest,
        audit_mat_file,
        state_width,
    )

    lookup: dict[int, np.ndarray] = {}
    for sample_id, state in zip(sample_ids, states):
        frozen_state = np.asarray(state, dtype=np.float64).copy()
        frozen_state.setflags(write=False)
        lookup[int(sample_id)] = frozen_state
    provenance = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "target_date": target_dates.astype(str),
            "cache_record_id": cache_record_ids.astype(str),
            "branch_id": branch_ids.astype(str),
            "partner_sample_id": partner_ids,
            "generation_run_id": run_ids.astype(str),
            "state_file_sha256": str(manifest["state_file_sha256"]),
            "state_resolution": "unique_window_cache_lookup",
            "cache_reuse_declared": True,
            "semantic_pairs_simulated_simultaneously": False,
        }
    )
    return TwinStateCache(
        split=split,
        state_lookup=MappingProxyType(lookup),
        provenance=provenance,
        manifest=MappingProxyType(dict(manifest)),
        audit_metrics=MappingProxyType(
            {name: MappingProxyType(values) for name, values in audit_metrics.items()}
        ),
        state_file=state_file.resolve(),
        state_file_sha256=str(manifest["state_file_sha256"]),
        audit_mat_file=audit_mat_file.resolve(),
        audit_json_file=audit_json_file.resolve(),
    )


def load_twin_state_caches(
    state_dir: Path,
    splits: Sequence[str],
    *,
    audit_dir: Path | None = None,
) -> tuple[dict[int, np.ndarray], pd.DataFrame, dict[str, TwinStateCache]]:
    """Load several split caches and reject any cross-split protocol drift."""
    if not splits:
        raise ValueError("at least one split is required")
    caches = {
        split: load_twin_state_cache(state_dir, split, audit_dir=audit_dir)
        for split in splits
    }
    first = caches[splits[0]].manifest
    for split in splits[1:]:
        current = caches[split].manifest
        changed = [field for field in _PINNED_MANIFEST_FIELDS if current[field] != first[field]]
        if changed:
            raise ValueError(
                f"twin state protocol changed between splits at: {sorted(changed)}"
            )
    lookup: dict[int, np.ndarray] = {}
    provenance_parts: list[pd.DataFrame] = []
    for split in splits:
        cache = caches[split]
        overlap = set(lookup).intersection(cache.state_lookup)
        if overlap:
            raise ValueError(f"duplicate sample IDs across twin caches: {sorted(overlap)}")
        lookup.update(cache.state_lookup)
        part = cache.provenance.copy()
        part.insert(0, "split", split)
        provenance_parts.append(part)
    return lookup, pd.concat(provenance_parts, ignore_index=True), caches


def _protocol_identity(manifest: Mapping[str, object]) -> str:
    """Hash every field that defines the fixed state-generation protocol."""
    identity = {field: manifest[field] for field in _PINNED_MANIFEST_FIELDS}
    serialized = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_twin_state_splits(
    state_dir: Path,
    split_payloads: Mapping[str, Mapping[str, object]],
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    """Drop-in strict replacement for the legacy pipeline state loader.

    ``split_payloads`` use the same ``{"X", "index", ...}`` shape as the
    physically isolated split loaders.  Numeric CPI targets are intentionally
    neither requested nor read here.  By default the global equivalence audit
    is taken from the sibling ``audits_twin`` directory used by the MATLAB
    runner.
    """
    if not split_payloads:
        raise ValueError("at least one split payload is required")
    state_dir = Path(state_dir)
    sibling_audit_dir = state_dir.parent / "audits_twin"
    audit_dir = sibling_audit_dir if sibling_audit_dir.is_dir() else state_dir

    lookup: dict[int, np.ndarray] = {}
    caches: dict[str, TwinStateCache] = {}
    split_audits: dict[str, object] = {}
    cache_records: list[dict[str, object]] = []
    first_manifest: Mapping[str, object] | None = None
    first_identity: str | None = None
    for split, payload in split_payloads.items():
        if split not in VALID_SPLITS:
            raise ValueError(f"unknown split: {split}")
        if "index" not in payload or "X" not in payload:
            raise ValueError(f"{split} payload must contain index and X")
        index = payload["index"]
        try:
            expected_ids = index["sample_id"].to_numpy(dtype=np.int64)
            expected_dates = index["target_date"].astype(str).to_numpy()
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"{split} payload index is missing ID/date columns") from exc
        expected_windows = np.asarray(payload["X"], dtype=np.float64)
        cache = load_twin_state_cache(
            state_dir,
            split,
            audit_dir=audit_dir,
            expected_sample_ids=expected_ids,
            expected_target_dates=expected_dates,
            expected_input_windows_raw=expected_windows,
        )
        caches[split] = cache
        identity = _protocol_identity(cache.manifest)
        if first_manifest is None:
            first_manifest = cache.manifest
            first_identity = identity
        else:
            changed = [
                field
                for field in _PINNED_MANIFEST_FIELDS
                if cache.manifest[field] != first_manifest[field]
            ]
            if changed or identity != first_identity:
                raise ValueError(
                    "twin state protocol changed between splits at: "
                    + ", ".join(sorted(changed))
                )
        overlap = set(lookup).intersection(cache.state_lookup)
        if overlap:
            raise ValueError(f"duplicate sample IDs across twin caches: {sorted(overlap)}")
        lookup.update(cache.state_lookup)
        split_audits[split] = {
            "state_file": str(cache.state_file),
            "sha256": cache.state_file_sha256,
            "shape": [len(cache.state_lookup), cache.state_width],
            "first_target_date": str(cache.provenance["target_date"].iloc[0]),
            "last_target_date": str(cache.provenance["target_date"].iloc[-1]),
            "cache_reuse_declared": True,
            "semantic_pairs_simulated_simultaneously": False,
        }
        for row in cache.provenance.itertuples(index=False):
            cache_records.append(
                {
                    "split": split,
                    "sample_id": int(row.sample_id),
                    "cache_record_id": str(row.cache_record_id),
                    "branch_id": str(row.branch_id),
                    "partner_sample_id": int(row.partner_sample_id),
                    "generation_run_id": str(row.generation_run_id),
                    "state_file_sha256": str(row.state_file_sha256),
                }
            )

    assert first_manifest is not None and first_identity is not None
    first_cache = caches[next(iter(split_payloads))]
    return lookup, {
        "status": "passed",
        "state_protocol": STATE_PROTOCOL,
        "protocol_identity_sha256": first_identity,
        "state_width": int(first_manifest["state_width"]),
        "loaded_splits": list(split_payloads),
        "cache_reuse_declared": True,
        "cache_generated_by_explicit_twin_model": True,
        "semantic_pairs_simulated_simultaneously": False,
        "pair_states_resolved_by_sample_id": True,
        "only_output_weights_trainable_downstream": True,
        "shared_branch_model_sha256": first_manifest["shared_branch_model_sha256"],
        "twin_model_sha256": first_manifest["twin_model_sha256"],
        "simulation_protocol_sha256": first_manifest["simulation_protocol_sha256"],
        "audit_mat_sha256": first_manifest["audit_mat_sha256"],
        "audit_json_sha256": first_manifest["audit_json_sha256"],
        "cache_records": cache_records,
        "twin_equivalence_audit": {
            name: dict(values) for name, values in first_cache.audit_metrics.items()
        },
        "splits": split_audits,
    }


__all__ = [
    "SCHEMA_VERSION",
    "STATE_PROTOCOL",
    "TwinStateCache",
    "load_twin_state_cache",
    "load_twin_state_caches",
    "load_twin_state_splits",
    "sha256_file",
]
