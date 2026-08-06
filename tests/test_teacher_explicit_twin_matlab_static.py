from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATLAB_DIR = ROOT / "matlab" / "optical_reservoir_cpi"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing contract source: {path}"
    return path.read_text(encoding="utf-8")


def _quoted_matlab_arguments(fragment: str) -> set[str]:
    return set(re.findall(r"'([^']+)'", fragment))


def test_twin_config_declares_formal_isolated_four_cycle_protocol() -> None:
    source = _read(MATLAB_DIR / "config_twin_cpi_rc.m")

    assert "config.sequence_protocol = 'isolated_repeated_window';" in source
    assert re.search(r"config\.repeat_count\s*=\s*4\s*;", source)
    assert re.search(r"config\.capture_cycle\s*=\s*4\s*;", source)
    assert (
        "explicit_twin_audited_unique_window_cache_v1" in source
    ), "the MATLAB config must pin the formal Python state protocol"
    assert "config.reset_between_runs = true;" in source


def test_shared_branch_is_explicitly_multi_instance() -> None:
    source = _read(MATLAB_DIR / "build_shared_reservoir_branch_model.m")

    assert "ModelReferenceNumInstancesAllowed" in source
    assert re.search(
        r"set_param\(\s*model_name\s*,\s*"
        r"'ModelReferenceNumInstancesAllowed'\s*,\s*'Multi'\s*\)",
        source,
        flags=re.DOTALL,
    )
    assert "'LoadInitialState', 'off'" in source
    assert "'SaveFinalState', 'off'" in source


def test_twin_topology_has_two_references_to_one_branch_and_separate_io() -> None:
    source = _read(MATLAB_DIR / "build_twin_shared_reservoir_model.m")

    assert source.count(
        "add_block('simulink/Ports & Subsystems/Model'"
    ) == 2
    assert source.count("'ModelName', branch_name") == 2
    assert source.count("add_block('simulink/Sources/From Workspace'") == 2
    assert source.count("add_block('simulink/Sinks/To Workspace'") == 2

    for variable in (
        "config.target_input_variable",
        "config.reference_input_variable",
        "config.target_output_variable",
        "config.reference_output_variable",
    ):
        assert variable in source

    required_wiring = (
        "target_source, 'PortHandles'",
        "target_branch, 'PortHandles'",
        "target_logger, 'PortHandles'",
        "reference_source, 'PortHandles'",
        "reference_branch, 'PortHandles'",
        "reference_logger, 'PortHandles'",
    )
    for fragment in required_wiring:
        assert fragment in source

    assert re.search(
        r"find_system\(\s*twin_name\s*,.*?'BlockType'\s*,\s*"
        r"'ModelReference'\s*\)",
        source,
        flags=re.DOTALL,
    )
    assert re.search(r"numel\(model_blocks\)\s*==\s*2", source)
    assert re.search(
        r"all\(strcmp\(get_param\(model_blocks,\s*'ModelName'\),\s*"
        r"branch_name\)\)",
        source,
        flags=re.DOTALL,
    )


def test_prepare_loads_named_input_fields_and_persists_no_labels() -> None:
    source = _read(MATLAB_DIR / "prepare_twin_window_cache.m")
    named_load = re.search(
        r"source\s*=\s*load\(\s*source_file\s*,(?P<fields>.*?)\);",
        source,
        flags=re.DOTALL,
    )
    assert named_load, "isolated data must be loaded with an explicit field allowlist"
    loaded_fields = _quoted_matlab_arguments(named_load.group("fields"))
    assert {
        "X",
        "sample_id",
        "x_start_date",
        "x_end_date",
        "target_date",
        "window_size",
    }.issubset(loaded_fields)
    assert loaded_fields.isdisjoint(
        {"y", "y_scaled", "target", "target_scaled", "cpi_actual"}
    )

    assert "whos('-file', input_file)" in source
    assert re.search(
        r"forbidden\s*=\s*\{.*?'y'.*?'y_scaled'.*?'target'.*?"
        r"'target_scaled'.*?'cpi_actual'.*?\}\s*;",
        source,
        flags=re.DOTALL,
    )
    assert re.search(
        r"assert\(\s*~any\(ismember\(persisted_names,\s*forbidden\)\)",
        source,
        flags=re.DOTALL,
    )


def test_test_state_generation_is_locked_behind_python_freeze_authorization() -> None:
    matlab_source = _read(MATLAB_DIR / "run_twin_state_cache.m")
    python_source = _read(
        ROOT / "scripts" / "run_teacher_explicit_twin_mom_closed50.py"
    )

    test_guard = matlab_source.index("if strcmp(split, 'test')")
    prepare_call = matlab_source.index("prepare_twin_window_cache(split, config)")
    assert test_guard < prepare_call
    assert "test_generation_authorization.json" in matlab_source
    assert "assert(isfile(authorization_file)" in matlab_source
    assert (
        "validation_frozen_authorized_for_test_state_generation"
        in matlab_source
    )
    assert "protocol_identity_sha256" in matlab_source

    assert "validation_frozen_not_tested" in python_source
    assert "write_test_generation_authorization" in python_source
    validation_call = python_source.index("run_validation_search(")
    authorization_call = python_source.index(
        "write_test_generation_authorization(", validation_call
    )
    assert validation_call < authorization_call


def test_matlab_state_and_manifest_names_match_python_strict_contract() -> None:
    matlab_source = _read(MATLAB_DIR / "run_twin_state_cache.m")
    python_source = _read(ROOT / "src" / "twin_state_cache_contract.py")

    assert "sprintf('state_cache_%s.mat', split)" in matlab_source
    assert "sprintf('state_cache_%s.manifest.json', split)" in matlab_source
    assert 'f"state_cache_{split}.mat"' in python_source
    assert 'f"state_cache_{split}.manifest.json"' in python_source


def test_matlab_training_demo_only_solves_the_shared_output_layer() -> None:
    source = _read(MATLAB_DIR / "fit_twin_shared_output_weights.m")

    assert "state_cache_train.mat" in source
    assert "cpi_train_isolated.mat" in source
    assert "solution = design \\ response;" in source
    assert "model.Wout = solution(2:end);" in source
    assert "model.only_output_weights_trained = true;" in source
    assert "model.reservoir_parameters_trained = false;" in source
    assert "numel(pair_i) == 741" in source


def test_protocol_document_says_matlab_has_not_run_and_no_metrics_exist() -> None:
    final_protocol = _read(
        ROOT / "docs" / "teacher_explicit_twin_matlab_protocol.md"
    )
    legacy_protocol = _read(
        ROOT / "docs" / "teacher_shared_optical_reservoir_protocol.md"
    )

    assert "协议状态：待 MATLAB 实际执行" in final_protocol
    assert "当前电脑没有安装 MATLAB/Simulink" in final_protocol
    assert "尚未生成正式的显式双分支状态" in final_protocol
    assert "当前尚不存在本文方案的正式验证指标和测试指标" in final_protocol
    assert "LEGACY（已降级，不是老师最新最终方案）" in legacy_protocol
    assert "旧连续状态指标不是老师最终方案结果" in legacy_protocol or (
        "连续缓存状态和由此得到的指标，均不能再称为"
        in legacy_protocol
    )
