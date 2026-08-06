function config = config_twin_cpi_rc()
%CONFIG_TWIN_CPI_RC Configuration for the explicit shared twin reservoir.
%
% The two branches are Model Reference instances of the same generated
% SL_RC_shared_branch.slx file.  Reservoir parameters stay fixed; MATLAB only
% produces states and Python fits the shared linear output weights.

base = config_cpi_rc();
profile_root = fullfile(base.project_dir, '..', ...
    'optical_reservoir_cpi_mom_recent50_20260730');

config = base;
config.profile_root = profile_root;
config.isolated_data_dir = fullfile(profile_root, 'data');
config.input_dir = fullfile(profile_root, 'inputs_twin');
config.state_dir = fullfile(profile_root, 'states_twin');
config.audit_dir = fullfile(profile_root, 'audits_twin');
config.source_model_file = fullfile(profile_root, ...
    'SL_RC_mom_recent50.slx');
if ~isfile(config.source_model_file)
    config.source_model_file = base.model_file;
end
config.branch_model_file = fullfile(profile_root, ...
    'SL_RC_shared_branch.slx');
config.twin_model_file = fullfile(profile_root, 'Twin_SL_RC.slx');

config.target_input_variable = 'simin_target';
config.reference_input_variable = 'simin_reference';
config.target_output_variable = 'TwinTargetStateData';
config.reference_output_variable = 'TwinReferenceStateData';
config.logger_sample_time_seconds = config.theta_seconds;

% Teacher-final protocol: a target window and a reference window enter two
% independent instances of the exact same referenced optical reservoir at
% the same simulation time.  A 12-month window is masked into 50 virtual
% nodes, repeated four times, and only the fourth cycle is retained.  Four
% cycles ensure the retained cycle starts after more than two 2.04 ns
% feedback delays.  This is intentionally different from the legacy
% continuous-serial state cache.
config.repeat_count = 4;
config.capture_cycle = 4;
config.sequence_protocol = 'isolated_repeated_window';
config.sample_phase = 'node_end';
config.state_protocol = ...
    'explicit_twin_audited_unique_window_cache_v1';
config.schema_version = '1.0';
config.state_origin = ...
    'matlab_simulink_explicit_twin_model_reference_isolated_v1';
config.noise_seed = 1;
config.solver = 'ode4';
config.fixed_step_seconds = 1e-12;
config.audit_rtol = 1e-8;
config.audit_atol = 1e-10;
config.reset_between_runs = true;
config.cache_reuse_declared = true;
config.cache_generated_by_explicit_twin_model = true;
config.semantic_pairs_simulated_simultaneously = false;
config.pair_states_resolved_by_sample_id = true;
config.derived_pair_states_from_cache = true;
end
