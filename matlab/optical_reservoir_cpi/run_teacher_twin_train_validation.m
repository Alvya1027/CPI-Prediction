function outputs = run_teacher_twin_train_validation(config)
%RUN_TEACHER_TWIN_TRAIN_VALIDATION Build formal train/validation caches.
%
% Run this entry point before any model-selection work in Python.  It does
% not read or generate the test split.  The immutable global branch audit is
% generated from the first two training windows and validation reuses it.

if nargin < 1 || isempty(config)
    config = config_twin_cpi_rc();
end
build_shared_reservoir_branch_model(config, false);
build_twin_shared_reservoir_model(config, false);
prepare_twin_window_cache('train', config);
prepare_twin_window_cache('val', config);

outputs = struct();
outputs.train = run_twin_state_cache('train', config, true);
outputs.val = run_twin_state_cache('val', config, false);
fprintf(['Teacher-final explicit twin train/validation states are ready. ', ...
    'Freeze Python validation configuration before authorizing test.\n']);
end
