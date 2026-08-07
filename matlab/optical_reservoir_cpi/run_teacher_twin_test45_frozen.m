function outputs = run_teacher_twin_test45_frozen(config)
%RUN_TEACHER_TWIN_TEST45_FROZEN Generate the unchanged 47 test states.

if nargin < 1 || isempty(config)
    config = config_twin_cpi_rc_train45_noval();
end
outputs = run_twin_state_cache('test', config, false);
fprintf('Frozen train45/no-validation Twin test47 states are ready.\n');
end
