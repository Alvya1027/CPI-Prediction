function outputs = run_teacher_twin_train45_noval(config)
%RUN_TEACHER_TWIN_TRAIN45_NOVAL Generate only the 45 training states.
%
% Test generation stays locked until Python records the fixed train-only
% configuration and writes test_generation_authorization.json.

if nargin < 1 || isempty(config)
    config = config_twin_cpi_rc_train45_noval();
end
build_shared_reservoir_branch_model(config, false);
build_twin_shared_reservoir_model(config, false);
prepare_twin_window_cache('train', config);

outputs = struct();
outputs.train = run_twin_state_cache('train', config, true);
fprintf(['Teacher explicit Twin train45 states are ready. ', ...
    'Freeze the fixed train-only Python configuration before test.\n']);
end
