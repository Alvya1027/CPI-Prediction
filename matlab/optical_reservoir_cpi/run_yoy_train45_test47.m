function outputs = run_yoy_train45_test47(config)
%RUN_YOY_TRAIN45_TEST47 Run the single reservoir on YoY train45/test47.
%
% No validation input or state is created.  The reservoir model is copied to
% an independent dated YoY profile so the earlier MoM and legacy YoY outputs
% remain untouched.

if nargin < 1 || isempty(config)
    base = config_cpi_rc();
    profile_root = fullfile(base.project_dir, '..', ...
        'optical_reservoir_cpi_yoy_train45_noval_20260807');
    config = base;
    config.data_file = fullfile(profile_root, 'data', 'cpi_windows.mat');
    config.input_dir = fullfile(profile_root, 'inputs');
    config.response_dir = fullfile(profile_root, 'responses');
    config.state_dir = fullfile(profile_root, 'states');
    config.model_file = fullfile(profile_root, 'SL_RC_yoy_train45_noval.slx');
    if ~exist(profile_root, 'dir')
        mkdir(profile_root);
    end
    if ~isfile(config.model_file)
        copyfile(base.model_file, config.model_file);
    end
end

assert(isfile(config.data_file), ...
    'Missing YoY data. Run prepare_optical_reservoir_yoy_train45_noval.py first.');

outputs = struct();
outputs.input_summary = prepare_yoy_cpi_inputs(config);
for split = {'train', 'test'}
    split_name = split{1};
    fprintf('\n=== Running YoY %s split ===\n', split_name);
    outputs.(split_name).response_file = run_cpi_simulation(split_name, config);
    outputs.(split_name).state_file = extract_cpi_states(split_name, config);
end
save(fullfile(fileparts(config.data_file), '..', 'run_summary.mat'), ...
    'outputs', 'config', '-v7');
disp('Single YoY train45/test47 reservoir states are ready.');
end
