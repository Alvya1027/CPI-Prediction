function outputs = run_mom_recent50_all()
%RUN_MOM_RECENT50_ALL Run the isolated MoM optical-reservoir experiment.
%
% The Python preparation script must be run first.  All generated inputs,
% responses and states live under the dated MoM profile directory; the
% existing YoY ``data``, ``inputs``, ``responses`` and ``states`` folders are
% not used.

base = config_cpi_rc();
profile_root = fullfile(base.project_dir, '..', ...
    'optical_reservoir_cpi_mom_recent50_20260730');
config = base;
config.data_file = fullfile(profile_root, 'data', 'cpi_windows.mat');
config.input_dir = fullfile(profile_root, 'inputs');
config.response_dir = fullfile(profile_root, 'responses');
config.state_dir = fullfile(profile_root, 'states');
config.model_file = fullfile(profile_root, 'SL_RC_mom_recent50.slx');

assert(isfile(config.data_file), ...
    'Missing MoM data. Run scripts/prepare_optical_reservoir_mom_recent50.py first.');
if ~isfile(config.model_file)
    copyfile(base.model_file, config.model_file);
end

outputs = run_all_cpi_simulations(config);
save(fullfile(profile_root, 'run_summary.mat'), 'outputs', 'config', '-v7');
disp('Isolated MoM optical-reservoir states are ready.');
end
