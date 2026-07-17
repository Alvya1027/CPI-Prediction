function config = config_cpi_rc()
%CONFIG_CPI_RC Shared parameters for the CPI optical reservoir simulation.

project_dir = fileparts(mfilename('fullpath'));

config.project_dir = project_dir;
config.data_file = fullfile(project_dir, 'data', 'cpi_windows.mat');
config.model_file = fullfile(project_dir, 'SL_RC.slx');
config.input_dir = fullfile(project_dir, 'inputs');
config.response_dir = fullfile(project_dir, 'responses');
config.state_dir = fullfile(project_dir, 'states');

config.window_size = 12;
config.num_virtual_nodes = 50;
config.theta_seconds = 4e-11;
config.feedback_delay_seconds = 2.04e-9;
config.warmup_seconds = 4e-6;
config.input_gain = 0.004;
config.target_masked_amplitude = 0.5;
config.random_seed = 42;
end
