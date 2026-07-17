function summary = prepare_cpi_inputs()
%PREPARE_CPI_INPUTS Build fixed-mask Simulink inputs for all CPI splits.

config = config_cpi_rc();
data = load(config.data_file);

assert(double(data.window_size) == config.window_size, ...
    'The exported CPI window size does not match config_cpi_rc.m.');

rng(config.random_seed, 'twister');
mask = 2 * randi([0, 1], config.window_size, config.num_virtual_nodes) - 1;

train_projection = data.X_train_scaled * mask / sqrt(config.window_size);
train_peak = max(abs(train_projection(:)));
assert(train_peak > 0, 'The training projection is all zeros.');

% Fit this scale on training data only, then reuse it for validation and test.
mask_scale = config.target_masked_amplitude / train_peak;

if ~exist(config.input_dir, 'dir')
    mkdir(config.input_dir);
end

split_names = {'train', 'val', 'test'};
summary = struct();

for split_index = 1:numel(split_names)
    split = split_names{split_index};
    x_scaled = data.(sprintf('X_%s_scaled', split));
    target = data.(sprintf('y_%s', split));
    target_scaled = data.(sprintf('y_%s_scaled', split));
    sample_id = data.(sprintf('sample_id_%s', split));
    x_start_date = data.(sprintf('x_start_date_%s', split));
    x_end_date = data.(sprintf('x_end_date_%s', split));
    target_date = data.(sprintf('target_date_%s', split));

    assert(size(x_scaled, 2) == config.window_size, ...
        'Split %s has the wrong window size.', split);

    masked_input = (x_scaled * mask / sqrt(config.window_size)) * mask_scale;

    % Transpose before reshape so each sample contributes 50 consecutive nodes.
    serialized_input = reshape(masked_input.', [], 1);
    drive_signal = serialized_input * config.input_gain;
    time_seconds = (0:numel(drive_signal) - 1).' * config.theta_seconds;
    simin = [time_seconds, drive_signal];
    simulation_stop_time = config.warmup_seconds + ...
        numel(drive_signal) * config.theta_seconds;

    output_file = fullfile(config.input_dir, sprintf('simin_%s.mat', split));
    save(output_file, 'simin', 'masked_input', 'mask', 'mask_scale', ...
        'target', 'target_scaled', 'sample_id', 'x_start_date', ...
        'x_end_date', 'target_date', 'simulation_stop_time', 'config', '-v7');

    summary.(split).num_samples = size(x_scaled, 1);
    summary.(split).num_time_points = numel(drive_signal);
    summary.(split).masked_min = min(masked_input(:));
    summary.(split).masked_max = max(masked_input(:));
    summary.(split).simulation_stop_time = simulation_stop_time;
    summary.(split).output_file = output_file;
end

save(fullfile(config.input_dir, 'mask_parameters.mat'), ...
    'mask', 'mask_scale', 'config', 'summary', '-v7');

disp(summary);
end
