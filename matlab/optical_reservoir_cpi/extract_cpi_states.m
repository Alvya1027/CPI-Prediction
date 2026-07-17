function state_file = extract_cpi_states(split)
%EXTRACT_CPI_STATES Convert a logged response into one 50-node state per sample.

if nargin < 1
    split = 'train';
end

valid_splits = {'train', 'val', 'test'};
assert(any(strcmp(split, valid_splits)), ...
    'split must be train, val, or test.');

config = config_cpi_rc();
input_file = fullfile(config.input_dir, sprintf('simin_%s.mat', split));
response_file = fullfile(config.response_dir, sprintf('response_%s.mat', split));
assert(exist(input_file, 'file') == 2, 'Missing input file: %s', input_file);
assert(exist(response_file, 'file') == 2, 'Missing response file: %s', response_file);

input_data = load(input_file);
response_data = load(response_file, 'ScopeData');
assert(isfield(response_data, 'ScopeData') && ~isempty(response_data.ScopeData), ...
    'ScopeData is empty in %s.', response_file);

[response_time, response_signal] = unpack_scope_data(response_data.ScopeData);
response_time = double(response_time(:));
response_signal = double(response_signal(:));
assert(numel(response_time) == numel(response_signal), ...
    'Scope time and signal lengths do not match.');
assert(isreal(response_signal), 'The logged reservoir response must be real-valued.');

valid = isfinite(response_time) & isfinite(response_signal);
response_time = response_time(valid);
response_signal = response_signal(valid);
[response_time, unique_indices] = unique(response_time, 'stable');
response_signal = response_signal(unique_indices);

num_samples = numel(input_data.sample_id);
num_points = num_samples * config.num_virtual_nodes;
sample_times = config.warmup_seconds + ...
    (0:num_points - 1).' * config.theta_seconds;
tolerance = max(config.theta_seconds, eps(max(abs(response_time))));
assert(min(response_time) <= sample_times(1) + tolerance, ...
    'The response starts after the first required state time.');
assert(max(response_time) >= sample_times(end) - tolerance, ...
    ['The response ends before all states are available. Check the model ', ...
     'StopTime and Scope logging limit.']);

serialized_state = interp1(response_time, response_signal, sample_times, 'linear');
assert(all(isfinite(serialized_state)), 'State interpolation produced invalid values.');

% Each input sample contributes num_virtual_nodes consecutive response values.
state_matrix = reshape(serialized_state, config.num_virtual_nodes, num_samples).';
sample_id = input_data.sample_id;
target = input_data.target;
target_scaled = input_data.target_scaled;
target_date = input_data.target_date;
mask = input_data.mask;
mask_scale = input_data.mask_scale;

if ~exist(config.state_dir, 'dir')
    mkdir(config.state_dir);
end
state_file = fullfile(config.state_dir, sprintf('states_%s.mat', split));
save(state_file, 'state_matrix', 'sample_id', 'target', 'target_scaled', ...
    'target_date', 'mask', 'mask_scale', 'config', 'input_file', ...
    'response_file', '-v7');

fprintf('Saved %d x %d %s states to %s\n', ...
    size(state_matrix, 1), size(state_matrix, 2), split, state_file);
end


function [time, signal] = unpack_scope_data(scope_data)
%UNPACK_SCOPE_DATA Support the common Simulink Scope save formats.

if isnumeric(scope_data)
    assert(size(scope_data, 2) >= 2, ...
        'Numeric ScopeData must contain time and signal columns.');
    time = scope_data(:, 1);
    signal = scope_data(:, 2);
elseif isa(scope_data, 'timeseries')
    time = scope_data.Time;
    values = scope_data.Data;
    signal = values(:, 1);
elseif isstruct(scope_data) && isfield(scope_data, 'time') && ...
        isfield(scope_data, 'signals')
    time = scope_data.time;
    values = scope_data.signals.values;
    signal = values(:, 1);
else
    error(['Unsupported ScopeData format. Configure the output Scope to ', ...
        'save an Array or Structure With Time.']);
end
end
