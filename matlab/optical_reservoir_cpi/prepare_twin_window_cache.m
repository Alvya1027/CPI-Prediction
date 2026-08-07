function input_file = prepare_twin_window_cache(split, config)
%PREPARE_TWIN_WINDOW_CACHE Prepare label-free isolated-window twin inputs.
%
% Training fits the input scaler, fixed binary mask and mask amplitude.  The
% resulting transform is reused without refitting for validation and test.
% No CPI target/y value is loaded into or saved from this function.

if nargin < 1 || isempty(split)
    split = 'train';
end
if nargin < 2 || isempty(config)
    config = config_twin_cpi_rc();
end
split = char(lower(string(split)));
valid_splits = {'train', 'val', 'test'};
if isfield(config, 'valid_splits')
    valid_splits = cellstr(string(config.valid_splits));
end
assert(any(strcmp(split, valid_splits)), ...
    'split is not enabled for this experiment profile.');

repeat_count = twin_repeat_count(config);
assert(repeat_count == 4, ...
    'The formal isolated repeated-window protocol requires repeat_count=4.');
assert(config.num_virtual_nodes == 50, ...
    'The formal twin protocol expects exactly 50 virtual nodes.');
assert(config.window_size == 12, ...
    'The formal twin protocol expects 12-month input windows.');

if ~exist(config.input_dir, 'dir')
    mkdir(config.input_dir);
end
source_file = fullfile(config.isolated_data_dir, ...
    sprintf('cpi_%s_isolated.mat', split));
assert(isfile(source_file), ...
    'Missing isolated %s input file: %s', split, source_file);

% Deliberately name the allowed variables.  In particular, do not load y,
% y_scaled, target, or target_scaled from the isolated MAT file.
source = load(source_file, 'X', 'sample_id', 'x_start_date', ...
    'x_end_date', 'target_date', 'window_size');
required = {'X', 'sample_id', 'x_start_date', 'x_end_date', ...
    'target_date', 'window_size'};
for required_index = 1:numel(required)
    assert(isfield(source, required{required_index}), ...
        'Missing field %s in %s.', required{required_index}, source_file);
end

input_window_raw = double(source.X);
sample_id = double(source.sample_id(:));
x_start_date = source.x_start_date(:);
x_end_date = source.x_end_date(:);
target_date = source.target_date(:);
assert(size(input_window_raw, 2) == config.window_size, ...
    'The %s input window width is not %d.', split, config.window_size);
assert(size(input_window_raw, 1) == numel(sample_id), ...
    'Input rows and sample IDs are misaligned for %s.', split);
assert(all(isfinite(input_window_raw(:))) && isreal(input_window_raw), ...
    'Input windows must be finite real values.');
assert(numel(unique(sample_id)) == numel(sample_id), ...
    'sample_id values must be unique within %s.', split);

transform_file = fullfile(config.input_dir, ...
    'twin_input_transform_train_only.mat');
if strcmp(split, 'train')
    input_scaler_mean = mean(input_window_raw, 1);
    centered = input_window_raw - input_scaler_mean;
    input_scaler_scale = sqrt(mean(centered .^ 2, 1));
    input_scaler_scale(input_scaler_scale == 0) = 1;
    input_window_scaled = centered ./ input_scaler_scale;

    rng(config.random_seed, 'twister');
    mask = 2 * randi([0, 1], config.window_size, ...
        config.num_virtual_nodes) - 1;
    train_projection = input_window_scaled * mask / sqrt(config.window_size);
    train_peak = max(abs(train_projection(:)));
    assert(train_peak > 0 && isfinite(train_peak), ...
        'The training mask projection has an invalid peak.');
    mask_scale = config.target_masked_amplitude / train_peak;
    train_fit_sample_id = sample_id;
    mask_sha256 = sha256_numeric(mask);
    transform_schema_version = '1.0';
    expected_train_count = 50;
    if isfield(config, 'train_count')
        expected_train_count = double(config.train_count);
    end
    assert(numel(train_fit_sample_id) == expected_train_count, ...
        'Expected %d training windows, found %d.', ...
        expected_train_count, numel(train_fit_sample_id));
    transform_fit_scope = sprintf('train_%d_only', expected_train_count);
    save(transform_file, 'transform_schema_version', ...
        'transform_fit_scope', 'input_scaler_mean', ...
        'input_scaler_scale', 'mask', 'mask_scale', 'mask_sha256', ...
        'train_fit_sample_id', '-v7');
else
    assert(isfile(transform_file), ...
        ['Missing train-only twin transform. Run ', ...
         'prepare_twin_window_cache(''train'') first.']);
    transform = load(transform_file, 'input_scaler_mean', ...
        'input_scaler_scale', 'mask', 'mask_scale', 'mask_sha256', ...
        'train_fit_sample_id');
    expected_train_count = 50;
    if isfield(config, 'train_count')
        expected_train_count = double(config.train_count);
    end
    assert(numel(transform.train_fit_sample_id) == expected_train_count, ...
        'The twin transform was not fitted on exactly %d training windows.', ...
        expected_train_count);
    input_scaler_mean = double(transform.input_scaler_mean);
    input_scaler_scale = double(transform.input_scaler_scale);
    mask = double(transform.mask);
    mask_scale = double(transform.mask_scale);
    mask_sha256 = char(string(transform.mask_sha256));
    input_window_scaled = ...
        (input_window_raw - input_scaler_mean) ./ input_scaler_scale;
end

assert(isequal(size(mask), [config.window_size, config.num_virtual_nodes]), ...
    'The fixed twin mask must be 12 x 50.');
masked_input_cycle = ...
    (input_window_scaled * mask / sqrt(config.window_size)) * mask_scale;
drive_cycle = masked_input_cycle * config.input_gain;
assert(all(isfinite(drive_cycle(:))) && isreal(drive_cycle), ...
    'The prepared optical-reservoir drive must be finite and real.');

schema_version = '1.0';
state_protocol = 'explicit_twin_audited_unique_window_cache_v1';
sequence_protocol = 'isolated_repeated_window';
state_mode = 'isolated_repeated_window';
capture_cycle = repeat_count;
sample_phase = 'node_end';
theta_seconds = config.theta_seconds;
feedback_delay_seconds = config.feedback_delay_seconds;
input_transport_delay_seconds = config.warmup_seconds;
num_virtual_nodes = config.num_virtual_nodes;
input_transform_sha256 = sha256_file(transform_file);
% Hash only the allowlisted input payload.  The isolated source MAT also
% contains labels for later Python supervision; hashing the whole container
% here would read label bytes even though selective load() did not import
% those variables into the MATLAB state-generation workspace.
allowed_source_payload = struct( ...
    'input_window_raw', input_window_raw, ...
    'sample_id', sample_id, ...
    'x_start_date', {cellstr(string(x_start_date))}, ...
    'x_end_date', {cellstr(string(x_end_date))}, ...
    'target_date', {cellstr(string(target_date))}, ...
    'window_size', double(source.window_size));
source_input_sha256 = sha256_text( ...
    jsonencode(orderfields(allowed_source_payload)));

input_file = fullfile(config.input_dir, ...
    sprintf('twin_windows_%s.mat', split));
save(input_file, 'schema_version', 'state_protocol', 'sequence_protocol', ...
    'state_mode', ...
    'split', 'input_window_raw', 'input_window_scaled', ...
    'masked_input_cycle', 'drive_cycle', 'sample_id', 'x_start_date', ...
    'x_end_date', 'target_date', 'mask', 'mask_scale', 'mask_sha256', ...
    'input_scaler_mean', 'input_scaler_scale', ...
    'input_transform_sha256', 'source_input_sha256', 'repeat_count', ...
    'capture_cycle', 'sample_phase', 'theta_seconds', ...
    'feedback_delay_seconds', 'input_transport_delay_seconds', ...
    'num_virtual_nodes', '-v7');

% Guard the persisted schema against accidental label leakage.
persisted = whos('-file', input_file);
persisted_names = {persisted.name};
forbidden = {'y', 'y_scaled', 'target', 'target_scaled', 'cpi_actual'};
assert(~any(ismember(persisted_names, forbidden)), ...
    'A forbidden target-label field was written to %s.', input_file);
fprintf('Prepared %d label-free %s twin windows: %s\n', ...
    size(input_window_raw, 1), split, input_file);
end


function repeat_count = twin_repeat_count(config)
if isfield(config, 'repeat_count')
    repeat_count = double(config.repeat_count);
else
    repeat_count = 4;
end
assert(isscalar(repeat_count) && isfinite(repeat_count) && ...
    repeat_count == floor(repeat_count) && repeat_count >= 1, ...
    'repeat_count must be a positive integer.');
end


function digest = sha256_numeric(value)
message_digest = java.security.MessageDigest.getInstance('SHA-256');
bytes = typecast(double(value(:)), 'int8');
message_digest.update(bytes);
raw = typecast(message_digest.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2).', 1, []));
end


function digest = sha256_text(value)
message_digest = java.security.MessageDigest.getInstance('SHA-256');
bytes = unicode2native(char(value), 'UTF-8');
message_digest.update(typecast(uint8(bytes(:)), 'int8'));
raw = typecast(message_digest.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2).', 1, []));
end
