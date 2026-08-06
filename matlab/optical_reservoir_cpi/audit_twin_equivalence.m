function outputs = audit_twin_equivalence(split, config, state_file, rebuild)
%AUDIT_TWIN_EQUIVALENCE Audit shared branches, swapping and cache reuse.
%
% The MAT audit preserves the original h arrays for A/B, B/A, A/A,
% repeated A/A and cache comparisons.  JSON contains tolerances, metrics,
% hashes and an overall pass/fail status.

if nargin < 1 || isempty(split)
    split = 'train';
end
if nargin < 2 || isempty(config)
    config = config_twin_cpi_rc();
end
if nargin < 4
    rebuild = false;
end
split = char(lower(string(split)));
assert(strcmp(split, 'train'), ...
    'The immutable global twin audit must use the first two train windows.');
if nargin < 3 || isempty(state_file)
    state_file = fullfile(config.state_dir, ...
        'state_cache_train.mat');
end
assert(isfile(state_file), 'Missing twin state cache: %s', state_file);
if ~exist(config.audit_dir, 'dir')
    mkdir(config.audit_dir);
end
audit_mat_file = fullfile(config.audit_dir, 'audit_twin_equivalence.mat');
audit_json_file = fullfile(config.audit_dir, 'audit_twin_equivalence.json');
assert(~xor(isfile(audit_mat_file), isfile(audit_json_file)), ...
    ['The immutable global audit is incomplete. Remove or restore both ', ...
     'audit files before an explicit rebuild.']);
if isfile(audit_mat_file) && isfile(audit_json_file) && ~rebuild
    existing = jsondecode(fileread(audit_json_file));
    assert(isfield(existing, 'audit_passed') && ...
        isequal(existing.audit_passed, true), ...
        'The immutable global twin audit does not declare a pass.');
    assert(isfield(existing, 'audit_mat_sha256') && ...
        strcmp(char(string(existing.audit_mat_sha256)), ...
        sha256_file(audit_mat_file)), ...
        'The immutable global twin audit MAT hash is invalid.');
    outputs = struct('pass', true, 'mat_file', audit_mat_file, ...
        'json_file', audit_json_file, 'metrics', existing.metrics);
    return;
end

input_file = fullfile(config.input_dir, ...
    'twin_windows_train.mat');
assert(isfile(input_file), 'Missing prepared twin inputs: %s', input_file);
input_data = load(input_file, 'drive_cycle', 'sample_id', ...
    'mask_sha256', 'input_transform_sha256', 'state_protocol');
cache = load(state_file, 'state_matrix', 'sample_id', ...
    'simulation_protocol_sha256', 'shared_branch_model_sha256', ...
    'twin_model_sha256', 'reservoir_parameter_sha256', ...
    'mask_sha256', 'state_protocol');
sample_id = double(input_data.sample_id(:));
assert(numel(sample_id) >= 2, ...
    'Twin equivalence audit requires at least two unique windows.');
assert(isequal(sample_id, double(cache.sample_id(:))), ...
    'Prepared input and state-cache sample IDs disagree.');

row_a = 1;
row_b = 2;
sample_id_a = sample_id(row_a);
sample_id_b = sample_id(row_b);
drive_a = double(input_data.drive_cycle(row_a, :));
drive_b = double(input_data.drive_cycle(row_b, :));

ab = run_one_twin_window_pair(drive_a, drive_b, config, ...
    struct('audit_case', 'AB', 'sample_id_a', sample_id_a, ...
    'sample_id_b', sample_id_b));
ba = run_one_twin_window_pair(drive_b, drive_a, config, ...
    struct('audit_case', 'BA', 'sample_id_a', sample_id_a, ...
    'sample_id_b', sample_id_b));
aa = run_one_twin_window_pair(drive_a, drive_a, config, ...
    struct('audit_case', 'AA', 'sample_id_a', sample_id_a));
aa_repeat = run_one_twin_window_pair(drive_a, drive_a, config, ...
    struct('audit_case', 'AA_repeat', 'sample_id_a', sample_id_a));

h_a_ab_branch_a = ab.target_state;
h_b_ab_branch_b = ab.reference_state;
h_b_ba_branch_a = ba.target_state;
h_a_ba_branch_b = ba.reference_state;
h_a_aa_branch_a = aa.target_state;
h_a_aa_branch_b = aa.reference_state;
h_a_repeat_branch_a = aa_repeat.target_state;
h_a_repeat_branch_b = aa_repeat.reference_state;
h_a_cache = double(cache.state_matrix(row_a, :));
h_b_cache = double(cache.state_matrix(row_b, :));

rtol = 1e-8;
if isfield(config, 'audit_rtol')
    rtol = double(config.audit_rtol);
end
atol = 1e-10;
if isfield(config, 'audit_atol')
    atol = double(config.audit_atol);
end
assert(isscalar(rtol) && isfinite(rtol) && rtol >= 0, ...
    'Twin audit rtol must be a finite nonnegative scalar.');
assert(isscalar(atol) && isfinite(atol) && atol >= 0, ...
    'Twin audit atol must be a finite nonnegative scalar.');

metrics = struct();
metrics.same_input_branches = compare_states( ...
    h_a_aa_branch_a, h_a_aa_branch_b, atol, rtol);
metrics.swap_a = compare_states( ...
    h_a_ab_branch_a, h_a_ba_branch_b, atol, rtol);
metrics.swap_b = compare_states( ...
    h_b_ab_branch_b, h_b_ba_branch_a, atol, rtol);
metrics.partner_independence_branch_a = compare_states( ...
    h_a_ab_branch_a, h_a_aa_branch_a, atol, rtol);
metrics.partner_independence_branch_b = compare_states( ...
    h_a_ba_branch_b, h_a_aa_branch_b, atol, rtol);
metrics.repeat_branch_a = compare_states( ...
    h_a_aa_branch_a, h_a_repeat_branch_a, atol, rtol);
metrics.repeat_branch_b = compare_states( ...
    h_a_aa_branch_b, h_a_repeat_branch_b, atol, rtol);
metrics.cache_a = compare_states( ...
    h_a_cache, h_a_ab_branch_a, atol, rtol);
metrics.cache_b = compare_states( ...
    h_b_cache, h_b_ab_branch_b, atol, rtol);
metric_names = fieldnames(metrics);
audit_pass = true;
for metric_index = 1:numel(metric_names)
    audit_pass = audit_pass && metrics.(metric_names{metric_index}).pass;
end

protocol_hashes = {ab.simulation_protocol_sha256, ...
    ba.simulation_protocol_sha256, aa.simulation_protocol_sha256, ...
    aa_repeat.simulation_protocol_sha256, ...
    char(string(cache.simulation_protocol_sha256))};
assert(numel(unique(protocol_hashes)) == 1, ...
    'Simulation protocol hashes changed during twin audit.');
branch_hashes = {ab.shared_branch_model_sha256, ...
    ba.shared_branch_model_sha256, aa.shared_branch_model_sha256, ...
    aa_repeat.shared_branch_model_sha256, ...
    char(string(cache.shared_branch_model_sha256))};
assert(numel(unique(branch_hashes)) == 1, ...
    'Shared branch model hashes changed during twin audit.');
twin_hashes = {ab.twin_model_sha256, ba.twin_model_sha256, ...
    aa.twin_model_sha256, aa_repeat.twin_model_sha256, ...
    char(string(cache.twin_model_sha256))};
assert(numel(unique(twin_hashes)) == 1, ...
    'Twin model hashes changed during twin audit.');

schema_version = '1.0';
state_protocol = ...
    'explicit_twin_audited_unique_window_cache_v1';
simulation_protocol_sha256 = protocol_hashes{1};
shared_branch_model_sha256 = branch_hashes{1};
twin_model_sha256 = twin_hashes{1};
reservoir_parameter_sha256 = ab.reservoir_parameter_sha256;
mask_sha256 = char(string(input_data.mask_sha256));
input_transform_sha256 = char(string(input_data.input_transform_sha256));
state_file_sha256 = sha256_file(state_file);

save(audit_mat_file, 'schema_version', 'state_protocol', 'split', ...
    'sample_id_a', 'sample_id_b', 'atol', 'rtol', 'audit_pass', ...
    'metrics', 'h_a_ab_branch_a', 'h_b_ab_branch_b', ...
    'h_b_ba_branch_a', 'h_a_ba_branch_b', 'h_a_aa_branch_a', ...
    'h_a_aa_branch_b', 'h_a_repeat_branch_a', ...
    'h_a_repeat_branch_b', 'h_a_cache', 'h_b_cache', ...
    'simulation_protocol_sha256', 'shared_branch_model_sha256', ...
    'twin_model_sha256', 'reservoir_parameter_sha256', ...
    'mask_sha256', 'input_transform_sha256', 'state_file_sha256', '-v7');

audit_json = struct();
audit_json.schema_version = schema_version;
audit_json.state_protocol = state_protocol;
audit_json.audit_mat_sha256 = sha256_file(audit_mat_file);
audit_json.simulation_protocol_sha256 = simulation_protocol_sha256;
audit_json.shared_branch_model_sha256 = shared_branch_model_sha256;
audit_json.twin_model_sha256 = twin_model_sha256;
audit_json.audit_atol = atol;
audit_json.audit_rtol = rtol;
audit_json.audit_passed = audit_pass;
audit_json.status = pass_status(audit_pass);
audit_json.sample_id_a = sample_id_a;
audit_json.sample_id_b = sample_id_b;
audit_json.metrics = metrics;
audit_json.raw_arrays_file = file_basename(audit_mat_file);
audit_json.state_file_sha256 = state_file_sha256;
audit_json.reservoir_parameter_sha256 = reservoir_parameter_sha256;
audit_json.mask_sha256 = mask_sha256;
audit_json.input_transform_sha256 = input_transform_sha256;
write_json_file(audit_json_file, audit_json);

outputs = struct('pass', audit_pass, 'mat_file', audit_mat_file, ...
    'json_file', audit_json_file, 'metrics', metrics);
fprintf('Twin %s equivalence audit: %s\n', split, ...
    upper(pass_status(audit_pass)));
assert(audit_pass, ...
    'Twin equivalence audit failed. Inspect %s.', audit_json_file);
end


function metric = compare_states(left, right, atol, rtol)
left = double(left(:));
right = double(right(:));
assert(numel(left) == numel(right), ...
    'Twin audit state vectors have different lengths.');
difference = abs(left - right);
scale = max(abs(left), abs(right));
allowed = atol + rtol * scale;
relative = difference ./ max(scale, realmin('double'));
metric = struct();
metric.pass = all(difference <= allowed);
metric.max_abs_error = max(difference);
metric.max_rel_error = max(relative);
metric.rmse = sqrt(mean((left - right) .^ 2));
end


function status = pass_status(value)
if value
    status = 'passed';
else
    status = 'failed';
end
end


function name = file_basename(file_path)
[~, stem, extension] = fileparts(file_path);
name = [stem extension];
end


function write_json_file(file_path, value)
encoded = jsonencode(value, 'PrettyPrint', true);
file_id = fopen(file_path, 'w', 'n', 'UTF-8');
assert(file_id >= 0, 'Could not open JSON output: %s', file_path);
cleanup = onCleanup(@() fclose(file_id)); %#ok<NASGU>
fprintf(file_id, '%s\n', encoded);
end
