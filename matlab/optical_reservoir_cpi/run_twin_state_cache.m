function outputs = run_twin_state_cache(split, config, perform_audit)
%RUN_TWIN_STATE_CACHE Generate one audited state per unique input window.
%
% Unique windows are packed two at a time into the explicit Twin_SL_RC
% topology.  Semantic CPI relationship pairs are intentionally not rerun;
% downstream code resolves h_i and h_j by sample_id from this deterministic
% cache.  Test generation is blocked until Python writes a freeze
% authorization file.

if nargin < 1 || isempty(split)
    split = 'train';
end
if nargin < 2 || isempty(config)
    config = config_twin_cpi_rc();
end
if nargin < 3
    perform_audit = true;
end
split = char(lower(string(split)));
valid_splits = {'train', 'val', 'test'};
if isfield(config, 'valid_splits')
    valid_splits = cellstr(string(config.valid_splits));
end
assert(any(strcmp(split, valid_splits)), ...
    'split is not enabled for this experiment profile.');
state_protocol = 'explicit_twin_audited_unique_window_cache_v1';

authorization_file = '';
authorization_sha256 = '';
authorization_status = 'not_required';
authorization_protocol_identity_sha256 = '';
if strcmp(split, 'test')
    authorization_file = fullfile(config.input_dir, ...
        'test_generation_authorization.json');
    assert(isfile(authorization_file), ...
        ['Test state generation is locked. Python must freeze validation ', ...
         'configuration and write %s first.'], authorization_file);
    authorization = jsondecode(fileread(authorization_file));
    expected_authorization_status = ...
        'validation_frozen_authorized_for_test_state_generation';
    if isfield(config, 'test_authorization_status')
        expected_authorization_status = ...
            char(string(config.test_authorization_status));
    end
    assert(isfield(authorization, 'status') && ...
        strcmp(char(string(authorization.status)), ...
        expected_authorization_status), ...
        'The test-generation authorization has an invalid status.');
    assert(isfield(authorization, 'state_protocol') && ...
        strcmp(char(string(authorization.state_protocol)), state_protocol), ...
        'The authorization state_protocol does not match this generator.');
    assert(isfield(authorization, 'protocol_identity_sha256') && ...
        is_sha256(authorization.protocol_identity_sha256), ...
        ['The test-generation authorization lacks a valid ', ...
         'protocol_identity_sha256.']);
    validate_test_authorization_artifacts(authorization, config);
    authorization_sha256 = sha256_file(authorization_file);
    authorization_status = char(string(authorization.status));
    authorization_protocol_identity_sha256 = ...
        char(string(authorization.protocol_identity_sha256));
end

input_file = prepare_twin_window_cache(split, config);
input_variables = whos('-file', input_file);
input_variable_names = {input_variables.name};
forbidden = {'y', 'y_scaled', 'target', 'target_scaled', 'cpi_actual'};
assert(~any(ismember(input_variable_names, forbidden)), ...
    'The prepared twin input unexpectedly contains a target-label field.');
input_data = load(input_file, 'schema_version', 'state_protocol', ...
    'sequence_protocol', 'state_mode', 'split', 'input_window_raw', ...
    'input_window_scaled', 'masked_input_cycle', 'drive_cycle', ...
    'sample_id', 'x_start_date', 'x_end_date', 'target_date', ...
    'mask_sha256', 'input_transform_sha256', 'source_input_sha256', ...
    'repeat_count', 'capture_cycle', 'sample_phase');
assert(strcmp(char(string(input_data.state_protocol)), state_protocol), ...
    'Prepared input state_protocol does not match the formal generator.');
assert(strcmp(char(string(input_data.sequence_protocol)), ...
    'isolated_repeated_window'), ...
    'Prepared inputs are not isolated repeated windows.');
assert(strcmp(char(string(input_data.state_mode)), ...
    'isolated_repeated_window'), ...
    'Prepared input state_mode is not isolated_repeated_window.');
assert(double(input_data.repeat_count) == 4 && ...
    double(input_data.capture_cycle) == 4, ...
    'Formal twin inputs must repeat four cycles and capture cycle four.');

sample_id = double(input_data.sample_id(:));
input_window_raw = double(input_data.input_window_raw);
input_window_scaled = double(input_data.input_window_scaled);
masked_input_cycle = double(input_data.masked_input_cycle);
drive_cycle = double(input_data.drive_cycle);
x_start_date = input_data.x_start_date(:);
x_end_date = input_data.x_end_date(:);
target_date = input_data.target_date(:);
num_samples = numel(sample_id);
assert(size(drive_cycle, 1) == num_samples && ...
    size(drive_cycle, 2) == config.num_virtual_nodes, ...
    'Prepared drive_cycle has an invalid shape.');

state_matrix = nan(num_samples, config.num_virtual_nodes);
cache_record_id = arrayfun(@(identifier) ...
    sprintf('%s:%d', split, identifier), sample_id, ...
    'UniformOutput', false);
branch_id = cell(num_samples, 1);
partner_sample_id = nan(num_samples, 1);
generation_run_id = cell(num_samples, 1);
num_generation_runs = ceil(num_samples / 2);
first_run = [];

for generation_index = 1:num_generation_runs
    target_row = 2 * generation_index - 1;
    reference_row = min(target_row + 1, num_samples);
    run_identifier = sprintf('%s:run:%03d', split, generation_index);
    run_metadata = struct( ...
        'generation_run_id', run_identifier, ...
        'target_sample_id', sample_id(target_row), ...
        'reference_sample_id', sample_id(reference_row), ...
        'split', split);
    fprintf('Twin %s cache run %d/%d: sample %g with sample %g\n', ...
        split, generation_index, num_generation_runs, ...
        sample_id(target_row), sample_id(reference_row));
    run_result = run_one_twin_window_pair( ...
        drive_cycle(target_row, :), drive_cycle(reference_row, :), ...
        config, run_metadata);
    if isempty(first_run)
        first_run = run_result;
    else
        assert(strcmp(run_result.simulation_protocol_sha256, ...
            first_run.simulation_protocol_sha256), ...
            'Simulation protocol changed during cache generation.');
        assert(strcmp(run_result.shared_branch_model_sha256, ...
            first_run.shared_branch_model_sha256), ...
            'Shared branch model changed during cache generation.');
        assert(strcmp(run_result.twin_model_sha256, ...
            first_run.twin_model_sha256), ...
            'Twin top model changed during cache generation.');
    end

    state_matrix(target_row, :) = run_result.target_state;
    branch_id{target_row} = 'target';
    partner_sample_id(target_row) = sample_id(reference_row);
    generation_run_id{target_row} = run_identifier;
    if reference_row ~= target_row
        state_matrix(reference_row, :) = run_result.reference_state;
        branch_id{reference_row} = 'reference';
        partner_sample_id(reference_row) = sample_id(target_row);
        generation_run_id{reference_row} = run_identifier;
    end
end
assert(all(isfinite(state_matrix(:))) && isreal(state_matrix), ...
    'Twin cache generation produced an invalid state.');

schema_version = '1.0';
sequence_protocol = 'isolated_repeated_window';
state_mode = 'isolated_repeated_window';
sample_phase = 'node_end';
repeat_count = 4;
capture_cycle = 4;
sample_times_seconds = first_run.sample_times_seconds;
simulation_protocol_sha256 = first_run.simulation_protocol_sha256;
shared_branch_model_sha256 = first_run.shared_branch_model_sha256;
twin_model_sha256 = first_run.twin_model_sha256;
reservoir_parameter_sha256 = first_run.reservoir_parameter_sha256;
input_transform_sha256 = char(string(input_data.input_transform_sha256));
mask_sha256 = char(string(input_data.mask_sha256));
source_input_sha256 = char(string(input_data.source_input_sha256));
cache_reuse_declared = true;
semantic_pairs_simulated_simultaneously = false;
pair_states_resolved_by_sample_id = true;

if ~exist(config.state_dir, 'dir')
    mkdir(config.state_dir);
end
state_file = fullfile(config.state_dir, ...
    sprintf('state_cache_%s.mat', split));
save(state_file, 'schema_version', 'state_protocol', ...
    'sequence_protocol', 'state_mode', 'split', 'state_matrix', 'sample_id', ...
    'target_date', 'x_start_date', 'x_end_date', ...
    'input_window_raw', 'input_window_scaled', 'masked_input_cycle', ...
    'cache_record_id', 'branch_id', 'partner_sample_id', ...
    'generation_run_id', 'simulation_protocol_sha256', ...
    'shared_branch_model_sha256', 'twin_model_sha256', ...
    'reservoir_parameter_sha256', 'input_transform_sha256', ...
    'mask_sha256', 'source_input_sha256', 'repeat_count', ...
    'capture_cycle', 'sample_phase', 'sample_times_seconds', ...
    'cache_reuse_declared', 'semantic_pairs_simulated_simultaneously', ...
    'pair_states_resolved_by_sample_id', '-v7');

persisted = whos('-file', state_file);
persisted_names = {persisted.name};
assert(~any(ismember(persisted_names, forbidden)), ...
    'A forbidden target-label field was written to %s.', state_file);

audit_outputs = ensure_global_audit(config, split, state_file, ...
    perform_audit, simulation_protocol_sha256, ...
    shared_branch_model_sha256, twin_model_sha256);
[initial_condition_inventory_sha256, initial_condition_inventory] = ...
    inventory_initial_conditions(config.branch_model_file);
generator_script = [mfilename('fullpath') '.m'];
assert(isfile(generator_script), ...
    'Could not resolve this MATLAB generator script for hashing.');
simulink_details = ver('Simulink');
assert(~isempty(simulink_details), 'Simulink version information is missing.');

manifest = struct();
manifest.schema_version = schema_version;
manifest.state_protocol = state_protocol;
manifest.split = split;
manifest.state_file = file_basename(state_file);
manifest.state_file_sha256 = sha256_file(state_file);
manifest.audit_mat_file = file_basename(audit_outputs.mat_file);
manifest.audit_mat_sha256 = sha256_file(audit_outputs.mat_file);
manifest.audit_json_file = file_basename(audit_outputs.json_file);
manifest.audit_json_sha256 = sha256_file(audit_outputs.json_file);
manifest.source_reservoir_model_sha256 = ...
    sha256_file(config.source_model_file);
manifest.shared_branch_model_sha256 = shared_branch_model_sha256;
manifest.twin_model_sha256 = twin_model_sha256;
manifest.branch_a_model_sha256 = shared_branch_model_sha256;
manifest.branch_b_model_sha256 = shared_branch_model_sha256;
manifest.branch_a_parameter_sha256 = reservoir_parameter_sha256;
manifest.branch_b_parameter_sha256 = reservoir_parameter_sha256;
manifest.branch_a_mask_sha256 = mask_sha256;
manifest.branch_b_mask_sha256 = mask_sha256;
manifest.input_transform_sha256 = input_transform_sha256;
manifest.simulation_protocol_sha256 = simulation_protocol_sha256;
manifest.generator_script_sha256 = sha256_file(generator_script);
manifest.input_file_sha256 = sha256_file(input_file);
manifest.initial_condition_inventory_sha256 = ...
    initial_condition_inventory_sha256;
manifest.cache_reuse_declared = true;
manifest.cache_generated_by_explicit_twin_model = true;
manifest.semantic_pairs_simulated_simultaneously = false;
manifest.pair_states_resolved_by_sample_id = true;
manifest.derived_pair_states_from_cache = true;
manifest.same_model_reference_for_both_branches = true;
manifest.no_cross_branch_reservoir_coupling = true;
manifest.reset_between_runs = true;
manifest.num_records = num_samples;
manifest.state_width = config.num_virtual_nodes;
manifest.window_size = config.window_size;
manifest.repeat_count = repeat_count;
manifest.capture_cycle = capture_cycle;
manifest.theta_seconds = config.theta_seconds;
manifest.feedback_delay_seconds = config.feedback_delay_seconds;
manifest.input_transport_delay_seconds = config.warmup_seconds;
manifest.noise_seed = config.noise_seed;
manifest.solver = config.solver;
manifest.fixed_step_seconds = config.fixed_step_seconds;
manifest.matlab_release = version('-release');
manifest.simulink_version = simulink_details(1).Version;
manifest.sequence_protocol = sequence_protocol;
manifest.state_mode = state_mode;
manifest.sample_phase = sample_phase;
manifest.num_twin_generation_runs = num_generation_runs;
manifest.source_input_sha256 = source_input_sha256;
manifest.initial_condition_inventory = initial_condition_inventory;
manifest.test_generation_authorization_file = ...
    file_basename_or_empty(authorization_file);
manifest.test_generation_authorization_sha256 = authorization_sha256;
manifest.test_generation_authorization_status = authorization_status;
manifest.protocol_identity_sha256 = ...
    authorization_protocol_identity_sha256;
manifest_file = fullfile(config.state_dir, ...
    sprintf('state_cache_%s.manifest.json', split));
write_json_file(manifest_file, manifest);

outputs = struct('input_file', input_file, 'state_file', state_file, ...
    'manifest_file', manifest_file, 'audit', audit_outputs);
fprintf('Saved %d x %d audited %s twin states: %s\n', ...
    size(state_matrix, 1), size(state_matrix, 2), split, state_file);
end


function validate_test_authorization_artifacts(authorization, config)
% Fail before test inputs are opened when a frozen audited artifact drifted.
assert(isfield(authorization, 'allowed_split') && ...
    strcmp(char(string(authorization.allowed_split)), 'test'), ...
    'The authorization does not permit test state generation.');
assert(isfield(authorization, 'test_labels_accessed') && ...
    isequal(authorization.test_labels_accessed, false), ...
    'The authorization reports that test labels were already accessed.');
assert(isfield(authorization, 'test_state_generated') && ...
    isequal(authorization.test_state_generated, false), ...
    'The authorization reports that test states were already generated.');

audit_mat_file = fullfile(config.audit_dir, ...
    'audit_twin_equivalence.mat');
audit_json_file = fullfile(config.audit_dir, ...
    'audit_twin_equivalence.json');
train_state_file = fullfile(config.state_dir, 'state_cache_train.mat');
data_manifest_file = fullfile(config.isolated_data_dir, ...
    'isolated_split_manifest.json');
artifacts = { ...
    'shared_branch_model_sha256', config.branch_model_file; ...
    'twin_model_sha256', config.twin_model_file; ...
    'audit_mat_sha256', audit_mat_file; ...
    'audit_json_sha256', audit_json_file; ...
    'train_state_sha256', train_state_file; ...
    'data_manifest_sha256', data_manifest_file};
if ~isfield(config, 'no_validation_split') || ...
        ~isequal(config.no_validation_split, true)
    validation_state_file = fullfile(config.state_dir, ...
        'state_cache_val.mat');
    artifacts = [artifacts(1:5, :); ...
        {'validation_state_sha256', validation_state_file}; ...
        artifacts(6, :)];
end
for artifact_index = 1:size(artifacts, 1)
    field_name = artifacts{artifact_index, 1};
    file_path = artifacts{artifact_index, 2};
    assert(isfield(authorization, field_name) && ...
        is_sha256(authorization.(field_name)), ...
        'The authorization lacks a valid %s.', field_name);
    assert(isfile(file_path), ...
        'The frozen authorized artifact is missing: %s', file_path);
    assert(strcmpi(char(string(authorization.(field_name))), ...
        sha256_file(file_path)), ...
        ['The frozen authorized artifact changed after validation: ', ...
         '%s'], file_path);
end

audit_json = jsondecode(fileread(audit_json_file));
assert(isfield(audit_json, 'audit_passed') && ...
    isequal(audit_json.audit_passed, true), ...
    'The frozen authorized global twin audit no longer declares a pass.');
assert(isfield(authorization, 'simulation_protocol_sha256') && ...
    is_sha256(authorization.simulation_protocol_sha256), ...
    'The authorization lacks a valid simulation_protocol_sha256.');
assert(isfield(audit_json, 'simulation_protocol_sha256') && ...
    strcmpi(char(string(authorization.simulation_protocol_sha256)), ...
        char(string(audit_json.simulation_protocol_sha256))), ...
    ['The simulation protocol in the global audit differs from the ', ...
     'validation-frozen authorization.']);
end


function outputs = ensure_global_audit(config, split, state_file, ...
        perform_audit, simulation_hash, branch_hash, twin_hash)
audit_mat_file = fullfile(config.audit_dir, 'audit_twin_equivalence.mat');
audit_json_file = fullfile(config.audit_dir, 'audit_twin_equivalence.json');
if ~isfile(audit_mat_file) || ~isfile(audit_json_file)
    assert(perform_audit, ...
        'The global twin audit is missing and perform_audit is false.');
    assert(strcmp(split, 'train'), ...
        ['The global twin audit must be generated from the first two ', ...
         'training windows before validation or test cache generation.']);
    outputs = audit_twin_equivalence('train', config, state_file, false);
    return;
end
audit_json = jsondecode(fileread(audit_json_file));
assert(isfield(audit_json, 'audit_passed') && ...
    isequal(audit_json.audit_passed, true), ...
    'The existing global twin equivalence audit did not pass.');
assert(isfield(audit_json, 'audit_mat_sha256') && ...
    strcmp(char(string(audit_json.audit_mat_sha256)), ...
    sha256_file(audit_mat_file)), ...
    'The existing global audit MAT hash is invalid.');
assert(strcmp(char(string(audit_json.simulation_protocol_sha256)), ...
    simulation_hash), 'The global audit uses another simulation protocol.');
assert(strcmp(char(string(audit_json.shared_branch_model_sha256)), ...
    branch_hash), 'The global audit uses another shared branch model.');
assert(strcmp(char(string(audit_json.twin_model_sha256)), twin_hash), ...
    'The global audit uses another twin model.');
if strcmp(split, 'train')
    assert(isfield(audit_json, 'state_file_sha256') && ...
        strcmp(char(string(audit_json.state_file_sha256)), ...
        sha256_file(state_file)), ...
        ['The immutable global audit belongs to another training cache. ', ...
         'Explicitly rebuild the audit after reviewing the protocol change.']);
end
outputs = struct('pass', true, 'mat_file', audit_mat_file, ...
    'json_file', audit_json_file);
end


function [digest, inventory] = inventory_initial_conditions(branch_model_file)
[~, model_name] = fileparts(branch_model_file);
load_system(branch_model_file);
integrators = find_system(model_name, 'LookUnderMasks', 'all', ...
    'FollowLinks', 'on', 'BlockType', 'Integrator');
integrators = sort(integrators(:));
relative_paths = cell(size(integrators));
initial_conditions = cell(size(integrators));
for block_index = 1:numel(integrators)
    relative_paths{block_index} = erase(integrators{block_index}, ...
        [model_name '/']);
    initial_conditions{block_index} = ...
        get_param(integrators{block_index}, 'InitialCondition');
end
inventory = struct();
inventory.integrator_paths = relative_paths;
inventory.initial_conditions = initial_conditions;
inventory.load_initial_state = 'off';
inventory.save_final_state = 'off';
inventory.fast_restart = 'off';
inventory.random_number_seed = 1;
inventory.reset_between_runs = true;
digest = sha256_text(jsonencode(orderfields(inventory)));
end


function digest = sha256_text(value)
message_digest = java.security.MessageDigest.getInstance('SHA-256');
bytes = unicode2native(char(value), 'UTF-8');
message_digest.update(typecast(uint8(bytes(:)), 'int8'));
raw = typecast(message_digest.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2).', 1, []));
end


function name = file_basename(file_path)
[~, stem, extension] = fileparts(file_path);
name = [stem extension];
end


function name = file_basename_or_empty(file_path)
if isempty(file_path)
    name = '';
else
    name = file_basename(file_path);
end
end


function result = is_sha256(value)
text_value = char(string(value));
result = ~isempty(regexp(text_value, '^[0-9a-fA-F]{64}$', 'once'));
end


function write_json_file(file_path, value)
encoded = jsonencode(value, 'PrettyPrint', true);
file_id = fopen(file_path, 'w', 'n', 'UTF-8');
assert(file_id >= 0, 'Could not open JSON output: %s', file_path);
cleanup = onCleanup(@() fclose(file_id)); %#ok<NASGU>
fprintf(file_id, '%s\n', encoded);
end
