function model = fit_twin_shared_output_weights(alpha, pair_weight, config)
%FIT_TWIN_SHARED_OUTPUT_WEIGHTS Fit only the teacher-approved linear Wout.
%
% Simulink has already produced one fixed 50-node state per training window.
% This function loads CPI labels only after state generation and solves the
% same closed-form joint absolute/difference objective as the Python formal
% evaluator.  No mask, laser, feedback, or reservoir parameter is updated.

if nargin < 1 || isempty(alpha)
    alpha = 0;
end
if nargin < 2 || isempty(pair_weight)
    pair_weight = 1;
end
if nargin < 3 || isempty(config)
    config = config_twin_cpi_rc();
end
assert(isscalar(alpha) && isfinite(alpha) && alpha >= 0, ...
    'alpha must be finite and non-negative.');
assert(isscalar(pair_weight) && isfinite(pair_weight) && ...
    pair_weight >= 0, ...
    'pair_weight must be finite and non-negative.');

state_file = fullfile(config.state_dir, 'state_cache_train.mat');
label_file = fullfile(config.isolated_data_dir, ...
    'cpi_train_isolated.mat');
assert(isfile(state_file), 'Missing formal Twin train cache: %s', state_file);
assert(isfile(label_file), 'Missing isolated train labels: %s', label_file);
states = load(state_file, 'state_matrix', 'sample_id', ...
    'x_start_date', 'x_end_date', 'state_protocol');
labels = load(label_file, 'sample_id', 'y');
assert(strcmp(char(string(states.state_protocol)), ...
    'explicit_twin_audited_unique_window_cache_v1'), ...
    'The supplied states are not from the formal explicit Twin protocol.');

H = double(states.state_matrix);
state_sample_id = double(states.sample_id(:));
label_sample_id = double(labels.sample_id(:));
y_source = double(labels.y(:));
assert(isequal(size(H), [50, config.num_virtual_nodes]), ...
    'Formal closed50 training states must be 50 x %d.', ...
    config.num_virtual_nodes);
assert(numel(unique(state_sample_id)) == 50, ...
    'Training sample IDs must be unique.');
[found, label_position] = ismember(state_sample_id, label_sample_id);
assert(all(found), 'At least one Twin state has no isolated training label.');
y = y_source(label_position);

% Fit state normalization on the same 50 training states only.
state_mean = mean(H, 1);
state_scale = std(H, 1, 1);
state_scale(state_scale == 0) = 1;
Z = (H - state_mean) ./ state_scale;

% The project gap=1 rule requires the target window start month to be at
% least one month after the reference window end month.  For the fixed 50
% sliding windows this yields 1+...+38 = 741 chronological relations.
target_start_month = month_number(states.x_start_date);
reference_end_month = month_number(states.x_end_date);
gap_matrix = target_start_month(:) - reference_end_month(:).';
[pair_i, pair_j] = find(gap_matrix >= 1);
assert(numel(pair_i) == 741, ...
    'Expected 741 closed50 gap=1 relations, found %d.', numel(pair_i));
assert(all(pair_j ~= pair_i), 'A training relation cannot be a self-pair.');

% Give every eligible target month the same total difference-loss weight.
unique_targets = unique(pair_i, 'stable');
pair_balance = zeros(numel(pair_i), 1);
for target_index = 1:numel(unique_targets)
    rows = pair_i == unique_targets(target_index);
    pair_balance(rows) = ...
        1 / (numel(unique_targets) * sum(rows));
end
assert(abs(sum(pair_balance) - 1) < 1e-12, ...
    'Target-balanced pair weights must sum to one.');

num_train = size(Z, 1);
width = size(Z, 2);
absolute_design = [ones(num_train, 1), Z] / sqrt(num_train);
absolute_response = y / sqrt(num_train);
design_parts = {absolute_design};
response_parts = {absolute_response};
if pair_weight > 0
    pair_scale = sqrt(pair_weight * pair_balance);
    pair_design = [zeros(numel(pair_i), 1), Z(pair_i, :) - Z(pair_j, :)];
    pair_response = y(pair_i) - y(pair_j);
    design_parts{end + 1} = pair_design .* pair_scale; %#ok<AGROW>
    response_parts{end + 1} = pair_response .* pair_scale; %#ok<AGROW>
end
if alpha > 0
    ridge_design = [zeros(width, 1), sqrt(alpha) * eye(width)];
    design_parts{end + 1} = ridge_design; %#ok<AGROW>
    response_parts{end + 1} = zeros(width, 1); %#ok<AGROW>
end
design = vertcat(design_parts{:});
response = vertcat(response_parts{:});
solution = design \ response;
assert(all(isfinite(solution)), 'Shared Wout solve produced invalid values.');

model = struct();
model.intercept = solution(1);
model.Wout = solution(2:end);
model.state_mean = state_mean;
model.state_scale = state_scale;
model.alpha = alpha;
model.pair_weight = pair_weight;
model.num_original_train_months = num_train;
model.num_derived_pair_relations = numel(pair_i);
model.num_pair_target_months = numel(unique_targets);
model.sample_id = state_sample_id;
model.pair_i_sample_id = state_sample_id(pair_i);
model.pair_j_sample_id = state_sample_id(pair_j);
model.state_protocol = char(string(states.state_protocol));
model.only_output_weights_trained = true;
model.reservoir_parameters_trained = false;

output_file = fullfile(config.state_dir, ...
    'twin_shared_output_weights_demo.mat');
save(output_file, 'model', '-v7');
fprintf(['Fitted one shared Wout from 50 months and %d derived ', ...
    'relations: %s\n'], numel(pair_i), output_file);
end


function value = month_number(raw_dates)
dates = string(raw_dates(:));
assert(all(strlength(dates) == 7) && ...
    all(extractBetween(dates, 5, 5) == "-"), ...
    'Window dates must use YYYY-MM format.');
years = str2double(extractBetween(dates, 1, 4));
months = str2double(extractBetween(dates, 6, 7));
assert(all(isfinite(years)) && all(months >= 1 & months <= 12), ...
    'Window dates contain an invalid month.');
value = years * 12 + months;
end
