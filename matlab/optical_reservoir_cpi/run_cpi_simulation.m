function response_file = run_cpi_simulation(split)
%RUN_CPI_SIMULATION Run the copied optical reservoir model for one split.

if nargin < 1
    split = 'train';
end

valid_splits = {'train', 'val', 'test'};
assert(any(strcmp(split, valid_splits)), ...
    'split must be train, val, or test.');

config = config_cpi_rc();
input_file = fullfile(config.input_dir, sprintf('simin_%s.mat', split));
assert(exist(input_file, 'file') == 2, ...
    'Input file not found. Run prepare_cpi_inputs first.');
assert(exist(config.model_file, 'file') == 2, ...
    'SL_RC.slx is missing from the CPI working directory.');

input_data = load(input_file);
evalin('base', 'clear ScopeData ScopeData1');
assignin('base', 'simin', input_data.simin);

[~, model_name] = fileparts(config.model_file);
load_system(config.model_file);
set_param(model_name, 'StopTime', ...
    num2str(input_data.simulation_stop_time, '%.17g'));

simulation_output = sim(model_name);

ScopeData = [];
scope_names = {'ScopeData', 'ScopeData1'};
for name_index = 1:numel(scope_names)
    scope_name = scope_names{name_index};
    try
        ScopeData = simulation_output.get(scope_name);
    catch
        ScopeData = [];
    end
    if isempty(ScopeData) && ...
            evalin('base', sprintf('exist(''%s'', ''var'')', scope_name))
        ScopeData = evalin('base', scope_name);
    end
    if ~isempty(ScopeData)
        break;
    end
end

if ~exist(config.response_dir, 'dir')
    mkdir(config.response_dir);
end

response_file = fullfile(config.response_dir, ...
    sprintf('response_%s.mat', split));
save(response_file, 'ScopeData', 'simulation_output', 'split', ...
    'input_file', 'config', '-v7.3');

if isempty(ScopeData)
    warning(['The simulation finished, but ScopeData was not found. ', ...
        'Check the To Workspace block name in SL_RC.slx.']);
else
    fprintf('Saved %s response to %s\n', split, response_file);
end
end
