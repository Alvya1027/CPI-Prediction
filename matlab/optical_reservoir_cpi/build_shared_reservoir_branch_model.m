function branch_model_file = build_shared_reservoir_branch_model(config, rebuild)
%BUILD_SHARED_RESERVOIR_BRANCH_MODEL Create one referenced SL_RC branch.
%
% The original root From Workspace source is replaced by an Inport and the
% reservoir response feeding Scope is branched to one Outport.  All physical
% subsystems and parameters remain copied from the same source model.

if nargin < 1 || isempty(config)
    config = config_twin_cpi_rc();
end
if nargin < 2
    rebuild = false;
end
assert(isfile(config.source_model_file), ...
    'Missing source SL_RC model: %s', config.source_model_file);
if ~exist(config.profile_root, 'dir')
    mkdir(config.profile_root);
end
branch_model_file = config.branch_model_file;
if isfile(branch_model_file) && ~rebuild
    return;
end

copyfile(config.source_model_file, branch_model_file, 'f');
[~, model_name] = fileparts(branch_model_file);
if bdIsLoaded(model_name)
    close_system(model_name, 0);
end
load_system(branch_model_file);
repair_legacy_noise_block(model_name);
remove_optional_spectrum_analyzer(model_name);

% Remove a previous logger from the copied model.  The twin top model owns
% both loggers so a referenced branch cannot overwrite another branch.
logger_path = [model_name '/CPI State Logger'];
if getSimulinkBlockHandle(logger_path) > 0
    logger_ports = get_param(logger_path, 'PortHandles');
    logger_line = get_param(logger_ports.Inport(1), 'Line');
    if logger_line > 0
        delete_line(logger_line);
    end
    delete_block(logger_path);
end

source_blocks = find_system(model_name, 'SearchDepth', 1, ...
    'BlockType', 'FromWorkspace');
assert(numel(source_blocks) == 1, ...
    'Expected exactly one root From Workspace source, found %d.', ...
    numel(source_blocks));
source_path = source_blocks{1};
source_position = get_param(source_path, 'Position');
source_ports = get_param(source_path, 'PortHandles');
source_line = get_param(source_ports.Outport(1), 'Line');
assert(source_line > 0, 'The root From Workspace source is not connected.');
destination_ports = get_param(source_line, 'DstPortHandle');
destination_ports = destination_ports(destination_ports > 0);
assert(~isempty(destination_ports), 'The root input source has no destination.');
delete_line(source_line);
delete_block(source_path);

input_path = [model_name '/ReservoirInput'];
input_handle = add_block('built-in/Inport', input_path, ...
    'Position', source_position, 'Port', '1');
input_ports = get_param(input_handle, 'PortHandles');
for destination_index = 1:numel(destination_ports)
    add_line(model_name, input_ports.Outport(1), ...
        destination_ports(destination_index), 'autorouting', 'on');
end

scope_path = [model_name '/Scope'];
assert(getSimulinkBlockHandle(scope_path) > 0, ...
    'The copied SL_RC model has no root Scope output.');
scope_ports = get_param(scope_path, 'PortHandles');
scope_line = get_param(scope_ports.Inport(1), 'Line');
assert(scope_line > 0, 'The root Scope is not connected.');
response_source_port = get_param(scope_line, 'SrcPortHandle');
output_path = [model_name '/ReservoirState'];
if getSimulinkBlockHandle(output_path) <= 0
    output_handle = add_block('built-in/Outport', output_path, ...
        'Position', [-2070, -905, -2040, -885], 'Port', '1');
    output_ports = get_param(output_handle, 'PortHandles');
    add_line(model_name, response_source_port, output_ports.Inport(1), ...
        'autorouting', 'on');
end

% A referenced model does not need its own visual Scope.  Removing it avoids
% multi-instance visualization restrictions and duplicate logging overhead;
% the same response signal is now exposed through ReservoirState and logged
% independently by the two blocks in Twin_SL_RC.
if getSimulinkBlockHandle(scope_path) > 0
    delete_block(scope_path);
end

% Both branches in Twin_SL_RC are instances of this one referenced model.
% Multi-instance support prevents Simulink from silently treating the
% branch as a single-use model.
set_param(model_name, 'ModelReferenceNumInstancesAllowed', 'Multi');

set_param(model_name, 'LoadInitialState', 'off', ...
    'SaveFinalState', 'off', 'FastRestart', 'off');
save_system(model_name, branch_model_file);
close_system(model_name, 0);
fprintf('Built shared reservoir branch: %s\n', branch_model_file);
end
