function configured = ensure_cpi_state_logger(model_name)
%ENSURE_CPI_STATE_LOGGER Add a version-independent reservoir output logger.

scope_path = [model_name '/Scope'];
logger_path = [model_name '/CPI State Logger'];
assert(getSimulinkBlockHandle(scope_path) > 0, ...
    'Could not find the reservoir output Scope in SL_RC.');

logger_handle = getSimulinkBlockHandle(logger_path);
if logger_handle <= 0
    logger_handle = add_block('simulink/Sinks/To Workspace', logger_path, ...
        'Position', [-2070, -962, -1970, -928]);
end
set_param(logger_handle, ...
    'VariableName', 'CPIStateData', ...
    'SaveFormat', 'Timeseries');

logger_lines = get_param(logger_handle, 'LineHandles');
if isempty(logger_lines.Inport) || logger_lines.Inport(1) <= 0
    scope_ports = get_param(scope_path, 'PortHandles');
    scope_line = get_param(scope_ports.Inport(1), 'Line');
    assert(scope_line > 0, 'The reservoir output Scope has no input signal.');
    source_port = get_param(scope_line, 'SrcPortHandle');
    logger_ports = get_param(logger_handle, 'PortHandles');
    add_line(model_name, source_port, logger_ports.Inport(1), ...
        'autorouting', 'on');
end

save_system(model_name);
configured = true;
fprintf('Configured CPIStateData logging for the reservoir output signal.\n');
end
