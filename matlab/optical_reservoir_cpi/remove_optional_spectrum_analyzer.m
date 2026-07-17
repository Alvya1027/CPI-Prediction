function replaced = remove_optional_spectrum_analyzer(model_name)
%REMOVE_OPTIONAL_SPECTRUM_ANALYZER Replace a display-only unavailable block.

block_path = [model_name '/OS'];
block_handle = getSimulinkBlockHandle(block_path);
replaced = false;
if block_handle <= 0 || strcmp(get_param(block_handle, 'BlockType'), 'Terminator')
    return;
end

position = get_param(block_handle, 'Position');
line_handles = get_param(block_handle, 'LineHandles');
source_port = get_param(line_handles.Inport(1), 'SrcPortHandle');
delete_line(line_handles.Inport(1));
delete_block(block_handle);

terminator = add_block('simulink/Sinks/Terminator', block_path, ...
    'Position', position);
terminator_ports = get_param(terminator, 'PortHandles');
add_line(model_name, source_port, terminator_ports.Inport(1), ...
    'autorouting', 'on');

save_system(model_name);
replaced = true;
fprintf(['Replaced the display-only Spectrum Analyzer with a Terminator; ', ...
    'reservoir dynamics are unchanged.\n']);
end
