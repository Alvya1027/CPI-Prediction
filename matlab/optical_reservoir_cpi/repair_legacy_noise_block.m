function repaired = repair_legacy_noise_block(model_name)
%REPAIR_LEGACY_NOISE_BLOCK Replace the removed laser-noise source.

repaired = false;
laser_path = [model_name '/' sprintf('SL\n') '/laser'];
block_name = sprintf('Gaussian Noise\nGenerator1');
block_path = [laser_path '/' block_name];
add_path = [laser_path '/Add'];
expected_position = [355, 258, 435, 302];

block_handle = getSimulinkBlockHandle(block_path);
if block_handle <= 0
    try
        block_handle = Simulink.ID.getHandle([model_name ':1253']);
    catch
        block_handle = -1;
    end
end

if block_handle > 0 && ~is_simulink_random_number(block_handle)
    expected_position = get_param(block_handle, 'Position');
    line_handles = get_param(block_handle, 'LineHandles');
    if ~isempty(line_handles.Outport) && line_handles.Outport(1) > 0
        delete_line(line_handles.Outport(1));
    end
    delete_block(block_handle);
    block_handle = -1;
    repaired = true;
end

if block_handle <= 0
    block_handle = add_block('simulink/Sources/Random Number', block_path, ...
        'Position', expected_position);
    repaired = true;
end

set_param(block_handle, ...
    'Mean', '0', ...
    'Variance', '2.5e8', ...
    'Seed', '1', ...
    'SampleTime', '1e-12');

noise_ports = get_param(block_handle, 'PortHandles');
noise_lines = get_param(block_handle, 'LineHandles');
if isempty(noise_lines.Outport) || noise_lines.Outport(1) <= 0
    assert(getSimulinkBlockHandle(add_path) > 0, ...
        'Could not find the laser Add block in SL_RC.');
    add_ports = get_param(add_path, 'PortHandles');
    destination_line = get_param(add_ports.Inport(2), 'Line');
    if destination_line > 0
        delete_line(destination_line);
    end
    add_line(laser_path, noise_ports.Outport(1), ...
        add_ports.Inport(2), 'autorouting', 'on');
    repaired = true;
end

if repaired
    save_system(model_name);
    fprintf(['Replaced and connected the unavailable laser Gaussian ', ...
        'noise source inside the laser subsystem.\n']);
end
end


function result = is_simulink_random_number(block_handle)
result = strcmp(get_param(block_handle, 'BlockType'), 'RandomNumber');
if result
    return;
end
try
    reference_block = get_param(block_handle, 'ReferenceBlock');
    result = contains(string(reference_block), ...
        'simulink/Sources/Random Number');
catch
    result = false;
end
end
