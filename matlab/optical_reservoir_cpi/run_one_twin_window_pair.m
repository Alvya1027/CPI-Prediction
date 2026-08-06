function result = run_one_twin_window_pair(target_drive_cycle, ...
        reference_drive_cycle, config, run_metadata)
%RUN_ONE_TWIN_WINDOW_PAIR Simulate one explicit two-branch window pair.
%
% Each 50-node drive is repeated four times from the same declared model
% initial conditions.  The final 50-node cycle is sampled at each node end.

if nargin < 3 || isempty(config)
    config = config_twin_cpi_rc();
end
if nargin < 4 || isempty(run_metadata)
    run_metadata = struct();
end
target_drive_cycle = double(target_drive_cycle(:));
reference_drive_cycle = double(reference_drive_cycle(:));
assert(numel(target_drive_cycle) == config.num_virtual_nodes, ...
    'The target drive cycle must contain exactly %d nodes.', ...
    config.num_virtual_nodes);
assert(numel(reference_drive_cycle) == config.num_virtual_nodes, ...
    'The reference drive cycle must contain exactly %d nodes.', ...
    config.num_virtual_nodes);
assert(all(isfinite(target_drive_cycle)) && isreal(target_drive_cycle), ...
    'The target drive must be finite and real.');
assert(all(isfinite(reference_drive_cycle)) && isreal(reference_drive_cycle), ...
    'The reference drive must be finite and real.');

repeat_count = 4;
if isfield(config, 'repeat_count')
    repeat_count = double(config.repeat_count);
end
assert(repeat_count == 4, ...
    'The formal isolated repeated-window protocol requires repeat_count=4.');
capture_cycle = repeat_count;

twin_model_file = build_twin_shared_reservoir_model(config, false);
assert(isfile(twin_model_file), ...
    'Twin model was not built: %s', twin_model_file);
if isempty(strfind(path, config.profile_root)) %#ok<STREMP>
    addpath(config.profile_root);
end
[~, twin_model_name] = fileparts(twin_model_file);
load_system(twin_model_file);

target_drive = repmat(target_drive_cycle, repeat_count, 1);
reference_drive = repmat(reference_drive_cycle, repeat_count, 1);
num_drive_points = numel(target_drive);
assert(numel(reference_drive) == num_drive_points, ...
    'Twin branch drive lengths do not match.');
drive_time_seconds = ...
    (0:num_drive_points - 1).' * config.theta_seconds;
simin_target = [drive_time_seconds, target_drive];
simin_reference = [drive_time_seconds, reference_drive];
simulation_stop_time = config.warmup_seconds + ...
    num_drive_points * config.theta_seconds;

simulation_input = Simulink.SimulationInput(twin_model_name);
simulation_input = setVariable(simulation_input, ...
    config.target_input_variable, simin_target);
simulation_input = setVariable(simulation_input, ...
    config.reference_input_variable, simin_reference);
simulation_input = setModelParameter(simulation_input, ...
    'StartTime', '0', ...
    'StopTime', num2str(simulation_stop_time, '%.17g'), ...
    'ReturnWorkspaceOutputs', 'on', ...
    'LoadInitialState', 'off', ...
    'SaveFinalState', 'off', ...
    'FastRestart', 'off');

evalin('base', sprintf('clear %s %s', ...
    config.target_output_variable, config.reference_output_variable));
simulation_output = sim(simulation_input);

target_log = find_logged_value(simulation_output, ...
    config.target_output_variable);
reference_log = find_logged_value(simulation_output, ...
    config.reference_output_variable);
[target_time, target_signal] = unpack_logged_value(target_log);
[reference_time, reference_signal] = unpack_logged_value(reference_log);

window_duration_seconds = ...
    config.num_virtual_nodes * config.theta_seconds;
capture_start_seconds = config.warmup_seconds + ...
    (capture_cycle - 1) * window_duration_seconds;
sample_times_seconds = capture_start_seconds + ...
    (1:config.num_virtual_nodes).' * config.theta_seconds;
target_state = sample_logged_state(target_time, target_signal, ...
    sample_times_seconds, config.theta_seconds);
reference_state = sample_logged_state(reference_time, reference_signal, ...
    sample_times_seconds, config.theta_seconds);

simulation_protocol = make_simulation_protocol(config, repeat_count, ...
    capture_cycle, sample_times_seconds);
simulation_protocol_sha256 = ...
    sha256_text(jsonencode(orderfields(simulation_protocol)));
shared_branch_model_sha256 = sha256_file(config.branch_model_file);
twin_model_sha256 = sha256_file(config.twin_model_file);
reservoir_parameter_sha256 = sha256_text(jsonencode(orderfields(struct( ...
    'shared_branch_model_sha256', shared_branch_model_sha256, ...
    'theta_seconds', config.theta_seconds, ...
    'feedback_delay_seconds', config.feedback_delay_seconds, ...
    'input_transport_delay_seconds', config.warmup_seconds, ...
    'input_gain', config.input_gain, ...
    'noise_seed', fixed_noise_seed(config), ...
    'solver', char(string(config.solver)), ...
    'fixed_step_seconds', double(config.fixed_step_seconds)))));

result = struct();
result.target_state = target_state(:).';
result.reference_state = reference_state(:).';
result.sample_times_seconds = sample_times_seconds(:).';
result.simulation_stop_time_seconds = simulation_stop_time;
result.repeat_count = repeat_count;
result.capture_cycle = capture_cycle;
result.sample_phase = 'node_end';
result.state_protocol = ...
    'explicit_twin_audited_unique_window_cache_v1';
result.sequence_protocol = 'isolated_repeated_window';
result.state_mode = 'isolated_repeated_window';
result.simulation_protocol = simulation_protocol;
result.simulation_protocol_sha256 = simulation_protocol_sha256;
result.shared_branch_model_sha256 = shared_branch_model_sha256;
result.twin_model_sha256 = twin_model_sha256;
result.reservoir_parameter_sha256 = reservoir_parameter_sha256;
result.run_metadata = run_metadata;
end


function value = find_logged_value(simulation_output, variable_name)
value = [];
try
    value = simulation_output.get(variable_name);
catch
    value = [];
end
if ~isempty(value)
    return;
end
try
    logsout = simulation_output.get('logsout');
    if isa(logsout, 'Simulink.SimulationData.Dataset')
        element = logsout.getElement(variable_name);
        if ~isempty(element)
            if isprop(element, 'Values')
                value = element.Values;
            else
                value = element;
            end
        end
    end
catch
    value = [];
end
if isempty(value) && ...
        evalin('base', sprintf('exist(''%s'', ''var'')', variable_name))
    value = evalin('base', variable_name);
end
assert(~isempty(value), ...
    'Simulation finished without logged value %s.', variable_name);
end


function [time, signal] = unpack_logged_value(value)
if isa(value, 'Simulink.SimulationData.Signal')
    value = value.Values;
end
if isa(value, 'timeseries')
    time = value.Time;
    raw_signal = value.Data;
elseif istimetable(value)
    time = seconds(value.Properties.RowTimes - value.Properties.RowTimes(1));
    raw_signal = value.Variables;
elseif isnumeric(value)
    assert(size(value, 2) >= 2, ...
        'Numeric twin log must contain time and signal columns.');
    time = value(:, 1);
    raw_signal = value(:, 2);
elseif isstruct(value) && isfield(value, 'time') && ...
        isfield(value, 'signals')
    time = value.time;
    raw_signal = value.signals.values;
elseif isstruct(value) && isfield(value, 'Time') && ...
        isfield(value, 'Data')
    time = value.Time;
    raw_signal = value.Data;
else
    error(['Unsupported twin To Workspace format. Use Timeseries, ', ...
        'Structure With Time, or a numeric time/signal array.']);
end
time = double(time(:));
raw_signal = double(raw_signal);
signal = reshape(raw_signal, size(raw_signal, 1), []);
signal = signal(:, 1);
assert(numel(time) == numel(signal), ...
    'Twin log time and signal lengths do not match.');
end


function state = sample_logged_state(time, signal, sample_times, theta)
valid = isfinite(time) & isfinite(signal);
time = double(time(valid));
signal = double(signal(valid));
[time, unique_index] = unique(time, 'stable');
signal = signal(unique_index);
assert(~isempty(time), 'The twin state log contains no finite samples.');
tolerance = max(theta / 2, 16 * eps(max(1, max(abs(time)))));
assert(min(time) <= sample_times(1) + tolerance, ...
    'Twin log starts after the first required final-cycle node.');
assert(max(time) >= sample_times(end) - tolerance, ...
    'Twin log ends before the last required final-cycle node.');
state = interp1(time, signal, sample_times, 'linear');
assert(all(isfinite(state)) && isreal(state), ...
    'Twin final-cycle state interpolation returned invalid values.');
end


function protocol = make_simulation_protocol(config, repeats, capture_cycle, ...
        sample_times)
protocol = struct();
protocol.schema_version = '1.0';
protocol.state_protocol = ...
    'explicit_twin_audited_unique_window_cache_v1';
protocol.sequence_protocol = 'isolated_repeated_window';
protocol.state_mode = 'isolated_repeated_window';
protocol.explicit_twin_topology = true;
protocol.shared_model_reference = true;
protocol.initial_state_reset_each_run = true;
protocol.common_noise_across_branches = true;
protocol.noise_seed = fixed_noise_seed(config);
protocol.repeat_count = repeats;
protocol.capture_cycle = capture_cycle;
protocol.sample_phase = 'node_end';
protocol.num_virtual_nodes = config.num_virtual_nodes;
protocol.theta_seconds = config.theta_seconds;
protocol.window_duration_seconds = ...
    config.num_virtual_nodes * config.theta_seconds;
protocol.feedback_delay_seconds = config.feedback_delay_seconds;
protocol.input_transport_delay_seconds = config.warmup_seconds;
protocol.input_gain = config.input_gain;
protocol.solver = char(string(config.solver));
protocol.fixed_step_seconds = double(config.fixed_step_seconds);
protocol.sample_times_seconds = sample_times(:).';
end


function seed = fixed_noise_seed(config)
if isfield(config, 'noise_seed')
    seed = double(config.noise_seed);
else
    seed = 1;
end
end


function digest = sha256_text(value)
message_digest = java.security.MessageDigest.getInstance('SHA-256');
bytes = unicode2native(char(value), 'UTF-8');
message_digest.update(typecast(uint8(bytes(:)), 'int8'));
raw = typecast(message_digest.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2).', 1, []));
end
