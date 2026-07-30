function outputs = run_all_cpi_simulations(config)
%RUN_ALL_CPI_SIMULATIONS Prepare inputs, simulate all splits, and cache states.

if nargin < 1 || isempty(config)
    config = config_cpi_rc();
end

outputs = struct();
outputs.input_summary = prepare_cpi_inputs(config);

split_names = {'train', 'val', 'test'};
for split_index = 1:numel(split_names)
    split = split_names{split_index};
    fprintf('\n=== Running %s split ===\n', split);
    outputs.(split).response_file = run_cpi_simulation(split, config);
    outputs.(split).state_file = extract_cpi_states(split, config);
end

disp('All CPI reservoir states are ready.');
end
