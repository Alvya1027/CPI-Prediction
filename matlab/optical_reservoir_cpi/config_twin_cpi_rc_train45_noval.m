function config = config_twin_cpi_rc_train45_noval()
%CONFIG_TWIN_CPI_RC_TRAIN45_NOVAL Explicit Twin train45/test47 profile.
%
% There is no validation split.  All readout/reference hyperparameters must
% be fixed before the test state is generated or its labels are opened.

config = config_twin_cpi_rc();
profile_root = fullfile(config.project_dir, '..', ...
    'optical_reservoir_cpi_mom_train45_noval_20260807');
config.profile_root = profile_root;
config.isolated_data_dir = fullfile(profile_root, 'data');
config.input_dir = fullfile(profile_root, 'inputs_twin');
config.state_dir = fullfile(profile_root, 'states_twin');
config.audit_dir = fullfile(profile_root, 'audits_twin');
config.source_model_file = config.model_file;
config.branch_model_file = fullfile(profile_root, ...
    'SL_RC_shared_branch.slx');
config.twin_model_file = fullfile(profile_root, 'Twin_SL_RC.slx');

config.valid_splits = {'train', 'test'};
config.train_count = 45;
config.split_protocol = 'train45_no_validation_test47_20260807';
config.test_authorization_status = ...
    'train_only_fixed_authorized_for_test_state_generation';
config.no_validation_split = true;
end
