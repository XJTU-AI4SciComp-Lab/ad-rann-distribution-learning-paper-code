clear;
clc;
close all;
warning('off', 'MATLAB:rankDeficientMatrix');
% Add the current example directory so MATLAB can find the config file.
this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

% Locate the project root, initialize the src paths, and configure DDAD.
[cfg,project_root] = config_ac_1d_dt('DDAD');

% DDAD training and the PDE solve automatically use the same frozen basis.
results = ac_1d_dt_study(cfg,project_root);

% Plot the numerical results.
plot_ac_1d_dt_result(results);