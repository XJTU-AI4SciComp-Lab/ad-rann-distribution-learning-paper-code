
clear;
clc;
close all;

% Ensure the configuration file in the current example directory is visible.
this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

% Locate the project root, add the common src tree, and activate only
% src/problem_dependent/nonlinear_sharp_layer/DDAD.
[cfg,project_root] = config_DDAD_sharp_layer();

% Run the nonlinear sharp-layer DDAD experiment.
results = DDAD_sharp_layer_study(cfg);

% Plot the numerical results.
plot_sharp_layer_result(results,cfg,'DDAD');