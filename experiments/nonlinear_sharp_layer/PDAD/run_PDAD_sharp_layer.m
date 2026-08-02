clear;
clc;
close all;

% Ensure that the configuration file in this example directory is visible.
this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

% Locate the project root, add the common src tree, and activate only
% src/problem_dependent/nonlinear_sharp_layer/PDAD.
[cfg,project_root] = config_PDAD_sharp_layer();

% Run the nonlinear sharp-layer PDAD experiment.
results = PDAD_sharp_layer_study(cfg);

% Plot the numerical results.
plot_sharp_layer_result(results,cfg,'PDAD');