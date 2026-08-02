clear;
clc;
close all;

% Ensure that the configuration file in the current example directory
% can be found even when this script is launched from another directory.
this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

% The configuration function locates the project root and adds src/.
[cfg,project_root] = config_burgers_2d_dt('DDAD');

% Paper refinement levels: 50, 100, 200, 400.
cfg.num_time_steps = 400;

results = burgers_2d_dt_study(cfg,project_root);

% Plotting is intentionally separated from the solver package.
% Use the scripts under:
%
%   results/burgers_2d_dt/plot_scripts/
%
% after the computation is complete.