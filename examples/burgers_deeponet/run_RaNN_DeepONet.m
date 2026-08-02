clear;
clc;
close all;

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

[cfg,project_root] = config_burgers_deeponet();

results = burgers_deeponet_study(cfg,project_root);

%figure_files = plot_burgers_deeponet_result(results);
