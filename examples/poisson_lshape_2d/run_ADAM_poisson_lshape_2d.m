clear;
clear functions;
clc;
close all;

warning('off','MATLAB:rankDeficientMatrix');

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir,'-begin');

[cfg,project_root] = config_poisson_lshape_2d('ADAM');

results = poisson_lshape_2d_study(cfg,project_root);
plot_poisson_lshape_result(results);
