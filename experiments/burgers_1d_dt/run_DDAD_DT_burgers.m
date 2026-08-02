clear;
clc;
close all;
warning('off', 'MATLAB:rankDeficientMatrix');
this_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(this_dir));

addpath(this_dir);
addpath(genpath(fullfile(project_root,'src')));

cfg = config_burgers_1d_dt('DDAD');

results = burgers_1d_dt_study(cfg,project_root);

plot_burgers_1d_dt_result(results);