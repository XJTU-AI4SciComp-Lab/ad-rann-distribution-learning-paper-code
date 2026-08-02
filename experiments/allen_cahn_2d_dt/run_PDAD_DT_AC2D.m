clear;
clc;
close all;
warning('off', 'all');
this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

[cfg,project_root] = config_ac_2d_dt('PDAD');

cfg.growth.enabled = true;
cfg.num_time_steps = 50;

results = ac_2d_dt_study(cfg,project_root);

plot_ac2d_dt_result(results);