clear;
clc;
close all;
warning('off', 'MATLAB:rankDeficientMatrix');
this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir,'-begin');

[cfg,project_root] = config_poisson_lshape_mixture();
results = poisson_lshape_mixture_study(cfg,project_root);
