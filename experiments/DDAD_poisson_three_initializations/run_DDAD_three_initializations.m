clear;
clc;
close all;
warning('off', 'MATLAB:rankDeficientMatrix');
this_dir = fileparts(mfilename('fullpath'));
experiments_dir = fileparts(this_dir);
project_root = fileparts(experiments_dir);

addpath(genpath(fullfile(project_root,'src')));
addpath(fullfile(project_root,'examples','poisson_2d'));
addpath(this_dir);

cfg = ddad_poisson_three_init_config();
results = ddad_poisson_three_init_study(cfg);
