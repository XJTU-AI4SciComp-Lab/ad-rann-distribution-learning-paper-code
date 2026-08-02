clear;
clc;
close all;

this_dir = fileparts(mfilename('fullpath'));
experiments_dir = fileparts(this_dir);
project_root = fileparts(experiments_dir);

addpath(genpath(fullfile(project_root,'src')));
addpath(fullfile(project_root,'examples','poisson_2d'));
addpath(this_dir);

cfg = activation_config();
results = activation_study_core(cfg);
