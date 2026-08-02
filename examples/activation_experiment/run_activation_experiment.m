clear;
clc;
close all;

this_dir = fileparts(mfilename('fullpath'));
examples_dir = fileparts(this_dir);
project_root = fileparts(examples_dir);

addpath(genpath(fullfile(project_root,'src')));
addpath(fullfile(examples_dir,'poisson_2d'));
addpath(this_dir);

cfg = activation_config();
results = activation_study_core(cfg);
