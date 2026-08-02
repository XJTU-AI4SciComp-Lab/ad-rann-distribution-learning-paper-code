clear;
clc;
close all;

this_dir = fileparts(mfilename('fullpath'));
examples_dir = fileparts(this_dir);
project_root = fileparts(examples_dir);

addpath(genpath(fullfile(project_root,'src')));
addpath(fullfile(examples_dir,'poisson_2d'));
addpath(this_dir);

cfg = ddad_poisson_three_init_config();
results = ddad_poisson_three_init_study(cfg);
