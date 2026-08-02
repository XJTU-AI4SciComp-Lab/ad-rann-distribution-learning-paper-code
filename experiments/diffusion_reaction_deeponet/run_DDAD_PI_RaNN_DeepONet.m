clear;
clear functions;
clc;
close all;
warning('off', 'MATLAB:rankDeficientMatrix');
this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

[cfg,project_root] = ...
    config_diffusion_reaction_deeponet('PI_RANN_DDAD');

results = diffusion_reaction_deeponet_study(cfg,project_root);

plot_diffusion_reaction_deeponet_result(results);
