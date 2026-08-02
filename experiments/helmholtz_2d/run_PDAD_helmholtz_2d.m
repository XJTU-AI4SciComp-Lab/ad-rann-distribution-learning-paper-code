clear;
clear functions;
clc;
close all;

% Choose one case:
%   '6_6'
%   '1_20'
case_name = '6_6';

[cfg,project_root] = config_helmholtz_2d(case_name,'PDAD');

% Reduced-feature training switch:
%   true  -> optimize p with m_train=1000, final solve with m=3000
%   false -> optimize p and solve with all m=3000 features
cfg.training_reduction.enabled = true;
cfg.training_reduction.num_features = 1000;

results = helmholtz_2d_study(cfg,project_root);

if cfg.plot.enabled
    plot_helmholtz_result(results);
end
