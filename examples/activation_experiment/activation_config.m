function cfg = activation_config()
%ACTIVATION_CONFIG Configuration for the tanh/sin activation experiment.
%
% This file intentionally reuses the existing Poisson configuration and
% overrides only the settings needed by the activation study.

[cfg,project_root] = config();

% -------------------------------------------------------------------------
% Common AD-RaNN protocol
% -------------------------------------------------------------------------
cfg.lambda = 1e-7;
cfg.num_features = 600;

cfg.optimizer.maxit = 50;
cfg.optimizer.learning_rate = 1;
cfg.optimizer.beta1 = 0.9;
cfg.optimizer.beta2 = 0.999;
cfg.optimizer.epsilon = 1e-8;

cfg.optimizer.lower_bound = [1e-3;1e-3];
cfg.optimizer.upper_bound = [100;100];

cfg.optimizer.parameterization = 'direct';
cfg.optimizer.selection_metric = 'residual_mse';

% Exactly 50 Adam updates: no early stopping.
cfg.optimizer.grad_tol = 1e-8;
cfg.optimizer.step_tol = 1e-11;
cfg.optimizer.relative_obj_tol = 1e-11;
cfg.optimizer.patience = Inf;
cfg.optimizer.min_delta = 0;

cfg.optimizer.verbose = false;
cfg.optimizer.store_moments = true;
cfg.optimizer.store_full_info = false;

cfg.compute_spectrum = false;
cfg.use_fast_evaluator = true;

% -------------------------------------------------------------------------
% Common initialization for the activation ablation
%
% To isolate the effect of the activation function, Gaussian, tanh, and sin
% all use the same initial distribution parameters p0=(4,8).
% No activation-specific frequency initialization is used in this ablation.
% -------------------------------------------------------------------------
cfg.study.initial_p = [4;8];

% -------------------------------------------------------------------------
% Study settings
% -------------------------------------------------------------------------
cfg.study.activations = {'tanh','sin'};
cfg.study.seeds = 1:100;

cfg.study.compute_fixed_baseline = true;
cfg.study.resume = true;

cfg.study.output_dir = fullfile( ...
    project_root, ...
    'results', ...
    'activation_experiment', ...
    'results_common_p0_48');


end
