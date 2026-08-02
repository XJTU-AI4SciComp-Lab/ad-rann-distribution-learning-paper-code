function cfg = ddad_poisson_three_init_config()
%DDAD_POISSON_THREE_INIT_CONFIG
%
% DDAD Poisson noisy-target experiment with three initializations:
%
%   p0 = (1,1), (4,4), (4,8)
%
% The noisy solution data are used ONLY to optimize p.
% After p* is selected, the final reported solution is recomputed from the
% ORIGINAL Poisson PDE least-squares system.
%
% For each seed, every initialization/noise combination uses the SAME:
%   - 2400 random data points,
%   - random-feature realization,
%   - base rand noise vector.
%
% Therefore the comparison is paired and only p0 / delta change.

[cfg,project_root] = config();
cfg.linear_solver.use_gpu = false;
cfg.linear_solver.gpu_id = 4;
% -------------------------------------------------------------------------
% Randomized-feature / DDAD settings
% -------------------------------------------------------------------------
cfg.activation = 'gaussian';

cfg.num_features = 600;
cfg.lambda = 1e-3;

%Three requested initializations. Each ROW is one p0.
cfg.ddad.initial_p_list = [ ...
    1, 1; ...
    4, 4; ...
    4, 8];

% cfg.ddad.initial_p_list = [ ...
%     1, 1;];

% -------------------------------------------------------------------------
% Solution-data sampling
% -------------------------------------------------------------------------
cfg.ddad.num_data_points = 2400;

% Keep the current experiment levels from the supplied config.
cfg.ddad.noise_levels = [ ...
    0; ...
    1e-6; ...
    1e-4; ...
    1e-2];

cfg.ddad.seeds = 1:100;

% -------------------------------------------------------------------------
% Adam
% -------------------------------------------------------------------------
cfg.optimizer.maxit = 50;
cfg.optimizer.learning_rate = 0.3;
cfg.optimizer.beta1 = 0.9;
cfg.optimizer.beta2 = 0.999;
cfg.optimizer.epsilon = 1e-8;

cfg.optimizer.lower_bound = [1e-3;1e-3];
cfg.optimizer.upper_bound = [300;300];

cfg.optimizer.parameterization = 'direct';
cfg.optimizer.selection_metric = 'residual_mse';

% Exactly 50 Adam updates; no early stopping.
cfg.optimizer.grad_tol = 1e-11;
cfg.optimizer.step_tol = 1e-11;
cfg.optimizer.relative_obj_tol = 1e-11;
cfg.optimizer.patience = Inf;
cfg.optimizer.min_delta = 0;

cfg.optimizer.verbose = false;
cfg.optimizer.store_moments = true;
cfg.optimizer.store_full_info = false;

cfg.compute_spectrum = false;

% -------------------------------------------------------------------------
% Study behavior
% -------------------------------------------------------------------------
cfg.ddad.compute_fixed_pde_baseline = true;
cfg.ddad.resume = true;

% Do not silently convert code/interface failures into NaN summaries.
cfg.ddad.stop_on_error = true;

% Unique project-level directory: no checkpoint collision with other runs.
cfg.ddad.output_dir = fullfile( ...
    project_root, ...
    'results', ...
    'DDAD_poisson_three_initializations');

end
