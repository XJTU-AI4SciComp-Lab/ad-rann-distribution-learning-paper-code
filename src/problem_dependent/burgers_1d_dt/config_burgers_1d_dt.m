function cfg = config_burgers_1d_dt(method)
%CONFIG_BURGERS_1D_DT Configuration for 1-D discrete-time Burgers.
%
% method = 'DDAD' or 'PDAD'.

    if nargin < 1 || isempty(method)
        method = 'DDAD';
    end

    method = upper(strtrim(char(method)));

    if ~ismember(method,{'DDAD','PDAD'})
        error('method must be ''DDAD'' or ''PDAD''.');
    end

    cfg = struct();

    % =====================================================================
    % Method / reproducibility
    % =====================================================================
    cfg.method = method;
    cfg.seed = 42;
    cfg.activation = 'gaussian';

    % =====================================================================
    % PDE
    %
    %   u_t + u u_x - nu u_xx = 0
    % =====================================================================
    cfg.x_domain = [-1,1];
    cfg.t_domain = [0,1];

    cfg.nu = 0.01/pi;

    cfg.num_time_steps =125;
    cfg.num_collocation_points = 1500;
    cfg.num_features = 700;

    cfg.boundary_penalty = 100;

    % =====================================================================
    % Initial distribution
    %
    % The historical scripts first optimized p from u^0.  Keep that
    % behavior by default for both methods so PDAD-DT and DDAD-DT start
    % from the same initialized trial space.
    % =====================================================================
    cfg.initial_p = 1;

    cfg.initialization.use_ddad = true;
    cfg.initialization.lambda = 1e-6;

    % =====================================================================
    % Algorithm-3 residual controls
    % =====================================================================
    cfg.adaptation.tau_k = 0.5;
    cfg.adaptation.tau_l = 1;
    cfg.adaptation.residual_epsilon = 1e-14;

    % At most one distribution update is allowed at a physical time step.
    cfg.adaptation.max_updates_per_step = 1;

    % =====================================================================
    % Ridge parameters used only while optimizing p
    % =====================================================================
    cfg.ddad.lambda = 1e-2;
    cfg.pdad.lambda = 1e-6;

    % =====================================================================
    % Shared Adam configuration
    %
    % Uses src/optimize_distribution_adam.m.
    % =====================================================================
    opt = struct();

    opt.maxit = 25;
    opt.learning_rate = 1;

    opt.beta1 = 0.9;
    opt.beta2 = 0.999;
    opt.epsilon = 1e-8;

    opt.lower_bound = 1e-3;
    opt.upper_bound = 300;

    opt.parameterization = 'direct';
    opt.selection_metric = 'residual_mse';

    opt.grad_tol = 1e-10;
    opt.step_tol = 1e-11;
    opt.relative_obj_tol = 1e-11;

    opt.patience = Inf;
    opt.min_delta = 0;

    opt.store_moments = true;
    opt.store_full_info = false;
    opt.verbose = false;

    cfg.initialization.optimizer = opt;
    cfg.ddad.optimizer = opt;
    cfg.pdad.optimizer = opt;

    % =====================================================================
    % Linear solver
    %
    % Final PDE coefficients are always obtained by the project
    % unregularized least-squares solver.
    % =====================================================================
    cfg.linear_solver.use_gpu = false;
    cfg.linear_solver.gpu_id = 1;
    cfg.linear_solver.compute_spectrum = false;

    % =====================================================================
    % Logging / output
    % =====================================================================
    cfg.verbose = true;
    cfg.print_every = 1;

    cfg.output_root_name = fullfile('results','burgers_1d_dt');
end
