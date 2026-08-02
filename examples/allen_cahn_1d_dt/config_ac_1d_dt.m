function [cfg,project_root] = config_ac_1d_dt(method)
%CONFIG_AC_1D_DT Configuration for 1-D discrete-time Allen--Cahn.
%
%   cfg = config_ac_1d_dt(method)
%   [cfg,project_root] = config_ac_1d_dt(method)
%
% method = 'DDAD' or 'PDAD'.
%
% Expected location:
%
%   <project-root>/examples/allen_cahn_1d_dt/config_ac_1d_dt.m
%
% The problem-dependent implementation is expected below:
%
%   <project-root>/src/problem_dependent/allen_cahn_1d_dt/

    project_root = initialize_ac1d_paths(mfilename('fullpath'));

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
    %   u_t - epsilon*u_xx + alpha*(u^3-u) = 0
    % =====================================================================
    cfg.x_domain = [-1,1];
    cfg.t_domain = [0,1];

    cfg.epsilon_ac = 1e-4;
    cfg.alpha = 5;

    % Nt must be compatible with cfg.num_saved_snapshots.
    cfg.num_time_steps = 500;
    cfg.num_saved_snapshots = 25;

    cfg.num_collocation_points = 1000;
    cfg.num_features = 700;

    % Periodic value and derivative constraints.
    cfg.boundary_penalty = 100;

    % =====================================================================
    % Initial distribution parameter
    %
    % 'fixed':
    %   directly use cfg.initialization.fixed_p.
    %
    % 'ddad':
    %   optimize from cfg.initialization.ddad_p0 using the initial data.
    % =====================================================================
    cfg.initialization.mode = 'fixed';   % 'fixed' or 'ddad'

    cfg.initialization.fixed_p = 35;
    cfg.initialization.ddad_p0 = 35;
    cfg.initialization.lambda = 1e-6;

    init_opt = base_optimizer();

    init_opt.maxit = 30;
    init_opt.learning_rate = 10;
    init_opt.lower_bound = 1e-3;
    init_opt.upper_bound = 300;

    cfg.initialization.optimizer = init_opt;

    % =====================================================================
    % Algorithm-3 residual controls
    % =====================================================================
    cfg.adaptation.tau_k = 0.5;
    cfg.adaptation.tau_l = 1.0;
    cfg.adaptation.residual_epsilon = 1e-14;



    % =====================================================================
    % DDAD training
    %
    % The same frozen random basis should be used in DDAD training and
    % the physical PDE solve.
    % =====================================================================
    cfg.ddad.lambda = 1e-2;

    ddad_opt = base_optimizer();

    ddad_opt.maxit = 25;
    ddad_opt.learning_rate = 1;
    ddad_opt.lower_bound = 1e-3;
    ddad_opt.upper_bound = 300;

    cfg.ddad.optimizer = ddad_opt;

    % =====================================================================
    % PDAD training
    % =====================================================================
    cfg.pdad.lambda = 1e-6;

    pdad_opt = base_optimizer();

    pdad_opt.maxit = 25;
    pdad_opt.learning_rate = 1;
    pdad_opt.lower_bound = 1e-3;
    pdad_opt.upper_bound = 300;

    cfg.pdad.optimizer = pdad_opt;

    % =====================================================================
    % Linear least-squares solver
    % =====================================================================
    cfg.linear_solver.use_gpu = false;
    cfg.linear_solver.gpu_id = 1;
    cfg.linear_solver.compute_spectrum = false;

    % Keep these fields if AC1D uses the common cached QR routines.
    cfg.linear_solver.cache_qr = true;
    cfg.linear_solver.rank_tolerance = 1e-12;

    % =====================================================================
    % Logging / output
    % =====================================================================
    cfg.verbose = true;
    cfg.print_every = 1;

    cfg.output_root_name = ...
        fullfile('results','allen_cahn_1d_dt');
end


function opt = base_optimizer()
%BASE_OPTIMIZER Common Adam options for the AC1D experiment.

    opt = struct();

    opt.maxit = 15;
    opt.learning_rate = 0.5;

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
end


function project_root = initialize_ac1d_paths(config_file)
%INITIALIZE_AC1D_PATHS Locate the project root and add src recursively.
%
% config_file is expected to be:
%
%   <project-root>/examples/allen_cahn_1d_dt/config_ac_1d_dt.m

    persistent cached_project_root path_initialized

    if ~isempty(cached_project_root) && ...
       exist(cached_project_root,'dir') == 7

        project_root = cached_project_root;

        if isempty(path_initialized) || ~path_initialized
            addpath(genpath(fullfile(project_root,'src')));
            path_initialized = true;
        end

        return;
    end

    if nargin < 1 || isempty(config_file)
        error('The full path of config_ac_1d_dt.m is required.');
    end

    example_dir = fileparts(config_file);

    % example_dir:
    %   <project-root>/examples/allen_cahn_1d_dt
    %
    % fileparts(example_dir):
    %   <project-root>/examples
    %
    % fileparts(fileparts(example_dir)):
    %   <project-root>
    project_root = fileparts(fileparts(example_dir));

    required_dirs = { ...
        fullfile(project_root,'src'), ...
        fullfile(project_root,'data'), ...
        fullfile(project_root,'examples')};

    for k = 1:numel(required_dirs)

        if exist(required_dirs{k},'dir') ~= 7
            error( ...
                'Required project directory not found: %s', ...
                required_dirs{k});
        end
    end

    src_dir = fullfile(project_root,'src');

    % This includes all public source folders, including:
    %
    %   src/features
    %   src/data_driven
    %   src/optimization
    %   src/least_squares
    %   src/problem_dependent/allen_cahn_1d_dt
    addpath(genpath(src_dir));

    rehash path;

    required_functions = { ...
        'build_random_weights_nd', ...
        'build_preactivation', ...
        'activation_derivatives', ...
        'prepare_data_cache', ...
        'evaluate_data_reduced_fast', ...
        'optimize_distribution_adam', ...
        'solve_ridge', ...
        'solve_least_squares', ...
        'relative_l2'};

    for k = 1:numel(required_functions)

        function_path = which(required_functions{k});

        if isempty(function_path)
            error( ...
                'Required project function not found: %s', ...
                required_functions{k});
        end
    end

    % Add AC1D-specific checks after these files have been created.
    ac1d_functions = { ...
    'ac_1d_dt_study', ...
    'evaluate_ac_state', ...
    'evaluate_ac_pdad_reduced_fast', ...
    'load_ac_reference', ...
    'plot_ac_1d_dt_result'};

    for k = 1:numel(ac1d_functions)

        function_path = which(ac1d_functions{k});

        if isempty(function_path)
            warning( ...
                'AC1D problem-dependent function not found yet: %s', ...
                ac1d_functions{k});
        end
    end

    cached_project_root = project_root;
    path_initialized = true;

    fprintf('Project root: %s\n',project_root);
    fprintf('AC1D example: %s\n',example_dir);
    fprintf('AC1D implementation: %s\n', ...
        fullfile( ...
            project_root, ...
            'src', ...
            'problem_dependent', ...
            'allen_cahn_1d_dt'));
end