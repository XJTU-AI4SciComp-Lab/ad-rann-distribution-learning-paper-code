function [cfg,project_root] = config_ac_2d_dt(method)
%CONFIG_AC_2D_DT Configuration for the 2-D Allen--Cahn experiment.
%
%   cfg = config_ac_2d_dt()
%   cfg = config_ac_2d_dt(method)
%   [cfg,project_root] = config_ac_2d_dt(method)
%
% method = 'DDAD' or 'PDAD'.
%
% Expected location:
%
%   <project-root>/experiments/allen_cahn_2d_dt/
%       config_ac_2d_dt.m
%
% Problem-dependent implementation:
%
%   <project-root>/src/problem_dependent/allen_cahn_2d_dt/

    % =====================================================================
    % Locate project root and add src recursively
    % =====================================================================
    project_root = initialize_ac2d_project();

    % =====================================================================
    % Method
    % =====================================================================
    if nargin < 1 || isempty(method)
        method = 'DDAD';
    end

    method = upper(strtrim(char(method)));

    if ~ismember(method,{'DDAD','PDAD'})
        error('method must be ''DDAD'' or ''PDAD''.');
    end

    cfg = struct();

    % =====================================================================
    % Method and reproducibility
    % =====================================================================
    cfg.method = method;
    cfg.seed = 42;
    cfg.activation = 'gaussian';

    % =====================================================================
    % PDE
    %
    %   u_t - epsilon*Delta u + alpha*(u^3-u) = 0
    % =====================================================================
    cfg.domain = [ ...
        -1,1; ...
        -1,1];

    cfg.t_domain = [0,1];

    cfg.epsilon_ac = 1e-4;
    cfg.alpha = 2;

    % Paper refinement levels: 50, 100, 200, 400.
    cfg.num_time_steps = 400;
    cfg.num_saved_snapshots = 10;

    % =====================================================================
    % Collocation and randomized trial space
    % =====================================================================
    cfg.num_collocation_x = 80;
    cfg.num_collocation_y = 80;

    cfg.num_boundary_per_side = 80;
    cfg.boundary_penalty = 100;

    cfg.m1 = 1000;
    cfg.m2 = 300;

    % =====================================================================
    % Initial DDAD update required by the revised Algorithm 3
    % =====================================================================
    cfg.initialization.mode = 'ddad';  % 'ddad' or 'fixed'

    cfg.initialization.p0 = [1;1];
    cfg.initialization.fixed_p = [1;1];
    cfg.initialization.lambda = 1e-7;

    init_opt = base_optimizer(2);

    init_opt.maxit = 10;
    init_opt.learning_rate = 0.5;

    init_opt.lower_bound = [1e-3;1e-3];
    init_opt.upper_bound = [200;200];

    cfg.initialization.optimizer = init_opt;

    % =====================================================================
    % Algorithm-3 residual controls
    % =====================================================================
    cfg.adaptation.tau_k = 2;
    cfg.adaptation.tau_l = 2.0;
    cfg.adaptation.residual_epsilon = 1e-14;



    % =====================================================================
    % DDAD update of p
    % =====================================================================
    cfg.ddad.lambda = 1e-5;

    ddad_opt = base_optimizer(2);

    ddad_opt.maxit = 10;
    ddad_opt.learning_rate = 1;

    ddad_opt.lower_bound = [1e-3;1e-3];
    ddad_opt.upper_bound = [300;300];

    cfg.ddad.optimizer = ddad_opt;

    % =====================================================================
    % PDAD update of p
    % =====================================================================
    cfg.pdad.lambda = 1e-5;

    pdad_opt = base_optimizer(2);

    pdad_opt.maxit = 10;
    pdad_opt.learning_rate = 1;

    pdad_opt.lower_bound = [1e-3;1e-3];
    pdad_opt.upper_bound = [300;300];

    cfg.pdad.optimizer = pdad_opt;

    % =====================================================================
    % Optional layer growth
    %
    % MAIN SWITCH:
    %   true  -> m1+m2 trial space
    %   false -> base m1 trial space only
    % =====================================================================
    cfg.growth.enabled = true;

    % Build one local block at the first accepted step.
    cfg.growth.build_at_first_step = true;

    % Options:
    %
    %   'on_global_update'
    %       Refresh the local block after p changes.
    %
    %   'every_step'
    %       Rebuild at every accepted physical time step.
    %
    %   'never_after_first'
    %       Build once and retain the same local block.
    cfg.growth.refresh_policy = 'on_global_update';

    cfg.growth.center_policy = 'top_abs';

    cfg.growth.rho0 = 1;
    cfg.growth.lambda = 1e-2;

    cfg.growth.use_previous_rho = true;
    cfg.growth.target_noise = 0;

    growth_opt = base_optimizer(1);

    growth_opt.maxit = 10;
    growth_opt.learning_rate = 0.3;

    growth_opt.lower_bound = 1e-3;
    growth_opt.upper_bound = 300;

    cfg.growth.optimizer = growth_opt;

    % =====================================================================
    % Linear least-squares solver
    %
    % The physical solve uses QR. The factorization is cached while p and
    % the growth block remain unchanged.
    % =====================================================================
    cfg.linear_solver.method = 'linsolve';

    cfg.linear_solver.use_gpu = false;
    cfg.linear_solver.gpu_id = 1;

    cfg.linear_solver.compute_spectrum = false;

    cfg.linear_solver.cache_qr = true;
    cfg.linear_solver.rank_tolerance = 1e-12;

    % =====================================================================
    % Logging and output
    % =====================================================================
    cfg.verbose = true;
    cfg.print_every = 10;

    cfg.output_root_name = ...
        fullfile('results','allen_cahn_2d_dt');
end


function opt = base_optimizer(dim)
%BASE_OPTIMIZER Common projected-Adam options.

    if nargin < 1 || isempty(dim)
        dim = 1;
    end

    if ~isscalar(dim) || dim < 1 || dim ~= floor(dim)
        error('dim must be a positive integer.');
    end

    opt = struct();

    opt.maxit = 20;
    opt.learning_rate = 0.5;

    opt.beta1 = 0.9;
    opt.beta2 = 0.999;
    opt.epsilon = 1e-8;

    if dim == 1

        opt.lower_bound = 1e-3;
        opt.upper_bound = 100;

    else

        opt.lower_bound = 1e-3*ones(dim,1);
        opt.upper_bound = 200*ones(dim,1);
    end

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


function project_root = initialize_ac2d_project()
%INITIALIZE_AC2D_PROJECT Locate the project root and add the src tree.
%
% This function assumes config_ac_2d_dt.m is located somewhere below the
% project root. It searches upward for a directory containing:
%
%   src/
%   data/
%   examples/
%
% It then adds only the complete src tree to the MATLAB path.

    persistent cached_project_root

    % =====================================================================
    % Return the cached root when it is still valid
    % =====================================================================
    if ~isempty(cached_project_root) && ...
       exist(cached_project_root,'dir') == 7

        project_root = cached_project_root;

        src_dir = fullfile(project_root,'src');

        if exist(src_dir,'dir') ~= 7
            error( ...
                'Cached project root no longer contains src/: %s', ...
                project_root);
        end

        % addpath is harmless when the path already exists and ensures
        % recovery after a user calls restoredefaultpath.
        addpath(genpath(src_dir));

        return;
    end

    % =====================================================================
    % Search upward from the current example directory
    % =====================================================================
    config_dir = fileparts(mfilename('fullpath'));

    probe = config_dir;
    project_root = '';

    for level = 1:12

        has_src = ...
            exist(fullfile(probe,'src'),'dir') == 7;

        has_data = ...
            exist(fullfile(probe,'data'),'dir') == 7;

        has_examples = ...
            exist(fullfile(probe,'examples'),'dir') == 7;

        has_ac2d_source = ...
            exist( ...
                fullfile( ...
                    probe, ...
                    'src', ...
                    'problem_dependent', ...
                    'allen_cahn_2d_dt'), ...
                'dir') == 7;

        if has_src && has_data && has_examples && has_ac2d_source

            project_root = probe;
            break;
        end

        parent = fileparts(probe);

        if strcmp(parent,probe)
            break;
        end

        probe = parent;
    end

    if isempty(project_root)

        error('config_ac_2d_dt:ProjectRootNotFound', ...
            [ ...
            'Could not locate the project root containing:\n', ...
            '  src/\n', ...
            '  data/\n', ...
            '  examples/\n', ...
            '  src/problem_dependent/allen_cahn_2d_dt/\n']);
    end

    % =====================================================================
    % Add the complete src tree
    % =====================================================================
    src_dir = fullfile(project_root,'src');

    addpath(genpath(src_dir));

    rehash path;

    % =====================================================================
    % Check common functions
    % =====================================================================
    required_common = { ...
        'build_random_weights_nd', ...
        'build_preactivation', ...
        'activation_derivatives', ...
        'prepare_data_cache', ...
        'evaluate_data_reduced_fast', ...
        'optimize_distribution_adam', ...
        'solve_ridge', ...
        'solve_least_squares', ...
        'relative_l2', ...
        'select_growth_centers', ...
        'build_growth_directions', ...
        'fit_growth_block_ddad', ...
        'evaluate_growth_features'};

    for k = 1:numel(required_common)

        function_path = which(required_common{k});

        if isempty(function_path)

            error('config_ac_2d_dt:MissingCommonFunction', ...
                'Required common function not found: %s', ...
                required_common{k});
        end
    end

    % =====================================================================
    % Check AC2D problem-dependent functions
    % =====================================================================
    required_ac2d = { ...
        'ac_2d_dt_study', ...
        'ac2d_growth_stage', ...
        'ac2d_initial_condition', ...
        'build_ac2d_problem', ...
        'evaluate_ac2d_base_state', ...
        'evaluate_ac2d_pdad_reduced_fast', ...
        'evaluate_ac2d_trial', ...
        'load_ac2d_reference', ...
        'plot_ac2d_dt_result', ...
        'prepare_ac2d_base_operator'};

    ac2d_source_dir = fullfile( ...
        project_root, ...
        'src', ...
        'problem_dependent', ...
        'allen_cahn_2d_dt');

    for k = 1:numel(required_ac2d)

        function_path = which(required_ac2d{k});

        if isempty(function_path)

            error('config_ac_2d_dt:MissingAC2DFunction', ...
                [ ...
                'Required AC2D function not found: %s\n', ...
                'Expected below:\n  %s'], ...
                required_ac2d{k}, ...
                ac2d_source_dir);
        end
    end

    % =====================================================================
    % Cache successful initialization
    % =====================================================================
    cached_project_root = project_root;

    fprintf('Project root: %s\n',project_root);
    fprintf('AC2D implementation: %s\n',ac2d_source_dir);
end
