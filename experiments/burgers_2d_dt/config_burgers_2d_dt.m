function [cfg,project_root] = config_burgers_2d_dt(method)
%CONFIG_BURGERS_2D_DT Configuration for the 2-D Burgers experiment.
%
%   cfg = config_burgers_2d_dt()
%   cfg = config_burgers_2d_dt(method)
%   [cfg,project_root] = config_burgers_2d_dt(method)
%
% method:
%   'DDAD' or 'PDAD'
%
% Expected configuration-file location:
%
%   <project-root>/experiments/burgers_2d_dt/
%       config_burgers_2d_dt.m
%
% Expected problem-dependent source location:
%
%   <project-root>/src/problem_dependent/burgers_2d_dt/

    % =====================================================================
    % Locate project root and add the complete src tree
    % =====================================================================
    project_root = initialize_burgers2d_project();

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
    %   u_t + u(u_x+u_y) - epsilon*Delta u = 0,
    %   (x,y) in (0,1)^2, 0<t<=1.
    %
    % Exact solution:
    %
    %   u(x,y,t)
    %     = 1/(1+exp((x+y-t)/(2*epsilon))).
    % =====================================================================
    cfg.domain = [ ...
        0,1; ...
        0,1];

    cfg.t_domain = [0,1];

    cfg.epsilon_burgers = 1e-2;

    % =====================================================================
    % Time discretization
    % =====================================================================
    % Paper refinement levels: 50, 100, 200, 400.
    cfg.num_time_steps = 400;

    % Store fields at:
    
    cfg.num_saved_snapshots = 25;

    cfg.evaluation_grid_size = 101;

    if mod(cfg.num_time_steps,cfg.num_saved_snapshots) ~= 0
        error([ ...
            'cfg.num_time_steps must be divisible by ', ...
            'cfg.num_saved_snapshots.']);
    end

    % =====================================================================
    % Collocation and randomized trial space
    % =====================================================================
    cfg.num_collocation_x = 100;
    cfg.num_collocation_y = 100;

    % The historical code used 200 intervals, hence 201 points per side.
    cfg.num_boundary_per_side = 201;

    cfg.boundary_penalty = 100;

    cfg.m1 = 2000;
    cfg.m2 = 300;

    % =====================================================================
    % Initial distribution update
    % =====================================================================
    cfg.initialization.mode = 'ddad';   % 'ddad' or 'fixed'

    cfg.initialization.p0 = [1;1];
    cfg.initialization.fixed_p = [1;1];

    % Ridge parameter used only while optimizing the initial distribution.
    cfg.initialization.lambda = 1e-7;

    init_opt = base_optimizer();

    init_opt.maxit = 10;
    init_opt.learning_rate = 0.5;

    cfg.initialization.optimizer = init_opt;

    % =====================================================================
    % Algorithm-3 residual controls
    % =====================================================================
    cfg.adaptation.tau_k = 2;
    cfg.adaptation.tau_l = 2;

    cfg.adaptation.residual_epsilon = 1e-14;

    % If the selected p is numerically unchanged, do not rebuild the
    % Burgers operators and feature matrices.
    cfg.adaptation.min_relative_parameter_change = 1e-12;



    % =====================================================================
    % DDAD update of p
    % =====================================================================
    cfg.ddad.lambda = 1e-5;

    ddad_opt = base_optimizer();

    ddad_opt.maxit = 10;
    ddad_opt.learning_rate = 1;

    cfg.ddad.optimizer = ddad_opt;

    % =====================================================================
    % PDAD update of p
    % =====================================================================
    cfg.pdad.lambda = 1e-5;

    pdad_opt = base_optimizer();

    pdad_opt.maxit = 10;
    pdad_opt.learning_rate = 1;

    cfg.pdad.optimizer = pdad_opt;

    % =====================================================================
    % Optional layer growth
    %
    % MAIN SWITCH:
    %
    %   false -> use only the m1 base features
    %   true  -> use the m1+m2 augmented trial space
    % =====================================================================
    cfg.growth.enabled = false;

    % Build the first local feature block at the first accepted time step.
    cfg.growth.build_at_first_step = true;

    % Options:
    %
    %   'on_global_update'
    %       Refresh after an effective p change.
    %
    %   'every_step'
    %       Rebuild at every accepted time step.
    %
    %   'never_after_first'
    %       Build once and retain the same local block.
    cfg.growth.refresh_policy = 'on_global_update';

    cfg.growth.center_policy = 'top_abs';

    cfg.growth.rho0 = 1;
    cfg.growth.lambda = 1e-3;

    cfg.growth.use_previous_rho = true;
    cfg.growth.target_noise = 0;

    growth_opt = base_optimizer();

    growth_opt.maxit = 10;
    growth_opt.learning_rate = 0.3;

    % Algorithm 1 selects the historical residual-MSE checkpoint.
    growth_opt.selection_metric = 'residual_mse';

    cfg.growth.optimizer = growth_opt;

    % =====================================================================
    % Common least-squares and ridge options
    %
    % Final PDE coefficients are computed by:
    %
    %   src/least_squares/solve_least_squares.m
    % =====================================================================
    cfg.linear_solver.method = 'linsolve';

    cfg.linear_solver.use_gpu = false;
    cfg.linear_solver.gpu_id = 1;

    cfg.linear_solver.compute_spectrum = false;

    % Number of physical rows processed in one chunk by the fast PDAD
    % gradient evaluator.
    cfg.linear_solver.feature_chunk_rows = 500;

    % =====================================================================
    % Logging
    % =====================================================================
    cfg.verbose = true;
    cfg.print_every = 10;

    % =====================================================================
    % Output
    % =====================================================================
    cfg.output_root_name = ...
        fullfile('results','burgers_2d_dt');
end


function opt = base_optimizer()
%BASE_OPTIMIZER Common projected-Adam options.
%
% The scalar bounds are automatically expanded by
% optimize_distribution_adam.m to match the dimension of p.

    opt = struct();

    opt.maxit = 20;
    opt.learning_rate = 0.5;

    opt.beta1 = 0.9;
    opt.beta2 = 0.999;
    opt.epsilon = 1e-8;

    % Unified bounds for all optimization parameters:
    %
    %   p_x, p_y and rho.
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


function project_root = initialize_burgers2d_project()
%INITIALIZE_BURGERS2D_PROJECT Locate the project root and add src.
%
% The project root must contain:
%
%   src/
%   data/
%   examples/
%   src/problem_dependent/burgers_2d_dt/
%
% Only the src tree is added recursively. The data, results, tests,
% experiments and other example folders are not added to the MATLAB path.

    persistent cached_project_root

    % =====================================================================
    % Search upward from the current config directory
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

        has_burgers2d_source = ...
            exist( ...
                fullfile( ...
                    probe, ...
                    'src', ...
                    'problem_dependent', ...
                    'burgers_2d_dt'), ...
                'dir') == 7;

        if has_src && ...
           has_data && ...
           has_examples && ...
           has_burgers2d_source

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

        error('config_burgers_2d_dt:ProjectRootNotFound', ...
            [ ...
            'Could not locate the project root containing:\n', ...
            '  src/\n', ...
            '  data/\n', ...
            '  examples/\n', ...
            '  src/problem_dependent/burgers_2d_dt/\n']);
    end

    % =====================================================================
    % Add src recursively
    % =====================================================================
    src_dir = fullfile(project_root,'src');

    addpath(genpath(src_dir));

    % Cache the successfully identified project root.
    cached_project_root = project_root;

    fprintf('Project root: %s\n',project_root);
    fprintf('Burgers2D implementation: %s\n', ...
        fullfile( ...
            project_root, ...
            'src', ...
            'problem_dependent', ...
            'burgers_2d_dt'));
end
