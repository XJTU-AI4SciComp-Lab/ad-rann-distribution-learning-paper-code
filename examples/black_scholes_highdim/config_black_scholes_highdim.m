function [cfg,project_root] = config_black_scholes_highdim(d)
%CONFIG_BLACK_SCHOLES_HIGHDIM Configuration for high-dimensional BS PDAD-ST.
%
%   [cfg,project_root] = config_black_scholes_highdim(d)
%
% The randomized feature has d spatial inputs and one time input.  Only
% two distribution parameters are optimized:
%
%   p = [r_s; r_t],
%
% where r_s is shared by all spatial dimensions and r_t controls time.

    if nargin < 1 || isempty(d)
        d = 100;
    end

    if ~isscalar(d) || d < 1 || d ~= floor(d)
        error('d must be a positive integer.');
    end

    [project_root,example_dir,problem_dir] = initialize_project();

    cfg = struct();

    %% Reproducibility and model

    cfg.problem_name = 'black_scholes_highdim';
    cfg.dimension = d;
    cfg.seed = 42;
    cfg.activation = 'tanh';

    %% Black-Scholes problem

    cfg.x_lower = 90;
    cfg.x_upper = 110;
    cfg.t_domain = [0,1];

    cfg.mu = -0.05;
    cfg.sigma = 1/10 + (1:d)'/200;

    % The payoff is psi(x)=max(max_i x_i-strike,0).
    % Change this value and regenerate the corresponding data file.
    cfg.payoff_strike = 100;

    %% Input normalization used only by the randomized feature map

    cfg.normalization.x_shift = cfg.x_lower;
    cfg.normalization.x_scale = cfg.x_upper-cfg.x_lower;

    %% Full and reduced randomized feature spaces

    cfg.num_features = 3200;

    cfg.training_reduction.enabled = true;
    cfg.training_reduction.num_features = 800;

    % Frozen basis is generated once at full size.  Reduced training uses
    % the first 800 columns of exactly the same realization.
    cfg.feature_domain = repmat([0,1],d+1,1);

    %% Distribution parameter

    cfg.initial_p = [0.1;0.1];
    cfg.parameter_names = {'r_s','r_t'};

    %% Ridge-reduced PDAD training

    cfg.ridge_lambda = 1e-6;
    cfg.training.constraint_penalty = 10;

    opt = struct();
    opt.maxit = 50;
    opt.learning_rate = 1;

    opt.beta1 = 0.9;
    opt.beta2 = 0.999;
    opt.epsilon = 1e-8;

    opt.lower_bound = 1e-3;
    opt.upper_bound = 300;

    % EXP SWITCH
    % false: optimize p directly.
    % true : write p=exp(s), optimize s, and return the physical p.
    % The common src/optimization/optimize_distribution_adam.m already
    % applies grad_s = grad_p .* p when parameterization='log'.
    opt.use_exp_parameterization = true;

    if opt.use_exp_parameterization
        opt.parameterization = 'log';
    else
        opt.parameterization = 'direct';
    end

    opt.selection_metric = 'residual_mse';

    opt.grad_tol = 1e-10;
    opt.step_tol = 1e-11;
    opt.relative_obj_tol = 1e-11;

    opt.patience = Inf;
    opt.min_delta = 0;
    opt.store_moments = true;
    opt.store_full_info = false;
    opt.verbose = true;

    cfg.optimizer = opt;

    %% Final unregularized refit

    cfg.final.constraint_penalty = 100;
    cfg.final.assembly_chunk_rows = 512;
    cfg.final.test_chunk_rows = 1024;

    cfg.linear_solver.use_gpu = false;
    cfg.linear_solver.gpu_id = 1;
    cfg.linear_solver.compute_spectrum = false;

    %% Pre-sampled data sizes

    % Reduced parameter-training set.
    cfg.data.train.num_interior = 3276;
    cfg.data.train.num_boundary = 1638;
    cfg.data.train.num_initial = 1638;

    % Full final PDE least-squares set.
    cfg.data.solve.num_interior = 32768;
    cfg.data.solve.num_boundary = 16384;
    cfg.data.solve.num_initial = 16384;

    % The paper excerpt does not prescribe a test-set size explicitly.
    % This independent Monte-Carlo reference set is configurable.
    cfg.data.num_test = 16384;

    cfg.data.num_mc_samples = 16384;
    cfg.data.mc_point_batch_size = 8;

    cfg.data.sampling_seed = 1000+d;
    cfg.data.mc_seed_train = 2000+d;
    cfg.data.mc_seed_solve = 3000+d;
    cfg.data.mc_seed_test = 4000+d;

    cfg.data.auto_generate_if_missing = false;
    cfg.data.overwrite = false;

    cfg.data_dir = fullfile(example_dir,'data');
    cfg.data_file = fullfile( ...
        cfg.data_dir,sprintf('black_scholes_d%d.mat',d));

    %% Output

    cfg.verbose = true;
    cfg.save_results = true;

    cfg.output_dir = fullfile( ...
        project_root,'results','black_scholes_highdim',sprintf('d_%d',d));

    cfg.example_dir = example_dir;
    cfg.problem_dir = problem_dir;
    cfg.project_root = project_root;

    validate_cfg(cfg);
end


function [project_root,example_dir,problem_dir] = initialize_project()

    example_dir = fileparts(mfilename('fullpath'));
    probe = example_dir;
    project_root = '';

    for level = 1:12
        if exist(fullfile(probe,'src'),'dir') == 7 && ...
                exist(fullfile(probe,'examples'),'dir') == 7
            project_root = probe;
            break
        end

        parent = fileparts(probe);
        if strcmp(parent,probe)
            break
        end
        probe = parent;
    end

    if isempty(project_root)
        error(['Could not locate a project root containing both ', ...
               'src/ and examples/.']);
    end

    src_dir = fullfile(project_root,'src');
    problem_dir = fullfile( ...
        src_dir,'problem_dependent','black_scholes_highdim');

    if exist(problem_dir,'dir') ~= 7
        error('Problem-dependent folder not found: %s',problem_dir);
    end

    addpath(genpath(src_dir),'-begin');
    addpath(problem_dir,'-begin');
    addpath(example_dir,'-begin');

    required = { ...
        'build_random_weights_nd', ...
        'subset_random_basis', ...
        'build_preactivation', ...
        'activation_derivatives', ...
        'optimize_distribution_adam', ...
        'solve_ridge', ...
        'solve_least_squares'};

    for k = 1:numel(required)
        if exist(required{k},'file') ~= 2
            error('Required common src function not found: %s.m',required{k});
        end
    end
end


function validate_cfg(cfg)

    if cfg.x_upper <= cfg.x_lower
        error('x_upper must exceed x_lower.');
    end

    if numel(cfg.sigma) ~= cfg.dimension || any(cfg.sigma <= 0)
        error('sigma must contain one positive value per spatial dimension.');
    end

    if cfg.payoff_strike <= 0
        error('payoff_strike must be positive.');
    end

    if cfg.num_features < 1
        error('num_features must be positive.');
    end

    if cfg.training_reduction.enabled
        mt = cfg.training_reduction.num_features;
        if mt < 1 || mt > cfg.num_features || mt ~= floor(mt)
            error('Invalid training_reduction.num_features.');
        end
    end

    if cfg.optimizer.use_exp_parameterization && ...
            ~strcmp(cfg.optimizer.parameterization,'log')
        error('EXP switch and optimizer.parameterization are inconsistent.');
    end

    if cfg.optimizer.lower_bound <= 0
        error('Distribution lower bound must be positive.');
    end

    if cfg.optimizer.upper_bound <= cfg.optimizer.lower_bound
        error('Distribution upper bound must exceed the lower bound.');
    end

    if cfg.data.num_mc_samples < 1 || ...
            cfg.data.num_mc_samples ~= floor(cfg.data.num_mc_samples)
        error('num_mc_samples must be a positive integer.');
    end
end
