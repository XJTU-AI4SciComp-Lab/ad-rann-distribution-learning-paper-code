function [cfg,project_root] = config_poisson_lshape_2d(method)
%CONFIG_POISSON_LSHAPE_2D Configuration for the singular L-shaped Poisson test.
%
%   [cfg,project_root] = config_poisson_lshape_2d('ADAM')
%   [cfg,project_root] = config_poisson_lshape_2d('ADAM_GROWTH')
%
% The project root is located automatically. The current src tree and the
% L-shaped problem directory are put at the beginning of the MATLAB path.

    if nargin < 1 || isempty(method)
        method = 'ADAM';
    end

    method = upper(strtrim(char(method)));

    if ~ismember(method,{'ADAM','ADAM_GROWTH'})
        error('method must be ADAM or ADAM_GROWTH.');
    end

    config_dir = fileparts(mfilename('fullpath'));
    project_root = locate_project_root(config_dir);

    src_dir = fullfile(project_root,'src');
    problem_dir = fullfile(src_dir,'problem_dependent','poisson_lshape_2d');
    poisson_core_dir = fullfile(src_dir,'problem_dependent','poisson_2d');

    if ~isfolder(src_dir)
        error('Cannot find src directory:\n%s',src_dir);
    end

    if ~isfolder(problem_dir)
        error(['Cannot find L-shaped problem directory:\n%s\n\n', ...
               'Copy the supplied folder to:\n', ...
               'src/problem_dependent/poisson_lshape_2d/'],problem_dir);
    end

    if ~isfolder(poisson_core_dir)
        error(['Cannot find the existing Poisson core directory:\n%s\n\n', ...
               'This example reuses prepare_poisson_cache.m and ', ...
               'evaluate_poisson_reduced_fast.m from poisson_2d.'], ...
               poisson_core_dir);
    end

    addpath(genpath(src_dir),'-begin');
    addpath(problem_dir,'-begin');
    addpath(config_dir,'-begin');
    rehash toolboxcache;

    assert_function_under('optimize_distribution_adam',src_dir);
    assert_function_under('feature_derivatives_2d',src_dir);
    assert_function_under('prepare_poisson_cache',poisson_core_dir);
    assert_function_under('evaluate_poisson_reduced_fast',poisson_core_dir);
    assert_function_under('poisson_lshape_2d_study',problem_dir);

    %% Problem and method
    cfg.method = method;
    cfg.seed = 42;
    cfg.activation = 'gaussian';

    cfg.domain = [ ...
        -1, 1; ...
        -1, 1];

    % The L-shaped domain is
    % (-1,1)^2 \ ([0,1] x [-1,0]).
    cfg.geometry_tolerance = 1e-12;

    %% Collocation and evaluation
    cfg.interior_grid = [80,80];
    cfg.interior_offset = 1e-6;

    % Number of intervals per unit boundary length. The total L-shaped
    % boundary length is eight, so this gives approximately 8*n points.
    cfg.boundary_intervals_per_unit = 100;

    cfg.test_grid = [201,201];
    cfg.evaluation_chunk_rows = 300;

    %% Least-squares formulation
    cfg.boundary_penalty = 100;
    cfg.lambda = 1e-5;

    %% Equal final feature budget
    % ADAM:        1100 global features.
    % ADAM_GROWTH: 1000 global + 100 residual-localized features.
    cfg.global_num_features = 1100;

    cfg.growth.enabled = strcmp(method,'ADAM_GROWTH');
    cfg.growth.m1 = 1000;
    cfg.growth.m2 = 100;

    %% Global distribution initialization and Adam
    cfg.initialization.p0 = [1;1];

    cfg.optimizer.maxit = 25;
    cfg.optimizer.learning_rate = 1;
    cfg.optimizer.beta1 = 0.9;
    cfg.optimizer.beta2 = 0.999;
    cfg.optimizer.epsilon = 1e-8;

    cfg.optimizer.lower_bound = 1e-3;
    cfg.optimizer.upper_bound = 300;

    cfg.optimizer.parameterization = 'direct';
    cfg.optimizer.selection_metric = 'residual_mse';

    cfg.optimizer.grad_tol = 1e-10;
    cfg.optimizer.step_tol = 1e-11;
    cfg.optimizer.relative_obj_tol = 1e-11;

    cfg.optimizer.patience = Inf;
    cfg.optimizer.min_delta = 0;
    cfg.optimizer.verbose = true;
    cfg.optimizer.store_moments = true;
    cfg.optimizer.store_full_info = false;

    %% Layer growth
    cfg.growth.center_policy = 'top_abs';
    cfg.growth.rho0 = 1;
    cfg.growth.lambda = 1e-3;
    cfg.growth.seed_offset = 42;

    cfg.growth.optimizer.maxit = 25;
    cfg.growth.optimizer.learning_rate = 0.3;
    cfg.growth.optimizer.beta1 = 0.9;
    cfg.growth.optimizer.beta2 = 0.999;
    cfg.growth.optimizer.epsilon = 1e-8;

    cfg.growth.optimizer.lower_bound = 1e-3;
    cfg.growth.optimizer.upper_bound = 300;

    cfg.growth.optimizer.parameterization = 'direct';
    cfg.growth.optimizer.selection_metric = 'residual_mse';

    cfg.growth.optimizer.grad_tol = 1e-11;
    cfg.growth.optimizer.step_tol = 1e-11;
    cfg.growth.optimizer.relative_obj_tol = 1e-11;

    cfg.growth.optimizer.patience = Inf;
    cfg.growth.optimizer.min_delta = 0;
    cfg.growth.optimizer.verbose = true;
    cfg.growth.optimizer.store_moments = true;
    cfg.growth.optimizer.store_full_info = false;

    %% Linear solver
    cfg.linear_solver.method = 'linsolve';
    cfg.linear_solver.use_gpu = false;
    cfg.linear_solver.gpu_id = 1;
    cfg.linear_solver.compute_spectrum = false;

    %% Output
    cfg.verbose = true;
    cfg.output_root = fullfile(project_root,'results','poisson_lshape_2d');
end


function project_root = locate_project_root(start_dir)
    current_dir = start_dir;

    while true
        if isfolder(fullfile(current_dir,'src')) && ...
                isfolder(fullfile(current_dir,'examples'))
            project_root = current_dir;
            return;
        end

        parent_dir = fileparts(current_dir);

        if strcmp(parent_dir,current_dir)
            error(['Unable to locate the project root from:\n%s\n\n', ...
                   'A valid root must contain src/ and examples/.'], ...
                   start_dir);
        end

        current_dir = parent_dir;
    end
end


function assert_function_under(function_name,expected_dir)
    selected_file = which(function_name);

    if isempty(selected_file)
        error('Required function not found: %s.m',function_name);
    end

    selected_file = normalize_path(selected_file);
    expected_dir = normalize_path(expected_dir);

    if ~startsWith(selected_file,[expected_dir,'/'])
        error(['Incorrect function selected on the MATLAB path.\n\n', ...
               'Function: %s\nSelected: %s\nExpected under: %s\n\n', ...
               'Run "which %s -all" to inspect duplicates.'], ...
               function_name,which(function_name),expected_dir,function_name);
    end
end


function p = normalize_path(p)
    p = strrep(char(p),'\','/');
    while numel(p) > 1 && p(end) == '/'
        p(end) = [];
    end
    if ispc
        p = lower(p);
    end
end
