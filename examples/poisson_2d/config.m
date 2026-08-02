function [cfg, project_root] = config_poisson_2d()
%CONFIG_POISSON_2D Settings for the 2D Poisson AD-RaNN example.
%
% Outputs:
%   cfg          - Configuration structure.
%   project_root - Root directory of the AD-RaNN project.
%
% Expected project structure:
%
%   project_root/
%   ├── examples/
%   │   └── poisson_2d/
%   │       ├── config_poisson_2d.m
%   │       └── run_poisson_2d.m
%   └── src/
%       └── problem_dependent/
%           └── poisson_2d/
%
% This configuration function:
%   1. Locates the project root.
%   2. Adds the complete src directory to the MATLAB path.
%   3. Constructs the Poisson configuration.

    %% ================================================================
    %  Initialize project paths
    %  ================================================================

    config_file = mfilename('fullpath');
    config_dir = fileparts(config_file);

    project_root = locate_project_root(config_dir);

    src_dir = fullfile(project_root, 'src');
    problem_dir = fullfile( ...
        src_dir, ...
        'problem_dependent', ...
        'poisson_2d');

    if ~isfolder(src_dir)
        error( ...
            'config_poisson_2d:MissingSrcDirectory', ...
            'Cannot find the src directory:\n%s', ...
            src_dir);
    end

    if ~isfolder(problem_dir)
        error( ...
            'config_poisson_2d:MissingProblemDirectory', ...
            ['Cannot find the Poisson problem directory:\n%s\n\n', ...
             'Expected location:\n', ...
             'src/problem_dependent/poisson_2d/'], ...
            problem_dir);
    end

    addpath(genpath(src_dir));

    %% ================================================================
    %  Reproducibility and domain
    %  ================================================================

    cfg.seed = 1;

    cfg.domain = [
        -1, 1;
        -1, 1
    ];

    %% ================================================================
    %  Trial space and collocation sets
    %  ================================================================

    cfg.num_features = 600;

    cfg.interior_grid = [30, 80];

    cfg.boundary_points_per_side = 100;

    cfg.test_grid = [101, 101];

    %% ================================================================
    %  PDE loss and regularization
    %  ================================================================

    cfg.boundary_penalty = 100;

    cfg.lambda = 1e-7;

    %% ================================================================
    %  Activation
    %  ================================================================

    % Supported choices:
    %   'gaussian'
    %   'tanh'
    %   'sin'
    %
    % The Gaussian choice follows the original Gaussian-feature code path.
    cfg.activation = 'gaussian';

    %% ================================================================
    %  Distribution initialization
    %  ================================================================

    % Supported initialization methods:
    %   'manual'
    %   'frequency'
    cfg.initialization.method = 'manual';

    cfg.initialization.manual_p = [4; 8];

    % Frequency-based initialization settings.
    cfg.initialization.frequency.max_r = 100;
    cfg.initialization.frequency.num_candidates = 50;
    cfg.initialization.frequency.Fs = 100;

    % Implementation-only settings. These do not alter the candidate set.
    cfg.initialization.frequency.batch_size = 64;
    cfg.initialization.frequency.exact_early_stop = true;

    %% ================================================================
    %  Adam optimization
    %  ================================================================

    cfg.optimizer.maxit = 50;
    cfg.optimizer.learning_rate = 1;

    cfg.optimizer.beta1 = 0.9;
    cfg.optimizer.beta2 = 0.999;
    cfg.optimizer.epsilon = 1e-8;

    % Scalar bounds are automatically expanded to match the dimension of p.
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

    % Save Adam first moments, second moments, and projection events.
    cfg.optimizer.store_moments = true;

    % Full per-iteration information may consume substantial memory.
    cfg.optimizer.store_full_info = false;

    %% ================================================================
    %  Linear solver
    %  ================================================================

    cfg.linear_solver.method = 'linsolve';

    % Only least-squares solves may use the GPU.
    cfg.linear_solver.use_gpu = false;
    cfg.linear_solver.gpu_id = 1;

    %% ================================================================
    %  Diagnostics and evaluator
    %  ================================================================

    % Spectrum computation is diagnostic only and is not used by Adam.
    cfg.compute_spectrum = false;

    % Use the cached Poisson-specific evaluator.
    cfg.use_fast_evaluator = true;

    %% ================================================================
    %  Output location
    %  ================================================================

    cfg.output_root = fullfile( ...
        project_root, ...
        'results', ...
        'poisson_2d');

end


function project_root = locate_project_root(start_dir)
%LOCATE_PROJECT_ROOT Search upward for the project root.
%
% A valid project root must contain both:
%   src/
%   examples/

    current_dir = start_dir;

    while true
        has_src = isfolder(fullfile(current_dir, 'src'));
        has_examples = isfolder(fullfile(current_dir, 'examples'));

        if has_src && has_examples
            project_root = current_dir;
            return;
        end

        parent_dir = fileparts(current_dir);

        if strcmp(parent_dir, current_dir)
            error( ...
                'config_poisson_2d:ProjectRootNotFound', ...
                ['Unable to locate the project root from:\n%s\n\n', ...
                 'A valid project root must contain both:\n', ...
                 '  src/\n', ...
                 '  examples/'], ...
                start_dir);
        end

        current_dir = parent_dir;
    end
end