function [cfg,project_root] = config_helmholtz_2d(case_name,method)
%CONFIG_HELMHOLTZ_2D Configuration for the 2-D Helmholtz AD-RaNN example.
%
%   [cfg,project_root] = config_helmholtz_2d(case_name,method)
%
% case_name:
%   '6_6'  -> (a1,a2)=(6,6),   interior grid 121 x 121
%   '1_20' -> (a1,a2)=(1,20),  interior grid 66 x 300
%
% method:
%   'FRE'  -> frequency initialization followed by a fixed-p PDE solve
%   'PDAD' -> frequency initialization followed by PDE-driven Adam
%
% The path initialization is intentionally contained in this config file.

    if nargin < 1 || isempty(case_name)
        case_name = '6_6';
    end

    if nargin < 2 || isempty(method)
        method = 'PDAD';
    end

    case_name = normalize_case_name(case_name);
    method = upper(strtrim(char(method)));

    if ~ismember(method,{'FRE','PDAD'})
        error('method must be ''FRE'' or ''PDAD''.');
    end

    %% Project paths

    config_dir = fileparts(mfilename('fullpath'));
    project_root = locate_project_root(config_dir);

    src_dir = fullfile(project_root,'src');
    problem_dir = fullfile(src_dir,'problem_dependent','helmholtz_2d');

    if ~isfolder(src_dir)
        error('Cannot find src directory:\n%s',src_dir);
    end

    if ~isfolder(problem_dir)
        error(['Cannot find Helmholtz problem directory:\n%s\n\n', ...
               'Expected: src/problem_dependent/helmholtz_2d/'],problem_dir);
    end

    % Put this project before stale copies already present on the MATLAB path.
    addpath(genpath(src_dir),'-begin');
    addpath(problem_dir,'-begin');

    %% Common problem settings

    cfg.problem_name = 'helmholtz_2d';
    cfg.case_name = case_name;
    cfg.method = method;

    cfg.seed = 42;
    cfg.domain = [-1,1;-1,1];
    cfg.k = 1;
    cfg.activation = 'gaussian';

    switch case_name

        case '6_6'
            cfg.a1 = 6;
            cfg.a2 = 6;
            cfg.interior_grid = [121,121];
            cfg.boundary_points_per_side = 121;
            cfg.test_grid = [241,241];
            cfg.initialization.manual_p = [12;12];

        case '1_20'
            cfg.a1 = 1;
            cfg.a2 = 20;
            cfg.interior_grid = [66,300];
            cfg.boundary_points_per_side = 300;
            cfg.test_grid = [201,801];
            cfg.initialization.manual_p = [6;36];

        otherwise
            error('Unsupported case: %s',case_name);
    end

    %% Random features

    % Final PDE solve uses this many features.
    cfg.num_features = 3000;

    % Optional reduced-feature training of p.
    % When enabled, p is optimized with the first m_train columns of the
    % same full random basis, then the final coefficients are recomputed
    % with all cfg.num_features features.
    cfg.training_reduction.enabled = true;
    cfg.training_reduction.num_features = 1000;

    %% Collocation and final evaluation

    cfg.interior_inset = 1e-6;
    cfg.boundary_penalty = 100;

    % Row chunks limit temporary feature-matrix memory. The assembled final
    % least-squares matrix is still stored because the common solver needs it.
    cfg.assembly_chunk_rows = 250;
    cfg.evaluation_chunk_rows = 2000;

    %% Frequency initialization

    cfg.initialization.method = 'frequency';

    % Same candidate grid and FFT protocol as the Poisson initializer.
    cfg.initialization.frequency.max_r = 100;
    cfg.initialization.frequency.num_candidates = 50;
    cfg.initialization.frequency.Fs = 100;
    cfg.initialization.frequency.batch_size = 64;
    cfg.initialization.frequency.exact_early_stop = true;

    %% PDAD reduced ridge objective

    cfg.lambda = 1e-9;
    cfg.use_fast_evaluator = true;

    %% Adam

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

    %% Linear algebra

    cfg.linear_solver.method = 'linsolve';
    cfg.linear_solver.use_gpu = false;
    cfg.linear_solver.gpu_id = 1;
    cfg.linear_solver.compute_spectrum = false;

    cfg.compute_spectrum = false;

    %% Output and plotting

    cfg.verbose = true;
    cfg.plot.enabled = true;
    cfg.plot.save = true;

    cfg.output_root = fullfile( ...
        project_root, ...
        'results', ...
        'helmholtz_2d', ...
        ['case_',case_name], ...
        lower(method));
end


function case_name = normalize_case_name(case_name)

    case_name = lower(strtrim(char(case_name)));
    case_name = strrep(case_name,'(', '');
    case_name = strrep(case_name,')', '');
    case_name = strrep(case_name,',', '_');
    case_name = strrep(case_name,'-', '_');
    case_name = strrep(case_name,' ', '');

    switch case_name
        case {'6_6','66'}
            case_name = '6_6';
        case {'1_20','120'}
            case_name = '1_20';
        otherwise
            error('case_name must identify (6,6) or (1,20).');
    end
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
                   'The project root must contain src/ and examples/.'], ...
                  start_dir);
        end

        current_dir = parent_dir;
    end
end
