function [cfg,project_root] = config_DDAD_sharp_layer()
%CONFIG_DDAD_SHARP_LAYER
%
%   cfg = config_DDAD_sharp_layer()
%   [cfg,project_root] = config_DDAD_sharp_layer()
%
% Global distribution:
%   DDAD.
%
% Layer-growth scale:
%   DDAD.
%
% Expected problem-dependent structure:
%
%   <project-root>/src/problem_dependent/nonlinear_sharp_layer/
%       PDAD/
%       DDAD/
%
% This configuration activates only the DDAD subtree and removes the PDAD
% subtree from the MATLAB path, preventing method-specific functions with
% identical names from shadowing one another.
%
% There are two independent training-reduction switches:
%
%   1) cfg.ad_training_reduction.enabled
%      Controls only the global DDAD training of p:
%        - number of base features used during p training;
%        - number of existing PDE collocation points used during p training.
%
%   2) cfg.growth_training_reduction.enabled
%      Controls only the DDAD training of the layer-growth scale rho:
%        - number of base features;
%        - number of growth features;
%        - number of existing PDE interior collocation points.
%
% Both switches affect training only.
% All Newton PDE solves always use the full cfg.m1 and cfg.m2 spaces.

    % =====================================================================
    % Locate project root and activate only the DDAD implementation
    % =====================================================================
    project_root = initialize_sharp_layer_project('DDAD');

    % =====================================================================
    % Load the existing method-specific base configuration
    % =====================================================================
    if isempty(which('config'))
        error('config_DDAD_sharp_layer:MissingBaseConfig', ...
            ['Could not find config.m after activating the DDAD source ', ...
             'directory.']);
    end

    cfg = config();

    if ~isstruct(cfg)
        error('config_DDAD_sharp_layer:InvalidBaseConfig', ...
            'config() must return a structure.');
    end

    cfg.project_root = project_root;
    cfg.method = 'DDAD';
    cfg.growth_method = 'DDAD';

    % Ensure the inherited generic optimizer also uses the common bounds.
    if ~isfield(cfg,'optimizer') || ~isstruct(cfg.optimizer)
        cfg.optimizer = struct();
    end

    cfg.optimizer.lower_bound = 1e-3;
    cfg.optimizer.upper_bound = 300;

    % =====================================================================
    % Reproducibility
    % =====================================================================
    cfg.seed = 42;

    % =====================================================================
    % Problem
    % =====================================================================
    cfg.domain = [ ...
        0,1; ...
        0,1];

    cfg.activation = 'gaussian';

    cfg.epsilon_layer = 0.01;

    % =====================================================================
    % Full trial-space sizes used by all PDE/Newton solves
    % =====================================================================
    cfg.m1 = 2500;
    cfg.m2 = 500;

    % =====================================================================
    % Switch 1: global DDAD training reduction
    %
    % false:
    %   p training uses all m1 features and all configured PDE points.
    %
    % true:
    %   p training uses the manually specified smaller feature/point set.
    %
    % This switch has no effect on layer-growth rho training.
    % =====================================================================
    cfg.ad_training_reduction.enabled = false;

    cfg.ad_training_reduction.m1_train = 1000;

    % Global DDAD samples the numerical target only at existing interior
    % PDE collocation points.
    cfg.ad_training_reduction.num_points = 3000;

    % =====================================================================
    % Switch 2: layer-growth DDAD training reduction
    %
    % false:
    %   rho training uses all configured interior PDE points, all m1 base
    %   features and all m2 growth features.
    %
    % true:
    %   rho training uses the manually specified subset below.
    %
    % Residual centers are still selected from all problem.Xi.
    % The augmented Newton solve always uses the full m1+m2 space.
    % =====================================================================
    cfg.growth_training_reduction.enabled = false;

    cfg.growth_training_reduction.m1_train = 1000;
    cfg.growth_training_reduction.m2_train = 100;
    cfg.growth_training_reduction.num_points = 3000;

    % =====================================================================
    % PDE collocation and test sets
    % =====================================================================
    cfg.interior_grid = [100,100];
    cfg.boundary_points_per_side = 100;
    cfg.test_grid = [101,101];

    cfg.boundary_penalty = 10000;

    cfg.chunk_rows = 500;

    % =====================================================================
    % Linear solver
    % =====================================================================
    cfg.linear_solver.use_gpu = true;
    cfg.linear_solver.gpu_id = 1;
    cfg.linear_solver.compute_spectrum = false;

    % =====================================================================
    % Initial global distribution
    % =====================================================================
    cfg.initial_p = [15;15];

    % =====================================================================
    % Newton nonlinear solver
    % =====================================================================
    cfg.newton.maxit = 15;

    cfg.newton.residual_tol = 1e-8;
    cfg.newton.step_tol = 1e-10;
    cfg.newton.damping = 1.0;

    cfg.newton.line_search = true;
    cfg.newton.line_search_beta = 0.5;
    cfg.newton.line_search_min_alpha = 1e-6;
    cfg.newton.line_search_maxit = 20;

    cfg.newton.verbose = true;
    cfg.newton.initialization = 'linear_poisson';

    % =====================================================================
    % Global DDAD update control
    % =====================================================================
    cfg.ad.max_updates =2;
    cfg.ad.min_newton_iteration = 2;
    cfg.ad.trigger_relative_change = 0.10;

    cfg.ad.lambda = 1e-5;

    cfg.ad.optimizer = make_optimizer( ...
        cfg.optimizer, ...
        15, ...                  % maxit
        0.5, ...                 % learning rate
        'residual_mse', ...      % checkpoint metric
        1e-8, ...                % gradient tolerance
        1e-8, ...                % step tolerance
        1e-10, ...               % relative objective tolerance
        false);                  % verbose

    % =====================================================================
    % Numerical-target perturbation for DDAD components
    %
    % No separately generated physical solution-data points are used.
    % Both global DDAD and growth DDAD sample existing problem.Xi points.
    % =====================================================================
    cfg.data.noise_delta = 0;

    % =====================================================================
    % DDAD layer growth
    % =====================================================================
    cfg.growth.center_policy = 'top_abs';

    cfg.growth.rho0 = 1;
    cfg.growth.lambda = 1e-5;

    cfg.growth.optimizer = make_optimizer( ...
        cfg.optimizer, ...
        15, ...                  % maxit
        0.5, ...                 % learning rate
        'objective', ...         % checkpoint metric
        1e-11, ...               % gradient tolerance
        1e-11, ...               % step tolerance
        1e-11, ...               % relative objective tolerance
        true);                   % verbose

    % =====================================================================
    % Full augmented Newton solve after appending the growth block
    % =====================================================================
    cfg.growth.newton_maxit = 5;
    cfg.growth.newton_damping = 1.0;

    cfg.growth.line_search = true;
    cfg.growth.line_search_beta = 0.5;
    cfg.growth.line_search_min_alpha = 1e-6;
    cfg.growth.line_search_maxit = 20;

    cfg.growth.verbose = true;

    % =====================================================================
    % Output
    % =====================================================================
    cfg.output_dir_name = fullfile( ...
        'results','nonlinear_sharp_layer','DDAD');
end


function opt = make_optimizer( ...
    base_opt,maxit,learning_rate,selection_metric, ...
    grad_tol,step_tol,relative_obj_tol,verbose)
%MAKE_OPTIMIZER Construct one projected-Adam configuration.
%
% All distribution parameters use:
%
%   lower_bound = 1e-3
%   upper_bound = 300

    if nargin < 1 || isempty(base_opt)
        opt = struct();
    else
        opt = base_opt;
    end

    opt.maxit = maxit;
    opt.learning_rate = learning_rate;

    opt.beta1 = 0.9;
    opt.beta2 = 0.999;
    opt.epsilon = 1e-8;

    opt.lower_bound = 1e-3;
    opt.upper_bound = 300;

    opt.parameterization = 'direct';
    opt.selection_metric = selection_metric;

    opt.grad_tol = grad_tol;
    opt.step_tol = step_tol;
    opt.relative_obj_tol = relative_obj_tol;

    opt.patience = Inf;
    opt.min_delta = 0;

    opt.store_moments = true;
    opt.store_full_info = false;
    opt.verbose = logical(verbose);
end


function project_root = initialize_sharp_layer_project(method)
%INITIALIZE_SHARP_LAYER_PROJECT
% Locate the project root and activate one method-specific subtree.

    method = upper(strtrim(char(method)));

    if ~ismember(method,{'PDAD','DDAD'})
        error('method must be ''PDAD'' or ''DDAD''.');
    end

    persistent cached_project_root

    % =====================================================================
    % Locate project root
    % =====================================================================
    if ~isempty(cached_project_root) && ...
       exist(cached_project_root,'dir') == 7

        project_root = cached_project_root;

    else

        config_dir = fileparts(mfilename('fullpath'));

        probe = config_dir;
        project_root = '';

        for level = 1:12

            has_src = ...
                exist(fullfile(probe,'src'),'dir') == 7;

            has_examples = ...
                exist(fullfile(probe,'examples'),'dir') == 7;

            sharp_root_candidate = fullfile( ...
                probe, ...
                'src', ...
                'problem_dependent', ...
                'nonlinear_sharp_layer');

            has_pdad = ...
                exist(fullfile(sharp_root_candidate,'PDAD'),'dir') == 7;

            has_ddad = ...
                exist(fullfile(sharp_root_candidate,'DDAD'),'dir') == 7;

            if has_src && has_examples && has_pdad && has_ddad
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

            error('config_DDAD_sharp_layer:ProjectRootNotFound', ...
                [ ...
                'Could not locate a project root containing:\n', ...
                '  src/\n', ...
                '  examples/\n', ...
                '  src/problem_dependent/nonlinear_sharp_layer/PDAD/\n', ...
                '  src/problem_dependent/nonlinear_sharp_layer/DDAD/\n']);
        end

        cached_project_root = project_root;
    end

    % =====================================================================
    % Add public src and isolate the selected method
    % =====================================================================
    src_dir = fullfile(project_root,'src');

    sharp_root = fullfile( ...
        project_root, ...
        'src', ...
        'problem_dependent', ...
        'nonlinear_sharp_layer');

    pdad_dir = fullfile(sharp_root,'PDAD');
    ddad_dir = fullfile(sharp_root,'DDAD');

    addpath(genpath(src_dir));

    remove_path_tree(pdad_dir);
    remove_path_tree(ddad_dir);

    switch method

        case 'PDAD'
            method_dir = pdad_dir;

        case 'DDAD'
            method_dir = ddad_dir;
    end

    addpath(genpath(method_dir),'-begin');

    rehash path;
end


function remove_path_tree(root_dir)
%REMOVE_PATH_TREE Remove all existing MATLAB path entries below root_dir.

    if exist(root_dir,'dir') ~= 7
        return;
    end

    tree_entries = strsplit(genpath(root_dir),pathsep);
    current_entries = strsplit(path,pathsep);

    for k = 1:numel(tree_entries)

        entry = tree_entries{k};

        if isempty(entry)
            continue;
        end

        is_present = any(strcmpi(current_entries,entry));

        if is_present
            rmpath(entry);

            current_entries = current_entries( ...
                ~strcmpi(current_entries,entry));
        end
    end
end
