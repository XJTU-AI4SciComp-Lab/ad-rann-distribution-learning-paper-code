function results = poisson_lshape_2d_study(cfg,project_root)
%POISSON_LSHAPE_2D_STUDY AD-RaNN on the singular L-shaped Poisson problem.
%
% This routine intentionally reuses the existing project components:
%   build_random_weights_nd
%   prepare_poisson_cache
%   evaluate_poisson_reduced_fast
%   optimize_distribution_adam
%   solve_least_squares
%   select_growth_centers
%   build_growth_directions
%   fit_growth_block_ddad
%   evaluate_growth_features
%   relative_l2 / relative_linf

    if nargin < 2 || isempty(project_root)
        project_root = locate_project_root_from_problem();
    end

    if ~strcmpi(cfg.activation,'gaussian')
        error(['The global fast evaluator used by this example is the ', ...
               'existing Gaussian Poisson evaluator. Set ', ...
               'cfg.activation=''gaussian''.']);
    end

    validate_required_functions();

    output_dir = fullfile(cfg.output_root,lower(cfg.method));
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    total_timer = tic;

    problem = build_poisson_lshape_problem(cfg);

    if cfg.growth.enabled
        num_base_features = cfg.growth.m1;
    else
        num_base_features = cfg.global_num_features;
    end

    base_basis = build_poisson_lshape_basis( ...
        num_base_features,cfg.domain,cfg.seed,cfg.geometry_tolerance);

    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    cache = prepare_poisson_cache(problem,base_basis);

    global_objective = @(p) ...
        evaluate_poisson_reduced_fast( ...
            p,cache,cfg.lambda,ls_opts);

    p0 = cfg.initialization.p0(:);

    [obj0,grad0,info0] = global_objective(p0);
    assert_finite_objective(obj0,grad0,'global initial objective');

    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('POISSON ON AN L-SHAPED DOMAIN -- %s\n',cfg.method);
    fprintf('====================================================================\n');
    fprintf('seed                         = %d\n',cfg.seed);
    fprintf('activation                   = %s\n',cfg.activation);
    fprintf('interior points              = %d\n',problem.num_interior);
    fprintf('boundary points              = %d\n',problem.num_boundary);
    fprintf('global/base features         = %d\n',num_base_features);
    fprintf('growth features              = %d\n', ...
        cfg.growth.enabled*cfg.growth.m2);
    fprintf('initial p                    = [%.6f, %.6f]\n',p0(1),p0(2));
    fprintf('initial residual MSE         = %.6e\n',info0.residual_mse);
    fprintf('====================================================================\n\n');

    global_timer = tic;

    [p_opt,global_history] = ...
        optimize_distribution_adam( ...
            p0,global_objective,cfg.optimizer);

    global_optimization_time = toc(global_timer);

    % Make the paper's first-layer W^(0), b^(0) explicit. The centers C
    % were sampled and verified inside the L-shaped domain.
    base_basis.W = base_basis.Z.*p_opt;
    base_basis.b = -sum(base_basis.C.*base_basis.W,1);

    [M_base,y_rhs,base_parts] = ...
        assemble_poisson_lshape_matrix( ...
            p_opt,problem,base_basis,cfg.activation);

    [coef_base,base_ls_info] = ...
        solve_least_squares(M_base,y_rhs,cfg.linear_solver);

    base_residual = M_base*coef_base-y_rhs;
    base_pde_residual = base_parts.Mi*coef_base-problem.fi;

    growth_basis = [];
    growth_centers = zeros(0,2);
    growth_center_indices = zeros(0,1);
    growth_center_scores = zeros(0,1);
    growth_directions = zeros(2,0);
    rho_opt = [];
    growth_history = [];
    growth_ddad_cache = [];
    growth_optimization_time = 0;

    M_final = M_base;
    coef_final = coef_base;

    if cfg.growth.enabled
        [growth_centers,growth_center_indices,growth_center_scores] = ...
            select_growth_centers( ...
                problem.Xi,base_pde_residual, ...
                cfg.growth.m2,cfg.growth.center_policy);

        if ~all(poisson_lshape_is_inside( ...
                growth_centers,false,cfg.geometry_tolerance))
            error('A layer-growth center lies outside the L-shaped domain.');
        end

        growth_directions = build_growth_directions( ...
            2,size(growth_centers,1), ...
            cfg.seed+cfg.growth.seed_offset);

        base_i = evaluate_poisson_lshape_base_state( ...
            problem.Xi,p_opt,base_basis,coef_base,2);

        base_b = evaluate_poisson_lshape_base_state( ...
            problem.Xb,p_opt,base_basis,coef_base,0);

        center_state = evaluate_poisson_lshape_base_state( ...
            growth_centers,p_opt,base_basis,coef_base,0);

        growth_timer = tic;

        [growth_basis,rho_opt,growth_history,growth_ddad_cache] = ...
            fit_growth_block_ddad( ...
                problem.Xi,base_i.u,base_i.phi,base_i.u, ...
                growth_centers,growth_directions,center_state.u, ...
                cfg.growth.rho0,cfg.growth.lambda,ls_opts, ...
                cfg.growth.optimizer);

        growth_optimization_time = toc(growth_timer);

        % Equations (4.5)-(4.8): the second-layer preactivation is
        %   h_j*(w0*phi1(x)-u0(c_j)).
        % These fields make the implicit W^(1)=H*w0 and
        % b^(1)=-H*u0(Xerr) construction explicit for inspection.
        growth_scale = sqrt(growth_basis.direction_norm_sq);
        growth_basis.second_layer_weights = ...
            growth_scale.*coef_base.';
        growth_basis.second_layer_bias = ...
            -growth_scale.*center_state.u;
        growth_basis.base_output_weights = coef_base(:);
        growth_basis.spatial_bias_centers = growth_centers;

        [M_final,~,~] = build_poisson_lshape_growth_system( ...
            problem,M_base,y_rhs,growth_basis,rho_opt,base_i,base_b);

        [coef_final,final_ls_info] = ...
            solve_least_squares( ...
                M_final,y_rhs,cfg.linear_solver);
    else
        final_ls_info = base_ls_info;
    end

    final_residual = M_final*coef_final-y_rhs;

    evaluation = evaluate_poisson_lshape_approximation( ...
        cfg,p_opt,base_basis,coef_final,rho_opt,growth_basis);

    method_time = toc(total_timer);

    results = struct();
    results.cfg = cfg;
    results.project_root = project_root;
    results.output_dir = output_dir;

    results.problem = problem;

    results.base_basis = base_basis;
    results.growth_basis = growth_basis;
    results.growth_centers = growth_centers;
    results.growth_center_indices = growth_center_indices;
    results.growth_center_scores = growth_center_scores;
    results.growth_directions = growth_directions;
    results.growth_ddad_cache = growth_ddad_cache;

    results.p0 = p0;
    results.p_opt = p_opt;
    results.rho_opt = rho_opt;

    results.global_history = global_history;
    results.growth_history = growth_history;

    results.coef = coef_final;
    results.frozen_base_coef = coef_base;
    results.num_base_features = num_base_features;
    results.num_growth_features = size(growth_centers,1);
    results.num_total_features = numel(coef_final);

    results.base_residual_mse = mean(base_residual.^2);
    results.base_pde_residual_mse = mean(base_pde_residual.^2);
    results.final_residual_mse = mean(final_residual.^2);

    results.global_optimization_time = global_optimization_time;
    results.growth_optimization_time = growth_optimization_time;
    results.method_time = method_time;

    results.base_ls_info = base_ls_info;
    results.final_ls_info = final_ls_info;

    results.evaluation = evaluation;

    summary = table( ...
        string(cfg.method), ...
        results.num_base_features, ...
        results.num_growth_features, ...
        results.num_total_features, ...
        p_opt(1),p_opt(2), ...
        scalar_or_nan(rho_opt), ...
        evaluation.relative_l2, ...
        evaluation.relative_linf, ...
        evaluation.relative_h1_seminorm, ...
        evaluation.relative_h1, ...
        results.final_residual_mse, ...
        method_time, ...
        'VariableNames',{ ...
            'Method', ...
            'BaseFeatures', ...
            'GrowthFeatures', ...
            'TotalFeatures', ...
            'p1', ...
            'p2', ...
            'rho', ...
            'RelativeL2', ...
            'RelativeLinf', ...
            'RelativeH1Seminorm', ...
            'RelativeH1', ...
            'FinalResidualMSE', ...
            'MethodTimeSec'});

    results.summary = summary;

    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('FINAL RESULT\n');
    fprintf('====================================================================\n');
    fprintf('p*                           = [%.6f, %.6f]\n', ...
        p_opt(1),p_opt(2));

    if ~isempty(rho_opt)
        fprintf('rho*                         = %.6f\n',rho_opt);
    end

    fprintf('total features               = %d\n',results.num_total_features);
    fprintf('relative L2 error            = %.6e\n',evaluation.relative_l2);
    fprintf('relative Linf error          = %.6e\n',evaluation.relative_linf);
    fprintf('relative H1 seminorm error   = %.6e\n', ...
        evaluation.relative_h1_seminorm);
    fprintf('relative H1 error            = %.6e\n',evaluation.relative_h1);
    fprintf('final residual MSE           = %.6e\n', ...
        results.final_residual_mse);
    fprintf('method time                  = %.6f s\n',method_time);
    fprintf('====================================================================\n\n');

    disp(summary);

    result_file = fullfile(output_dir, ...
        sprintf('poisson_lshape_2d_%s.mat',lower(cfg.method)));

    summary_file = fullfile(output_dir, ...
        sprintf('poisson_lshape_2d_%s_summary.csv',lower(cfg.method)));

    save(result_file,'results','-v7.3');
    writetable(summary,summary_file);

    fprintf('Saved MAT     : %s\n',result_file);
    fprintf('Saved summary : %s\n',summary_file);
end


function validate_required_functions()
    required = { ...
        'build_random_weights_nd', ...
        'prepare_poisson_cache', ...
        'evaluate_poisson_reduced_fast', ...
        'optimize_distribution_adam', ...
        'solve_least_squares', ...
        'activation_derivatives', ...
        'build_preactivation', ...
        'select_growth_centers', ...
        'build_growth_directions', ...
        'fit_growth_block_ddad', ...
        'evaluate_growth_features', ...
        'evaluate_poisson_lshape_base_state', ...
        'build_poisson_lshape_growth_system', ...
        'relative_l2', ...
        'relative_linf'};

    for k = 1:numel(required)
        if exist(required{k},'file') ~= 2
            error('Required src function not found: %s.m',required{k});
        end
    end
end


function assert_finite_objective(objective,gradient,label)
    if ~isfinite(objective) || any(~isfinite(gradient(:)))
        error('%s is non-finite.',label);
    end
end


function value = scalar_or_nan(x)
    if isempty(x)
        value = NaN;
    else
        value = x(1);
    end
end


function project_root = locate_project_root_from_problem()
    current_dir = fileparts(mfilename('fullpath'));

    while true
        if isfolder(fullfile(current_dir,'src')) && ...
                isfolder(fullfile(current_dir,'examples'))
            project_root = current_dir;
            return;
        end

        parent_dir = fileparts(current_dir);
        if strcmp(parent_dir,current_dir)
            error('Unable to locate project root.');
        end
        current_dir = parent_dir;
    end
end
