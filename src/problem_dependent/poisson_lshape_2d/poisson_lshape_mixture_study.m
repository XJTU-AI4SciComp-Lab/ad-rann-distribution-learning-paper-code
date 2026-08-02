function results = poisson_lshape_mixture_study(cfg,project_root)
%POISSON_LSHAPE_MIXTURE_STUDY Seed-42 parameterization comparison.
%
% Compares:
%   (1) one global diagonal distribution with 1100 features;
%   (2) an additive two-component distribution with 550 near-origin and
%       550 far-origin centers and four independently optimized scales.

    if nargin < 2 || isempty(project_root)
        project_root = fileparts(fileparts(fileparts( ...
            fileparts(mfilename('fullpath')))));
    end

    if ~strcmpi(cfg.activation,'gaussian')
        error('The fast mixture evaluator requires Gaussian activation.');
    end

    if ~isfolder(cfg.output_root)
        mkdir(cfg.output_root);
    end

    problem = build_poisson_lshape_problem(cfg);
    seeds = cfg.study.seeds(:);
    nseed = numel(seeds);

    global_runs = initialize_runs(nseed,2);
    mixture_runs = initialize_runs(nseed,4);
    representative = struct();

    optimizer_global = cfg.optimizer;
    optimizer_mixture = cfg.optimizer;

    ls_reduced = cfg.linear_solver;
    ls_reduced.compute_spectrum = false;

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('L-SHAPE: GLOBAL DIAGONAL VS TWO-COMPONENT DISTRIBUTION\n');
    fprintf('seeds=%d, features=%d vs %d+%d, radius split=%.3f\n', ...
        nseed,cfg.global_num_features,cfg.mixture.num_near, ...
        cfg.mixture.num_far,cfg.mixture.radius_split);
    fprintf('============================================================\n');

    for j = 1:nseed
        seed = seeds(j);

        %% Same-budget global diagonal baseline
        method_timer = tic;
        basis_global = build_poisson_lshape_basis( ...
            cfg.global_num_features,cfg.domain,seed, ...
            cfg.geometry_tolerance);
        cache_global = prepare_poisson_cache(problem,basis_global);
        objective_global = @(p) evaluate_poisson_reduced_fast( ...
            p,cache_global,cfg.lambda,ls_reduced);

        optimization_timer = tic;
        [p_global,history_global] = optimize_distribution_adam( ...
            cfg.initialization.p0,objective_global,optimizer_global);
        global_runs.optimization_time(j) = toc(optimization_timer);

        refit_timer = tic;
        [M_global,y] = assemble_poisson_lshape_matrix( ...
            p_global,problem,basis_global,cfg.activation);
        coef_global = solve_least_squares( ...
            M_global,y,cfg.linear_solver);
        global_runs.refit_time(j) = toc(refit_timer);
        residual_global = M_global*coef_global-y;

        eval_global = evaluate_poisson_lshape_approximation( ...
            cfg,p_global,basis_global,coef_global,[],[]);
        eval_global = add_regional_metrics( ...
            eval_global,cfg.mixture.radius_split);
        global_runs = store_run(global_runs,j,p_global,history_global, ...
            residual_global,eval_global,toc(method_timer));

        %% Two-component center-conditioned distribution
        method_timer = tic;
        basis_mixture = build_poisson_lshape_mixture_basis( ...
            cfg.mixture.num_near,cfg.mixture.num_far,cfg.domain,seed, ...
            cfg.mixture.radius_split,cfg.geometry_tolerance);
        cache_mixture = prepare_poisson_lshape_mixture_cache( ...
            problem,basis_mixture);
        objective_mixture = @(p) ...
            evaluate_poisson_lshape_mixture_reduced_fast( ...
                p,cache_mixture,cfg.lambda,ls_reduced);

        optimization_timer = tic;
        [p_mixture,history_mixture] = optimize_distribution_adam( ...
            cfg.mixture.p0,objective_mixture,optimizer_mixture);
        mixture_runs.optimization_time(j) = toc(optimization_timer);

        refit_timer = tic;
        [M_mixture,y] = assemble_poisson_lshape_mixture_matrix( ...
            p_mixture,problem,basis_mixture,cfg.activation);
        coef_mixture = solve_least_squares( ...
            M_mixture,y,cfg.linear_solver);
        mixture_runs.refit_time(j) = toc(refit_timer);
        residual_mixture = M_mixture*coef_mixture-y;

        eval_mixture = evaluate_poisson_lshape_mixture_approximation( ...
            cfg,p_mixture,basis_mixture,coef_mixture);
        mixture_runs = store_run(mixture_runs,j,p_mixture, ...
            history_mixture,residual_mixture,eval_mixture, ...
            toc(method_timer));

        if j == cfg.study.representative_seed_index
            basis_global.W = basis_global.Z.*p_global;
            basis_global.b = -sum(basis_global.C.*basis_global.W,1);
            basis_mixture.near.W = basis_mixture.near.Z.*p_mixture(1:2);
            basis_mixture.near.b = -sum( ...
                basis_mixture.near.C.*basis_mixture.near.W,1);
            basis_mixture.far.W = basis_mixture.far.Z.*p_mixture(3:4);
            basis_mixture.far.b = -sum( ...
                basis_mixture.far.C.*basis_mixture.far.W,1);

            representative.seed = seed;
            representative.global.basis = basis_global;
            representative.global.p = p_global;
            representative.global.coef = coef_global;
            representative.global.evaluation = eval_global;
            representative.mixture.basis = basis_mixture;
            representative.mixture.p = p_mixture;
            representative.mixture.coef = coef_mixture;
            representative.mixture.evaluation = eval_mixture;
        end

        fprintf(['seed %2d/%2d | global L2 %.3e | mixture L2 %.3e ', ...
            '| p_mix=%s\n'],j,nseed,eval_global.relative_l2, ...
            eval_mixture.relative_l2,mat2str(p_mixture.',5));
    end

    summary = build_summary(global_runs,mixture_runs);
    per_seed = build_per_seed_table(seeds,global_runs,mixture_runs);

    results = struct();
    results.cfg = cfg;
    results.project_root = project_root;
    results.problem_counts = [problem.num_interior,problem.num_boundary];
    results.global = global_runs;
    results.mixture = mixture_runs;
    results.summary = summary;
    results.per_seed = per_seed;
    results.representative = representative;

    result_file = fullfile(cfg.output_root, ...
        'poisson_lshape_mixture_study_results.mat');
    summary_file = fullfile(cfg.output_root, ...
        'poisson_lshape_mixture_study_summary.csv');
    per_seed_file = fullfile(cfg.output_root, ...
        'poisson_lshape_mixture_study_all_runs.csv');

    save(result_file,'results','-v7.3');
    writetable(summary,summary_file);
    writetable(per_seed,per_seed_file);

    fprintf('\n');
    disp(summary);
    fprintf('Saved MAT     : %s\n',result_file);
    fprintf('Saved summary : %s\n',summary_file);
    fprintf('Saved details : %s\n',per_seed_file);
end


function runs = initialize_runs(n,np)
    runs.p = nan(np,n);
    runs.selected_checkpoint = nan(n,1);
    runs.relative_l2 = nan(n,1);
    runs.relative_linf = nan(n,1);
    runs.near_relative_l2 = nan(n,1);
    runs.far_relative_l2 = nan(n,1);
    runs.near_absolute_linf = nan(n,1);
    runs.far_absolute_linf = nan(n,1);
    runs.residual_mse = nan(n,1);
    runs.optimization_time = nan(n,1);
    runs.refit_time = nan(n,1);
    runs.total_time = nan(n,1);
end


function runs = store_run(runs,j,p,history,residual,evaluation,total_time)
    runs.p(:,j) = p(:);
    runs.selected_checkpoint(j) = history.best_iteration;
    runs.relative_l2(j) = evaluation.relative_l2;
    runs.relative_linf(j) = evaluation.relative_linf;
    runs.near_relative_l2(j) = evaluation.near.relative_l2;
    runs.far_relative_l2(j) = evaluation.far.relative_l2;
    runs.near_absolute_linf(j) = evaluation.near.absolute_linf;
    runs.far_absolute_linf(j) = evaluation.far.absolute_linf;
    runs.residual_mse(j) = mean(residual.^2);
    runs.total_time(j) = total_time;
end


function evaluation = add_regional_metrics(evaluation,radius_split)
    radius = vecnorm(evaluation.X,2,2);
    near = radius <= radius_split;
    far = radius > radius_split;
    evaluation.near.num_points = nnz(near);
    evaluation.near.relative_l2 = relative_l2( ...
        evaluation.u_pred(near),evaluation.u_exact(near));
    evaluation.near.relative_linf = relative_linf( ...
        evaluation.u_pred(near),evaluation.u_exact(near));
    evaluation.near.absolute_linf = norm( ...
        evaluation.u_pred(near)-evaluation.u_exact(near),inf);
    evaluation.far.num_points = nnz(far);
    evaluation.far.relative_l2 = relative_l2( ...
        evaluation.u_pred(far),evaluation.u_exact(far));
    evaluation.far.relative_linf = relative_linf( ...
        evaluation.u_pred(far),evaluation.u_exact(far));
    evaluation.far.absolute_linf = norm( ...
        evaluation.u_pred(far)-evaluation.u_exact(far),inf);
end


function summary = build_summary(global_runs,mixture_runs)
    Method = ["GlobalDiagonal";"TwoComponentMixture"];
    ComponentFeatures = ["(1100,0)";"(550,550)"];
    PxNear = [global_runs.p(1);mixture_runs.p(1)];
    PyNear = [global_runs.p(2);mixture_runs.p(2)];
    PxFar = [NaN;mixture_runs.p(3)];
    PyFar = [NaN;mixture_runs.p(4)];
    TimeSec = [global_runs.total_time;mixture_runs.total_time];
    OverallRelativeL2 = [global_runs.relative_l2; ...
        mixture_runs.relative_l2];
    HighGradientRelativeL2 = [global_runs.near_relative_l2; ...
        mixture_runs.near_relative_l2];
    HighGradientLinf = [global_runs.near_absolute_linf; ...
        mixture_runs.near_absolute_linf];
    LowGradientRelativeL2 = [global_runs.far_relative_l2; ...
        mixture_runs.far_relative_l2];
    LowGradientLinf = [global_runs.far_absolute_linf; ...
        mixture_runs.far_absolute_linf];

    summary = table(Method,ComponentFeatures,PxNear,PyNear,PxFar,PyFar, ...
        TimeSec,OverallRelativeL2,HighGradientRelativeL2, ...
        HighGradientLinf,LowGradientRelativeL2,LowGradientLinf);
end


function T = build_per_seed_table(seeds,g,m)
    Seed = seeds;
    GlobalL2 = g.relative_l2;
    GlobalLinf = g.relative_linf;
    GlobalNearL2 = g.near_relative_l2;
    GlobalFarL2 = g.far_relative_l2;
    GlobalP1 = g.p(1,:).';
    GlobalP2 = g.p(2,:).';
    MixtureL2 = m.relative_l2;
    MixtureLinf = m.relative_linf;
    MixtureNearL2 = m.near_relative_l2;
    MixtureFarL2 = m.far_relative_l2;
    GlobalNearLinf = g.near_absolute_linf;
    GlobalFarLinf = g.far_absolute_linf;
    MixtureNearLinf = m.near_absolute_linf;
    MixtureFarLinf = m.far_absolute_linf;
    NearPx = m.p(1,:).';
    NearPy = m.p(2,:).';
    FarPx = m.p(3,:).';
    FarPy = m.p(4,:).';
    GlobalTime = g.total_time;
    MixtureTime = m.total_time;

    T = table(Seed,GlobalL2,GlobalLinf,GlobalNearL2,GlobalFarL2, ...
        GlobalNearLinf,GlobalFarLinf,GlobalP1,GlobalP2,MixtureL2, ...
        MixtureLinf,MixtureNearL2,MixtureFarL2,MixtureNearLinf, ...
        MixtureFarLinf,NearPx,NearPy,FarPx,FarPy,GlobalTime,MixtureTime);
end
