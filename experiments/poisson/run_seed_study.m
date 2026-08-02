clear; clc;

this_file = mfilename('fullpath');
exp_dir = fileparts(this_file);
root = fileparts(fileparts(exp_dir));
example_dir = fullfile(root,'examples','poisson_2d');

addpath(genpath(fullfile(root,'src')));
addpath(example_dir);

seeds = [2026 2027 2028 2029 2030];

cfg0 = config();

% Frequency initialization is independent of the random feature seed.
[p0,~] = frequency_initialization(cfg0);

rows = zeros(numel(seeds),7);

for i = 1:numel(seeds)

    cfg = config();

    cfg.seed = seeds(i);
    cfg.optimizer.verbose = false;

    basis = build_random_weights( ...
        cfg.num_features,cfg.domain,cfg.seed);

    problem.domain = cfg.domain;
    problem.boundary_penalty = cfg.boundary_penalty;

    problem.Xi = tensor_grid( ...
        cfg.domain,cfg.interior_grid,1e-6);

    problem.fi = rhs(problem.Xi);

    nB = cfg.boundary_points_per_side;

    x = linspace(cfg.domain(1,1),cfg.domain(1,2),nB)';
    y = linspace(cfg.domain(2,1),cfg.domain(2,2),nB)';

    problem.Xb = [ ...
        cfg.domain(1,1)*ones(nB,1), y; ...
        cfg.domain(1,2)*ones(nB,1), y; ...
        x, cfg.domain(2,1)*ones(nB,1); ...
        x, cfg.domain(2,2)*ones(nB,1)];

    problem.gb = exact_solution(problem.Xb);

    problem.y = [ ...
        problem.fi; ...
        cfg.boundary_penalty*problem.gb];

    cache = prepare_poisson_cache(problem,basis);

    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    objective_fun = @(p) ...
        evaluate_poisson_reduced_fast( ...
            p,cache,cfg.lambda,ls_opts);

    timer_all = tic;

    [p_opt,hist] = ...
        optimize_distribution_adam( ...
            p0,objective_fun,cfg.optimizer);

    [M,b,~] = build_system(p_opt,problem,basis);

    [coef,~] = ...
        solve_least_squares(M,b,cfg.linear_solver);

    Xtest = tensor_grid(cfg.domain,cfg.test_grid,0);

    pred = gaussian_features(Xtest,p_opt,basis)*coef;
    ref = exact_solution(Xtest);

    elapsed = toc(timer_all);

    e2 = relative_l2(pred,ref);
    ei = relative_linf(pred,ref);

    rows(i,:) = [ ...
        seeds(i), ...
        p_opt(1), ...
        p_opt(2), ...
        hist.best_selection_value, ...
        e2, ...
        ei, ...
        elapsed];

    fprintf( ...
        'seed=%d | L2=%.3e | Linf=%.3e | p=[%.3f %.3f]\n', ...
        seeds(i),e2,ei,p_opt(1),p_opt(2));
end

T = array2table( ...
    rows, ...
    'VariableNames', ...
    {'Seed','p1','p2','BestTrainingMSE', ...
     'RelativeL2','RelativeLinf','TimeSec'});

out_dir = fullfile(root,'results','poisson');

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

writetable(T,fullfile(out_dir,'seed_study.csv'));

fprintf('\nMean L2  = %.6e\n',mean(T.RelativeL2));
fprintf('Std L2   = %.6e\n',std(T.RelativeL2));
fprintf('Best L2  = %.6e\n',min(T.RelativeL2));
fprintf('Worst L2 = %.6e\n',max(T.RelativeL2));
