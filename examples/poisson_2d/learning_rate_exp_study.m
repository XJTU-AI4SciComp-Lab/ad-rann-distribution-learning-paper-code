clear;
clc;
close all;

warning('off','MATLAB:rankDeficientMatrix');

%% ========================================================================
%  AD-RaNN log/exp learning-rate sensitivity study
%
%  Put this file directly under:
%
%      examples/poisson_2d/
%
%  Then run:
%
%      learning_rate_exp_study
%
%  This experiment uses p=exp(s) and changes ONLY the Adam learning rate
%  in s-space.
%
%  Common settings:
%      p0             = (4,8)
%      m              = 600
%      lambda         = 1e-5
%      seeds          = 1:100
%      Adam updates   = 50
%      beta1          = 0.9
%      beta2          = 0.999
%      Adam epsilon   = 1e-8
%      activation     = Gaussian
%
%  Learning rates:
%      0.1, 0.3, 1, 3
%
%  IMPORTANT:
%  For the same seed, all learning rates use exactly the same randomized
%  feature realization, PDE data, and reduced objective/cache.
% =========================================================================


%% ========================================================================
%  Path
% =========================================================================

this_file = mfilename('fullpath');

if isempty(this_file)
    error('Please save this file as learning_rate_exp_study.m before running it.');
end

example_dir = fileparts(this_file);
root = fileparts(fileparts(example_dir));

addpath(genpath(fullfile(root,'src')));
addpath(example_dir);


%% ========================================================================
%  Base configuration
% =========================================================================

cfg_base = config();

% Force the protocol used in this sensitivity experiment.
cfg_base.num_features = 600;
cfg_base.lambda = 1e-5;

cfg_base.optimizer.maxit = 50;
cfg_base.optimizer.beta1 = 0.9;
cfg_base.optimizer.beta2 = 0.999;
cfg_base.optimizer.epsilon = 1e-8;
cfg_base.optimizer.parameterization = 'log';

cfg_base.optimizer.verbose = false;
cfg_base.optimizer.store_moments = true;
cfg_base.optimizer.store_full_info = false;

% Fixed 50 Adam updates: disable early stopping.
cfg_base.optimizer.grad_tol = 0;
cfg_base.optimizer.step_tol = 0;
cfg_base.optimizer.relative_obj_tol = 0;
cfg_base.optimizer.patience = Inf;
cfg_base.optimizer.min_delta = 0;

% No SVD during optimization.
cfg_base.compute_spectrum = false;

% Keep the parameter bounds from your current config.m.
% If you want to force a particular upper bound, uncomment ONE of:
%
% cfg_base.optimizer.lower_bound = [1e-3;1e-3];
% cfg_base.optimizer.upper_bound = [100;100];
%
% or
%
% cfg_base.optimizer.lower_bound = [1e-3;1e-3];
% cfg_base.optimizer.upper_bound = [300;300];

% Optional: select MATLAB GPU here if desired.
% cfg_base.linear_solver.gpu_id = 4;


%% ========================================================================
%  Learning rates / seeds / initialization
% =========================================================================

learning_rates = [0.1, 0.5, 1.0, 2.0];
num_lr = numel(learning_rates);

seeds = 1:100;
num_seeds = numel(seeds);

p0 = [4;8];


%% ========================================================================
%  Output
% =========================================================================

output_dir = fullfile(example_dir,'learning_rate_exp_study_results');

if exist(output_dir,'dir') ~= 7
    mkdir(output_dir);
end

checkpoint_file = fullfile( ...
    output_dir, ...
    'learning_rate_exp_study_checkpoint.mat');

result_mat_file = fullfile( ...
    output_dir, ...
    'learning_rate_exp_study_results.mat');

summary_csv_file = fullfile( ...
    output_dir, ...
    'learning_rate_exp_study_summary.csv');

detail_csv_file = fullfile( ...
    output_dir, ...
    'learning_rate_exp_study_all_runs.csv');


%% ========================================================================
%  Test grid
% =========================================================================

Xtest = tensor_grid( ...
    cfg_base.domain, ...
    cfg_base.test_grid, ...
    0);

ref_test = exact_solution(Xtest);


%% ========================================================================
%  Allocate
%
%  row    = learning rate
%  column = seed
% =========================================================================

p1_all = nan(num_lr,num_seeds);
p2_all = nan(num_lr,num_seeds);

best_iteration_all = nan(num_lr,num_seeds);
best_mse_all = nan(num_lr,num_seeds);

rel_l2_all = nan(num_lr,num_seeds);
rel_linf_all = nan(num_lr,num_seeds);

optimization_time_all = nan(num_lr,num_seeds);
refit_test_time_all = nan(num_lr,num_seeds);

success_all = false(num_lr,num_seeds);
completed_all = false(num_lr,num_seeds);

error_message = cell(num_lr,num_seeds);


%% ========================================================================
%  Resume checkpoint
% =========================================================================

if exist(checkpoint_file,'file') == 2

    try

        S = load(checkpoint_file,'state');

        if isfield(S,'state')

            state = S.state;

            compatible = ...
                isfield(state,'learning_rates') && ...
                isfield(state,'seeds') && ...
                isfield(state,'parameterization') && ...
                isequal(state.learning_rates(:),learning_rates(:)) && ...
                isequal(state.seeds(:),seeds(:)) && ...
                strcmp(state.parameterization,'log');

            if compatible

                p1_all = state.p1_all;
                p2_all = state.p2_all;

                best_iteration_all = state.best_iteration_all;
                best_mse_all = state.best_mse_all;

                rel_l2_all = state.rel_l2_all;
                rel_linf_all = state.rel_linf_all;

                optimization_time_all = state.optimization_time_all;
                refit_test_time_all = state.refit_test_time_all;

                success_all = state.success_all;
                completed_all = state.completed_all;

                error_message = state.error_message;

                fprintf('\nCheckpoint loaded: %d / %d runs completed.\n', ...
                    nnz(completed_all),num_lr*num_seeds);
            end
        end

    catch ME

        warning( ...
            'learning_rate_exp_study:CheckpointLoad', ...
            'Checkpoint could not be loaded: %s',ME.message);

    end
end


%% ========================================================================
%  Header
% =========================================================================

fprintf('\n');
fprintf('======================================================================\n');
fprintf('          AD-RaNN log/exp learning-rate sensitivity study\n');
fprintf('======================================================================\n');
fprintf('p0             = [%.3f, %.3f]\n',p0(1),p0(2));
fprintf('Features       = %d\n',cfg_base.num_features);
fprintf('lambda         = %.3e\n',cfg_base.lambda);
fprintf('Seeds          = 1:%d\n',num_seeds);
fprintf('Adam updates   = %d\n',cfg_base.optimizer.maxit);
fprintf('beta1          = %.3f\n',cfg_base.optimizer.beta1);
fprintf('beta2          = %.3f\n',cfg_base.optimizer.beta2);
fprintf('Adam epsilon   = %.3e\n',cfg_base.optimizer.epsilon);
fprintf('Learning rates = %s\n',mat2str(learning_rates));
fprintf('Parameterization = p=exp(s), learning rate acts in s-space\n');
fprintf('======================================================================\n\n');


%% ========================================================================
%  Main loop
%
%  Seed is the outer loop so all learning rates share exactly the same:
%      - randomized basis
%      - PDE system
%      - cache / reduced objective
% =========================================================================

for i = 1:num_seeds

    seed_i = seeds(i);

    % Skip expensive setup if all learning rates for this seed are done.
    if all(completed_all(:,i))

        fprintf('[seed %3d/%3d] all learning rates completed -> skipped\n', ...
            seed_i,num_seeds);

        continue;
    end


    %% --------------------------------------------------------------------
    %  Configuration for this seed
    % ---------------------------------------------------------------------

    cfg_seed = cfg_base;
    cfg_seed.seed = seed_i;


    %% --------------------------------------------------------------------
    %  Random basis
    % ---------------------------------------------------------------------

    basis = build_random_weights( ...
        cfg_seed.num_features, ...
        cfg_seed.domain, ...
        cfg_seed.seed);


    %% --------------------------------------------------------------------
    %  PDE data
    % ---------------------------------------------------------------------

    problem.domain = cfg_seed.domain;
    problem.boundary_penalty = cfg_seed.boundary_penalty;

    problem.Xi = tensor_grid( ...
        cfg_seed.domain, ...
        cfg_seed.interior_grid, ...
        1e-6);

    problem.fi = rhs(problem.Xi);


    nB = cfg_seed.boundary_points_per_side;

    x = linspace( ...
        cfg_seed.domain(1,1), ...
        cfg_seed.domain(1,2), ...
        nB)';

    y = linspace( ...
        cfg_seed.domain(2,1), ...
        cfg_seed.domain(2,2), ...
        nB)';


    problem.Xb = [ ...
        cfg_seed.domain(1,1)*ones(nB,1), y; ...
        cfg_seed.domain(1,2)*ones(nB,1), y; ...
        x, cfg_seed.domain(2,1)*ones(nB,1); ...
        x, cfg_seed.domain(2,2)*ones(nB,1)];


    problem.gb = exact_solution(problem.Xb);


    problem.y = [ ...
        problem.fi; ...
        cfg_seed.boundary_penalty*problem.gb];


    %% --------------------------------------------------------------------
    %  Least-squares options
    % ---------------------------------------------------------------------

    ls_opts = cfg_seed.linear_solver;
    ls_opts.compute_spectrum = cfg_seed.compute_spectrum;


    %% --------------------------------------------------------------------
    %  Shared reduced objective
    % ---------------------------------------------------------------------

    if cfg_seed.use_fast_evaluator

        cache = prepare_poisson_cache( ...
            problem, ...
            basis);

        objective_fun = @(p) ...
            evaluate_poisson_reduced_fast( ...
                p, ...
                cache, ...
                cfg_seed.lambda, ...
                ls_opts);

    else

        objective_fun = @(p) ...
            poisson_objective_learning_rate( ...
                p, ...
                problem, ...
                basis, ...
                cfg_seed.lambda, ...
                ls_opts);

    end


    %% --------------------------------------------------------------------
    %  Learning-rate loop
    % ---------------------------------------------------------------------

    for j = 1:num_lr

        lr = learning_rates(j);

        if completed_all(j,i)

            fprintf( ...
                '[lr=%4.1f | seed %3d/%3d] completed -> skipped\n', ...
                lr,seed_i,num_seeds);

            continue;
        end


        fprintf( ...
            '[lr=%4.1f | seed %3d/%3d] ... ', ...
            lr,seed_i,num_seeds);


        try

            cfg_run = cfg_seed;

            % The ONLY changed algorithmic parameter.
            cfg_run.optimizer.learning_rate = lr;


            %% ------------------------------------------------------------
            %  Adam optimization
            % -------------------------------------------------------------

            optimization_timer = tic;

            [p_opt,history] = ...
                optimize_distribution_adam( ...
                    p0, ...
                    objective_fun, ...
                    cfg_run.optimizer);

            optimization_time = toc(optimization_timer);


            %% ------------------------------------------------------------
            %  Final UNREGULARIZED least-squares refit + test
            % -------------------------------------------------------------

            refit_timer = tic;

            [M,y_rhs,~] = ...
                build_system( ...
                    p_opt, ...
                    problem, ...
                    basis);

            [coef,~] = ...
                solve_least_squares( ...
                    M, ...
                    y_rhs, ...
                    cfg_run.linear_solver);


            Phi_test = gaussian_features( ...
                Xtest, ...
                p_opt, ...
                basis);

            pred = Phi_test*coef;

            err_l2 = relative_l2( ...
                pred, ...
                ref_test);

            err_linf = relative_linf( ...
                pred, ...
                ref_test);

            refit_test_time = toc(refit_timer);


            %% ------------------------------------------------------------
            %  Store
            % -------------------------------------------------------------

            p1_all(j,i) = p_opt(1);
            p2_all(j,i) = p_opt(2);

            best_iteration_all(j,i) = ...
                history.best_iteration;

            best_mse_all(j,i) = ...
                history.best_selection_value;

            rel_l2_all(j,i) = err_l2;
            rel_linf_all(j,i) = err_linf;

            optimization_time_all(j,i) = ...
                optimization_time;

            refit_test_time_all(j,i) = ...
                refit_test_time;

            success_all(j,i) = true;
            completed_all(j,i) = true;


            fprintf( ...
                ['p*=[%.6f, %.6f] | ckpt=%d | ', ...
                 'MSE=%.3e | L2=%.3e | Linf=%.3e | opt=%.3f s\n'], ...
                p_opt(1), ...
                p_opt(2), ...
                history.best_iteration, ...
                history.best_selection_value, ...
                err_l2, ...
                err_linf, ...
                optimization_time);


        catch ME

            success_all(j,i) = false;
            completed_all(j,i) = true;

            error_message{j,i} = getReport( ...
                ME, ...
                'extended', ...
                'hyperlinks', ...
                'off');

            fprintf('FAILED\n');
            fprintf('%s\n',error_message{j,i});

        end


        %% ----------------------------------------------------------------
        %  Checkpoint after every completed (learning rate, seed) run
        % -----------------------------------------------------------------

        state = struct();

        state.learning_rates = learning_rates;
        state.seeds = seeds;
        state.parameterization = 'log';

        state.p1_all = p1_all;
        state.p2_all = p2_all;

        state.best_iteration_all = best_iteration_all;
        state.best_mse_all = best_mse_all;

        state.rel_l2_all = rel_l2_all;
        state.rel_linf_all = rel_linf_all;

        state.optimization_time_all = optimization_time_all;
        state.refit_test_time_all = refit_test_time_all;

        state.success_all = success_all;
        state.completed_all = completed_all;

        state.error_message = error_message;

        save(checkpoint_file,'state','-v7.3');

    end

    clear basis problem cache objective_fun;

end


%% ========================================================================
%  Summary statistics
% =========================================================================

num_success = zeros(num_lr,1);

L2_mean = nan(num_lr,1);
L2_std = nan(num_lr,1);
L2_median = nan(num_lr,1);
L2_best = nan(num_lr,1);
L2_worst = nan(num_lr,1);

Linf_mean = nan(num_lr,1);
Linf_std = nan(num_lr,1);

p1_mean = nan(num_lr,1);
p1_std = nan(num_lr,1);

p2_mean = nan(num_lr,1);
p2_std = nan(num_lr,1);

selected_checkpoint_mean = nan(num_lr,1);
selected_checkpoint_std = nan(num_lr,1);

optimization_mean = nan(num_lr,1);
optimization_std = nan(num_lr,1);

refit_test_mean = nan(num_lr,1);


for j = 1:num_lr

    valid = success_all(j,:);

    num_success(j) = nnz(valid);

    if ~any(valid)
        continue;
    end

    values_l2 = rel_l2_all(j,valid);

    L2_mean(j) = mean(values_l2);
    L2_std(j) = std(values_l2);
    L2_median(j) = median(values_l2);
    L2_best(j) = min(values_l2);
    L2_worst(j) = max(values_l2);

    Linf_mean(j) = mean(rel_linf_all(j,valid));
    Linf_std(j) = std(rel_linf_all(j,valid));

    p1_mean(j) = mean(p1_all(j,valid));
    p1_std(j) = std(p1_all(j,valid));

    p2_mean(j) = mean(p2_all(j,valid));
    p2_std(j) = std(p2_all(j,valid));

    selected_checkpoint_mean(j) = ...
        mean(best_iteration_all(j,valid));

    selected_checkpoint_std(j) = ...
        std(best_iteration_all(j,valid));

    optimization_mean(j) = ...
        mean(optimization_time_all(j,valid));

    optimization_std(j) = ...
        std(optimization_time_all(j,valid));

    refit_test_mean(j) = ...
        mean(refit_test_time_all(j,valid));

end


LearningRate = learning_rates(:);
Parameterization = repmat("log-exp",num_lr,1);

SummaryTable = table( ...
    Parameterization, ...
    LearningRate, ...
    num_success, ...
    L2_mean, ...
    L2_std, ...
    L2_median, ...
    L2_best, ...
    L2_worst, ...
    Linf_mean, ...
    Linf_std, ...
    p1_mean, ...
    p1_std, ...
    p2_mean, ...
    p2_std, ...
    selected_checkpoint_mean, ...
    selected_checkpoint_std, ...
    optimization_mean, ...
    optimization_std, ...
    refit_test_mean);


%% ========================================================================
%  Detailed table
% =========================================================================

nrows = num_lr*num_seeds;

detail_lr = nan(nrows,1);
detail_seed = nan(nrows,1);

detail_p1 = nan(nrows,1);
detail_p2 = nan(nrows,1);

detail_ckpt = nan(nrows,1);
detail_mse = nan(nrows,1);

detail_l2 = nan(nrows,1);
detail_linf = nan(nrows,1);

detail_opt_time = nan(nrows,1);
detail_refit_time = nan(nrows,1);

detail_success = false(nrows,1);

row = 0;

for j = 1:num_lr

    for i = 1:num_seeds

        row = row+1;

        detail_lr(row) = learning_rates(j);
        detail_seed(row) = seeds(i);

        detail_p1(row) = p1_all(j,i);
        detail_p2(row) = p2_all(j,i);

        detail_ckpt(row) = best_iteration_all(j,i);
        detail_mse(row) = best_mse_all(j,i);

        detail_l2(row) = rel_l2_all(j,i);
        detail_linf(row) = rel_linf_all(j,i);

        detail_opt_time(row) = optimization_time_all(j,i);
        detail_refit_time(row) = refit_test_time_all(j,i);

        detail_success(row) = success_all(j,i);

    end
end


AllRunsTable = table( ...
    detail_lr, ...
    detail_seed, ...
    detail_p1, ...
    detail_p2, ...
    detail_ckpt, ...
    detail_mse, ...
    detail_l2, ...
    detail_linf, ...
    detail_opt_time, ...
    detail_refit_time, ...
    detail_success, ...
    'VariableNames',{ ...
        'LearningRate', ...
        'Seed', ...
        'p1_opt', ...
        'p2_opt', ...
        'SelectedCheckpoint', ...
        'BestSelectionMSE', ...
        'RelativeL2', ...
        'RelativeLinf', ...
        'OptimizationTimeSec', ...
        'RefitTestTimeSec', ...
        'Success'});


%% ========================================================================
%  Print summary
% =========================================================================

fprintf('\n');
fprintf('=================================================================================================================\n');
fprintf('LEARNING-RATE FINAL SUMMARY\n');
fprintf('Parameterization: p=exp(s); lr is the s-space learning rate.\n');
fprintf('=================================================================================================================\n');
fprintf(' lr      success    mean L2       std L2        best L2       worst L2      mean p*             mean ckpt   opt/s\n');
fprintf('=================================================================================================================\n');

for j = 1:num_lr

    fprintf( ...
        '%4.1f     %3d/%3d    %.3e    %.3e    %.3e    %.3e    [%6.3f,%6.3f]      %6.2f     %.3f\n', ...
        learning_rates(j), ...
        num_success(j), ...
        num_seeds, ...
        L2_mean(j), ...
        L2_std(j), ...
        L2_best(j), ...
        L2_worst(j), ...
        p1_mean(j), ...
        p2_mean(j), ...
        selected_checkpoint_mean(j), ...
        optimization_mean(j));

end

fprintf('=================================================================================================================\n\n');

disp(SummaryTable);


%% ========================================================================
%  Save
% =========================================================================

results = struct();

results.cfg_base = cfg_base;

results.learning_rates = learning_rates;
results.seeds = seeds;
results.p0 = p0;
results.parameterization = 'log';

results.p1_all = p1_all;
results.p2_all = p2_all;

results.best_iteration_all = best_iteration_all;
results.best_mse_all = best_mse_all;

results.rel_l2_all = rel_l2_all;
results.rel_linf_all = rel_linf_all;

results.optimization_time_all = optimization_time_all;
results.refit_test_time_all = refit_test_time_all;

results.success_all = success_all;
results.completed_all = completed_all;

results.error_message = error_message;

results.SummaryTable = SummaryTable;
results.AllRunsTable = AllRunsTable;

save(result_mat_file,'results','-v7.3');

writetable(SummaryTable,summary_csv_file);
writetable(AllRunsTable,detail_csv_file);

fprintf('Saved MAT     : %s\n',result_mat_file);
fprintf('Saved summary : %s\n',summary_csv_file);
fprintf('Saved details : %s\n',detail_csv_file);


%% ========================================================================
%  Optional simple figure
% =========================================================================

fig = figure;

errorbar( ...
    learning_rates, ...
    L2_mean, ...
    L2_std, ...
    'o-', ...
    'LineWidth',1.2, ...
    'MarkerSize',6);

set(gca,'XScale','log','YScale','log');

box on;
grid off;

xlabel('Adam learning rate','FontSize',14);
ylabel('Mean relative \ell_2 error','FontSize',14);
title('Learning-rate sensitivity','FontSize',12);

set(gca, ...
    'FontSize',12, ...
    'LineWidth',1.0);

saveas(fig,fullfile(output_dir,'learning_rate_sensitivity.fig'));


%% ========================================================================
%  Generic objective fallback
% =========================================================================

function [obj,grad,info] = ...
    poisson_objective_learning_rate( ...
        p,problem,basis,lambda,ls_opts)

    [M,y,dM] = ...
        build_system( ...
            p, ...
            problem, ...
            basis);

    [obj,grad,info] = ...
        reduced_objective_gradient( ...
            M, ...
            y, ...
            dM, ...
            lambda, ...
            ls_opts);

end
