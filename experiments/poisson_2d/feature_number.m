clear;
clc;

warning('off','MATLAB:rankDeficientMatrix');

%% ========================================================================
%  Paths and base configuration
% =========================================================================

this_file = mfilename('fullpath');
study_dir = fileparts(this_file);
root = fileparts(fileparts(study_dir));
example_dir = fullfile(root,'examples','poisson_2d');

addpath(genpath(fullfile(root,'src')));
addpath(example_dir);
addpath(study_dir);

cfg_base = config();
cfg_base.lambda = 1e-7;
cfg_base.optimizer.verbose = false;
cfg_base.optimizer.store_moments = true;
cfg_base.optimizer.store_full_info = false;
cfg_base.compute_spectrum = false;

p0 = [4;8];

% Feature dimensions used for distribution learning.
m_list = [200;400;600;800;1000];
m_expand = 1000;

% The original grid comes from config.m: [30,80].
base_interior_grid = cfg_base.interior_grid;

% Requested collocation expansion.
expanded_interior_grid = [45,120];

seeds = 1:100;
num_m = numel(m_list);
num_seeds = numel(seeds);

base_num_interior = prod(base_interior_grid);
expanded_num_interior = prod(expanded_interior_grid);

%% ========================================================================
%  GPU
% =========================================================================

if cfg_base.linear_solver.use_gpu

    gpu_id = cfg_base.linear_solver.gpu_id;

    if gpu_id > gpuDeviceCount
        error('GPU %d is not available. MATLAB sees only %d GPU(s).', ...
            gpu_id,gpuDeviceCount);
    end

    g = gpuDevice(gpu_id);
    fprintf('\nUsing GPU %d: %s\n',g.Index,g.Name);
    fprintf('Available memory: %.2f GB\n\n', ...
        g.AvailableMemory/1024^3);
end

%% ========================================================================
%  Shared test grid
% =========================================================================

Xtest = tensor_grid(cfg_base.domain,cfg_base.test_grid,0);
ref_test = exact_solution(Xtest);

%% ========================================================================
%  Result storage
%
%  Scenario 0: original m and original 30x80 grid, including Adam.
%  Scenario 1: only m -> 1000, fixed original grid and fixed p*.
%  Scenario 2: only grid -> 45x120, fixed original m and fixed p*.
%  Scenario 3: m -> 1000 and grid -> 45x120, fixed p*.
% =========================================================================

p1_opt_all = nan(num_m,num_seeds);
p2_opt_all = nan(num_m,num_seeds);
selected_checkpoint_all = nan(num_m,num_seeds);
best_mse_all = nan(num_m,num_seeds);

rel_l2_original_all = nan(num_m,num_seeds);
rel_linf_original_all = nan(num_m,num_seeds);

rel_l2_m1000_all = nan(num_m,num_seeds);
rel_linf_m1000_all = nan(num_m,num_seeds);

rel_l2_grid_all = nan(num_m,num_seeds);
rel_linf_grid_all = nan(num_m,num_seeds);

rel_l2_both_all = nan(num_m,num_seeds);
rel_linf_both_all = nan(num_m,num_seeds);

% Original AD-RaNN time. This includes only the original path.
original_basis_time_all = nan(num_m,num_seeds);
original_pde_time_all = nan(num_m,num_seeds);
original_cache_time_all = nan(num_m,num_seeds);
original_optimization_time_all = nan(num_m,num_seeds);
original_assembly_time_all = nan(num_m,num_seeds);
original_ls_time_all = nan(num_m,num_seeds);
original_test_time_all = nan(num_m,num_seeds);
original_total_time_all = nan(num_m,num_seeds);

% Only m -> 1000. The original run and other expansions are excluded.
m1000_basis_time_all = nan(num_m,num_seeds);
m1000_assembly_time_all = nan(num_m,num_seeds);
m1000_ls_time_all = nan(num_m,num_seeds);
m1000_test_time_all = nan(num_m,num_seeds);
m1000_total_time_all = nan(num_m,num_seeds);

% Only grid -> 45x120. The existing m-feature basis is reused.
grid_pde_time_all = nan(num_m,num_seeds);
grid_assembly_time_all = nan(num_m,num_seeds);
grid_ls_time_all = nan(num_m,num_seeds);
grid_test_time_all = nan(num_m,num_seeds);
grid_total_time_all = nan(num_m,num_seeds);

% Simultaneous m -> 1000 and grid -> 45x120.
% This block independently rebuilds both the basis and PDE data.
both_basis_time_all = nan(num_m,num_seeds);
both_pde_time_all = nan(num_m,num_seeds);
both_assembly_time_all = nan(num_m,num_seeds);
both_ls_time_all = nan(num_m,num_seeds);
both_test_time_all = nan(num_m,num_seeds);
both_total_time_all = nan(num_m,num_seeds);

success_original_all = false(num_m,num_seeds);
success_m1000_all = false(num_m,num_seeds);
success_grid_all = false(num_m,num_seeds);
success_both_all = false(num_m,num_seeds);

error_original_all = cell(num_m,num_seeds);
error_m1000_all = cell(num_m,num_seeds);
error_grid_all = cell(num_m,num_seeds);
error_both_all = cell(num_m,num_seeds);

%% ========================================================================
%  Output
% =========================================================================

result_dir = fullfile( ...
    root,'results','poisson_2d','feature_number_study');

if exist(result_dir,'dir') ~= 7
    mkdir(result_dir);
end

fprintf('\n');
fprintf('====================================================================\n');
fprintf(' AD-RaNN feature-number and collocation-expansion experiment\n');
fprintf('====================================================================\n');
fprintf('Training feature numbers     = ');
fprintf('%d ',m_list);
fprintf('\n');
fprintf('Expanded feature number      = %d\n',m_expand);
fprintf('Original interior grid       = %dx%d (%d points)\n', ...
    base_interior_grid(1),base_interior_grid(2),base_num_interior);
fprintf('Expanded interior grid       = %dx%d (%d points)\n', ...
    expanded_interior_grid(1),expanded_interior_grid(2), ...
    expanded_num_interior);
fprintf('Seeds                        = 1:%d\n',num_seeds);
fprintf('Initial p                    = [%.6f, %.6f]\n',p0);
fprintf('Ridge lambda during Adam     = %.3e\n',cfg_base.lambda);
fprintf('\nEach successful original run is followed by:\n');
fprintf('  A. only m -> %d;\n',m_expand);
fprintf('  B. only grid -> %dx%d;\n', ...
    expanded_interior_grid(1),expanded_interior_grid(2));
fprintf('  C. both expansions simultaneously.\n');
fprintf(['All three expansion paths fix the same learned p* and perform ', ...
    'no Adam updates.\n']);
fprintf('Each expansion time excludes the original and the other paths.\n');
fprintf('====================================================================\n\n');

%% ========================================================================
%  Main experiment
% =========================================================================

for j = 1:num_m

    m = m_list(j);

    fprintf('\n');
    fprintf('####################################################################\n');
    fprintf('Training feature number %d / %d: m = %d\n',j,num_m,m);
    fprintf('####################################################################\n');

    for i = 1:num_seeds

        seed_i = seeds(i);

        fprintf('[m=%4d | seed %3d/%3d] ', ...
            m,seed_i,num_seeds);

        cfg = cfg_base;
        cfg.seed = seed_i;
        cfg.num_features = m;

        ls_opts = cfg.linear_solver;
        ls_opts.compute_spectrum = cfg.compute_spectrum;

        %% ----------------------------------------------------------------
        %  Scenario 0: original AD-RaNN
        % -----------------------------------------------------------------

        original_timer = tic;

        try
            t = tic;
            basis = build_random_weights( ...
                m,cfg.domain,seed_i);
            original_basis_time_all(j,i) = toc(t);

            t = tic;
            problem = build_poisson_problem( ...
                cfg,base_interior_grid);
            original_pde_time_all(j,i) = toc(t);

            t = tic;

            if cfg.use_fast_evaluator
                cache = prepare_poisson_cache(problem,basis);
                objective_fun = @(p) evaluate_poisson_reduced_fast( ...
                    p,cache,cfg.lambda,ls_opts);
            else
                objective_fun = @(p) poisson_objective_feature_seed( ...
                    p,problem,basis,cfg.lambda,ls_opts);
            end

            original_cache_time_all(j,i) = toc(t);

            t = tic;
            [p_opt,history] = optimize_distribution_adam( ...
                p0,objective_fun,cfg.optimizer);
            original_optimization_time_all(j,i) = toc(t);

            t = tic;
            [M,y_rhs,~] = build_system(p_opt,problem,basis);
            original_assembly_time_all(j,i) = toc(t);

            t = tic;
            coef = solve_least_squares(M,y_rhs,cfg.linear_solver);
            original_ls_time_all(j,i) = toc(t);

            t = tic;
            [err_l2,err_linf] = evaluate_refit( ...
                Xtest,ref_test,p_opt,basis,coef);
            original_test_time_all(j,i) = toc(t);

            p1_opt_all(j,i) = p_opt(1);
            p2_opt_all(j,i) = p_opt(2);
            selected_checkpoint_all(j,i) = history.best_iteration;
            best_mse_all(j,i) = history.best_selection_value;

            rel_l2_original_all(j,i) = err_l2;
            rel_linf_original_all(j,i) = err_linf;

            original_total_time_all(j,i) = toc(original_timer);
            success_original_all(j,i) = true;

            fprintf(['p*=[%.5f %.5f] | base L2=%.3e ', ...
                '| base time=%.3f s'], ...
                p_opt(1),p_opt(2),err_l2, ...
                original_total_time_all(j,i));

            clear M y_rhs coef cache objective_fun

        catch ME
            original_total_time_all(j,i) = toc(original_timer);
            error_original_all{j,i} = getReport( ...
                ME,'extended','hyperlinks','off');
            fprintf('ORIGINAL FAILED\n%s\n', ...
                error_original_all{j,i});
            continue;
        end

        %% ----------------------------------------------------------------
        %  Scenario 1: only m -> 1000
        %
        %  Reuse the original PDE points. The timer includes only:
        %  new 1000-feature basis + assembly + LS + test.
        % -----------------------------------------------------------------

        scenario_timer = tic;

        try
            t = tic;
            basis_m1000 = build_random_weights( ...
                m_expand,cfg.domain,seed_i);
            m1000_basis_time_all(j,i) = toc(t);

            t = tic;
            [M_m1000,y_m1000,~] = build_system( ...
                p_opt,problem,basis_m1000);
            m1000_assembly_time_all(j,i) = toc(t);

            t = tic;
            coef_m1000 = solve_least_squares( ...
                M_m1000,y_m1000,cfg.linear_solver);
            m1000_ls_time_all(j,i) = toc(t);

            t = tic;
            [err_l2_m1000,err_linf_m1000] = evaluate_refit( ...
                Xtest,ref_test,p_opt,basis_m1000,coef_m1000);
            m1000_test_time_all(j,i) = toc(t);

            rel_l2_m1000_all(j,i) = err_l2_m1000;
            rel_linf_m1000_all(j,i) = err_linf_m1000;
            m1000_total_time_all(j,i) = toc(scenario_timer);
            success_m1000_all(j,i) = true;

            fprintf(' | m1000 L2=%.3e time=%.3f s', ...
                err_l2_m1000,m1000_total_time_all(j,i));

            clear basis_m1000 M_m1000 y_m1000 coef_m1000

        catch ME
            m1000_total_time_all(j,i) = toc(scenario_timer);
            error_m1000_all{j,i} = getReport( ...
                ME,'extended','hyperlinks','off');
            fprintf('\nONLY m->1000 FAILED\n%s\n', ...
                error_m1000_all{j,i});
        end

        %% ----------------------------------------------------------------
        %  Scenario 2: only grid -> 45x120
        %
        %  Reuse the original m-feature basis. The timer includes only:
        %  expanded PDE data + assembly + LS + test.
        % -----------------------------------------------------------------

        scenario_timer = tic;

        try
            t = tic;
            problem_grid = build_poisson_problem( ...
                cfg,expanded_interior_grid);
            grid_pde_time_all(j,i) = toc(t);

            t = tic;
            [M_grid,y_grid,~] = build_system( ...
                p_opt,problem_grid,basis);
            grid_assembly_time_all(j,i) = toc(t);

            t = tic;
            coef_grid = solve_least_squares( ...
                M_grid,y_grid,cfg.linear_solver);
            grid_ls_time_all(j,i) = toc(t);

            t = tic;
            [err_l2_grid,err_linf_grid] = evaluate_refit( ...
                Xtest,ref_test,p_opt,basis,coef_grid);
            grid_test_time_all(j,i) = toc(t);

            rel_l2_grid_all(j,i) = err_l2_grid;
            rel_linf_grid_all(j,i) = err_linf_grid;
            grid_total_time_all(j,i) = toc(scenario_timer);
            success_grid_all(j,i) = true;

            fprintf(' | grid L2=%.3e time=%.3f s', ...
                err_l2_grid,grid_total_time_all(j,i));

            clear problem_grid M_grid y_grid coef_grid

        catch ME
            grid_total_time_all(j,i) = toc(scenario_timer);
            error_grid_all{j,i} = getReport( ...
                ME,'extended','hyperlinks','off');
            fprintf('\nONLY grid->%dx%d FAILED\n%s\n', ...
                expanded_interior_grid(1), ...
                expanded_interior_grid(2), ...
                error_grid_all{j,i});
        end

        %% ----------------------------------------------------------------
        %  Scenario 3: m -> 1000 and grid -> 45x120 simultaneously
        %
        %  This path does not reuse Scenario 1 or Scenario 2 objects.
        %  Its independent timer includes:
        %  new basis + new PDE data + assembly + LS + test.
        % -----------------------------------------------------------------

        scenario_timer = tic;

        try
            t = tic;
            basis_both = build_random_weights( ...
                m_expand,cfg.domain,seed_i);
            both_basis_time_all(j,i) = toc(t);

            t = tic;
            problem_both = build_poisson_problem( ...
                cfg,expanded_interior_grid);
            both_pde_time_all(j,i) = toc(t);

            t = tic;
            [M_both,y_both,~] = build_system( ...
                p_opt,problem_both,basis_both);
            both_assembly_time_all(j,i) = toc(t);

            t = tic;
            coef_both = solve_least_squares( ...
                M_both,y_both,cfg.linear_solver);
            both_ls_time_all(j,i) = toc(t);

            t = tic;
            [err_l2_both,err_linf_both] = evaluate_refit( ...
                Xtest,ref_test,p_opt,basis_both,coef_both);
            both_test_time_all(j,i) = toc(t);

            rel_l2_both_all(j,i) = err_l2_both;
            rel_linf_both_all(j,i) = err_linf_both;
            both_total_time_all(j,i) = toc(scenario_timer);
            success_both_all(j,i) = true;

            fprintf(' | both L2=%.3e time=%.3f s\n', ...
                err_l2_both,both_total_time_all(j,i));

            clear basis_both problem_both M_both y_both coef_both

        catch ME
            both_total_time_all(j,i) = toc(scenario_timer);
            error_both_all{j,i} = getReport( ...
                ME,'extended','hyperlinks','off');
            fprintf('\nBOTH EXPANSIONS FAILED\n%s\n', ...
                error_both_all{j,i});
        end

        % At m=1000, the m-only and original feature spaces are identical.
        if m == m_expand && success_m1000_all(j,i)
            delta = abs( ...
                rel_l2_original_all(j,i)-rel_l2_m1000_all(j,i));

            if delta > 1e-12
                fprintf(['WARNING: repeated m=1000 refit differs from ', ...
                    'the original by %.3e.\n'],delta);
            end
        end
    end

    %% --------------------------------------------------------------------
    %  Intermediate save after each training feature dimension
    % ---------------------------------------------------------------------

    intermediate_file = fullfile(result_dir,sprintf( ...
        'feature_and_grid_study_after_m_%d.mat',m));

    save_study(intermediate_file);

    fprintf('\nCompleted m=%d: original %d, m-only %d, ',m, ...
        nnz(success_original_all(j,:)), ...
        nnz(success_m1000_all(j,:)));
    fprintf('grid-only %d, both %d successful runs.\n', ...
        nnz(success_grid_all(j,:)), ...
        nnz(success_both_all(j,:)));
end

%% ========================================================================
%  Statistics
% =========================================================================

valid_original = success_original_all;
valid_m1000 = success_original_all & success_m1000_all;
valid_grid = success_original_all & success_grid_all;
valid_both = success_original_all & success_both_all;

stats_l2_original = summarize_rows( ...
    rel_l2_original_all,valid_original);
stats_l2_m1000 = summarize_rows( ...
    rel_l2_m1000_all,valid_m1000);
stats_l2_grid = summarize_rows( ...
    rel_l2_grid_all,valid_grid);
stats_l2_both = summarize_rows( ...
    rel_l2_both_all,valid_both);

stats_linf_original = summarize_rows( ...
    rel_linf_original_all,valid_original);
stats_linf_m1000 = summarize_rows( ...
    rel_linf_m1000_all,valid_m1000);
stats_linf_grid = summarize_rows( ...
    rel_linf_grid_all,valid_grid);
stats_linf_both = summarize_rows( ...
    rel_linf_both_all,valid_both);

stats_p1 = summarize_rows(p1_opt_all,valid_original);
stats_p2 = summarize_rows(p2_opt_all,valid_original);
stats_checkpoint = summarize_rows( ...
    selected_checkpoint_all,valid_original);

original_setup_all = ...
    original_basis_time_all+ ...
    original_pde_time_all+ ...
    original_cache_time_all;

original_refit_all = ...
    original_assembly_time_all+ ...
    original_ls_time_all;

stats_original_setup = summarize_rows( ...
    original_setup_all,valid_original);
stats_original_optimization = summarize_rows( ...
    original_optimization_time_all,valid_original);
stats_original_refit = summarize_rows( ...
    original_refit_all,valid_original);
stats_original_test = summarize_rows( ...
    original_test_time_all,valid_original);
stats_original_total = summarize_rows( ...
    original_total_time_all,valid_original);

stats_m1000_total = summarize_rows( ...
    m1000_total_time_all,valid_m1000);
stats_grid_total = summarize_rows( ...
    grid_total_time_all,valid_grid);
stats_both_total = summarize_rows( ...
    both_total_time_all,valid_both);

TrainingFeatures = m_list;
OriginalInteriorPoints = base_num_interior*ones(num_m,1);
ExpandedFeatures = m_expand*ones(num_m,1);
ExpandedInteriorPoints = expanded_num_interior*ones(num_m,1);

OriginalSuccess = sum(valid_original,2);
M1000Success = sum(valid_m1000,2);
GridSuccess = sum(valid_grid,2);
BothSuccess = sum(valid_both,2);

SummaryTable = table( ...
    TrainingFeatures, ...
    OriginalInteriorPoints, ...
    ExpandedFeatures, ...
    ExpandedInteriorPoints, ...
    OriginalSuccess, ...
    M1000Success, ...
    GridSuccess, ...
    BothSuccess, ...
    stats_l2_original.mean, ...
    stats_l2_original.std, ...
    stats_l2_m1000.mean, ...
    stats_l2_m1000.std, ...
    stats_l2_grid.mean, ...
    stats_l2_grid.std, ...
    stats_l2_both.mean, ...
    stats_l2_both.std, ...
    stats_linf_original.mean, ...
    stats_linf_m1000.mean, ...
    stats_linf_grid.mean, ...
    stats_linf_both.mean, ...
    stats_p1.mean, ...
    stats_p1.std, ...
    stats_p2.mean, ...
    stats_p2.std, ...
    stats_checkpoint.mean, ...
    stats_checkpoint.std, ...
    stats_original_setup.mean, ...
    stats_original_optimization.mean, ...
    stats_original_refit.mean, ...
    stats_original_test.mean, ...
    stats_original_total.mean, ...
    stats_m1000_total.mean, ...
    stats_grid_total.mean, ...
    stats_both_total.mean, ...
    'VariableNames',{ ...
        'TrainingFeatures', ...
        'OriginalInteriorPoints', ...
        'ExpandedFeatures', ...
        'ExpandedInteriorPoints', ...
        'OriginalSuccess', ...
        'M1000Success', ...
        'GridSuccess', ...
        'BothSuccess', ...
        'L2OriginalMean', ...
        'L2OriginalStd', ...
        'L2M1000Mean', ...
        'L2M1000Std', ...
        'L2GridMean', ...
        'L2GridStd', ...
        'L2BothMean', ...
        'L2BothStd', ...
        'LinfOriginalMean', ...
        'LinfM1000Mean', ...
        'LinfGridMean', ...
        'LinfBothMean', ...
        'P1Mean', ...
        'P1Std', ...
        'P2Mean', ...
        'P2Std', ...
        'CheckpointMean', ...
        'CheckpointStd', ...
        'OriginalSetupMean', ...
        'OriginalOptimizationMean', ...
        'OriginalRefitMean', ...
        'OriginalTestMean', ...
        'OriginalTotalMean', ...
        'M1000OnlyTotalMean', ...
        'GridOnlyTotalMean', ...
        'BothTotalMean'});

%% ========================================================================
%  Display
% =========================================================================

fprintf('\n\n');
fprintf('====================================================================\n');
fprintf(' FINAL ACCURACY SUMMARY\n');
fprintf('====================================================================\n');
fprintf([' Train m   L2 original     L2 m->1000    ', ...
    'L2 grid->45x120   L2 both       p1*      p2*\n']);
fprintf('--------------------------------------------------------------------\n');

for j = 1:num_m
    fprintf('%7d   %.3e      %.3e      %.3e          %.3e    %7.3f  %7.3f\n', ...
        m_list(j), ...
        stats_l2_original.mean(j), ...
        stats_l2_m1000.mean(j), ...
        stats_l2_grid.mean(j), ...
        stats_l2_both.mean(j), ...
        stats_p1.mean(j), ...
        stats_p2.mean(j));
end

fprintf('\n');
fprintf('====================================================================\n');
fprintf(' INDEPENDENT MEAN TIMING SUMMARY (seconds)\n');
fprintf('====================================================================\n');
fprintf([' Train m   Original total   Only m->1000   ', ...
    'Only grid->45x120   Both expansions\n']);
fprintf('--------------------------------------------------------------------\n');

for j = 1:num_m
    fprintf('%7d      %10.4f      %10.4f          %10.4f          %10.4f\n', ...
        m_list(j), ...
        stats_original_total.mean(j), ...
        stats_m1000_total.mean(j), ...
        stats_grid_total.mean(j), ...
        stats_both_total.mean(j));
end

fprintf('\n');
disp(SummaryTable);

%% ========================================================================
%  Save
% =========================================================================

final_mat_file = fullfile( ...
    result_dir,'feature_number_and_grid_study_final.mat');

save_study(final_mat_file);

csv_file = fullfile( ...
    result_dir,'feature_number_and_grid_summary.csv');

writetable(SummaryTable,csv_file);

fprintf('\nResults saved to:\n%s\n%s\n', ...
    final_mat_file,csv_file);

%% ========================================================================
%  Figures
% =========================================================================

figure('Color','w');
hold on;
errorbar(m_list,stats_l2_original.mean,stats_l2_original.std, ...
    'o-','LineWidth',1.2,'MarkerSize',6);
errorbar(m_list,stats_l2_m1000.mean,stats_l2_m1000.std, ...
    's-','LineWidth',1.2,'MarkerSize',6);
errorbar(m_list,stats_l2_grid.mean,stats_l2_grid.std, ...
    'd-','LineWidth',1.2,'MarkerSize',6);
errorbar(m_list,stats_l2_both.mean,stats_l2_both.std, ...
    '^-','LineWidth',1.2,'MarkerSize',6);
hold off;
set(gca,'YScale','log','FontSize',12,'LineWidth',1);
box on;
xticks(m_list);
xlabel('Features used for distribution learning $m$', ...
    'Interpreter','latex');
ylabel('Mean relative $\ell_2$ error','Interpreter','latex');
legend({ ...
    'Original $(m,30\times80)$', ...
    'Only $m\rightarrow1000$', ...
    'Only grid $\rightarrow45\times120$', ...
    'Both expansions'}, ...
    'Interpreter','latex','Location','best');
title('Feature and collocation expansion');

figure('Color','w');
timing_matrix = [ ...
    stats_m1000_total.mean, ...
    stats_grid_total.mean, ...
    stats_both_total.mean];
bar(m_list,timing_matrix,'grouped');
set(gca,'FontSize',12,'LineWidth',1);
box on;
xticks(m_list);
xlabel('Features used for distribution learning $m$', ...
    'Interpreter','latex');
ylabel('Independent additional refit time / s', ...
    'Interpreter','latex');
legend({ ...
    'Only $m\rightarrow1000$', ...
    'Only grid $\rightarrow45\times120$', ...
    'Both expansions'}, ...
    'Interpreter','latex','Location','best');
title('Independent expansion costs');

figure('Color','w');
original_components = [ ...
    stats_original_setup.mean, ...
    stats_original_optimization.mean, ...
    stats_original_refit.mean, ...
    stats_original_test.mean];
bar(m_list,original_components,'stacked');
set(gca,'FontSize',12,'LineWidth',1);
box on;
xticks(m_list);
xlabel('Features used for distribution learning $m$', ...
    'Interpreter','latex');
ylabel('Original AD-RaNN runtime / s','Interpreter','latex');
legend({'Setup','Optimization','Final refit','Test'}, ...
    'Location','best');
title('Original runtime breakdown');

%% ========================================================================
%  Local functions
% =========================================================================

function problem = build_poisson_problem(cfg,interior_grid)

    problem.domain = cfg.domain;
    problem.boundary_penalty = cfg.boundary_penalty;
    problem.Xi = tensor_grid(cfg.domain,interior_grid,1e-6);
    problem.fi = rhs(problem.Xi);

    nB = cfg.boundary_points_per_side;

    x = linspace(cfg.domain(1,1),cfg.domain(1,2),nB)';
    y = linspace(cfg.domain(2,1),cfg.domain(2,2),nB)';

    problem.Xb = [ ...
        cfg.domain(1,1)*ones(nB,1),y; ...
        cfg.domain(1,2)*ones(nB,1),y; ...
        x,cfg.domain(2,1)*ones(nB,1); ...
        x,cfg.domain(2,2)*ones(nB,1)];

    problem.gb = exact_solution(problem.Xb);

    problem.y = [ ...
        problem.fi; ...
        cfg.boundary_penalty*problem.gb];
end


function [err_l2,err_linf] = evaluate_refit( ...
    Xtest,ref_test,p,basis,coef)

    Phi_test = gaussian_features(Xtest,p,basis);
    prediction = Phi_test*coef;

    err_l2 = relative_l2(prediction,ref_test);
    err_linf = relative_linf(prediction,ref_test);
end


function stats = summarize_rows(values,valid)

    num_rows = size(values,1);

    stats.mean = nan(num_rows,1);
    stats.std = nan(num_rows,1);
    stats.median = nan(num_rows,1);
    stats.best = nan(num_rows,1);
    stats.worst = nan(num_rows,1);
    stats.count = zeros(num_rows,1);

    for row = 1:num_rows
        use = valid(row,:) & isfinite(values(row,:));
        stats.count(row) = nnz(use);

        if any(use)
            selected = values(row,use);
            stats.mean(row) = mean(selected);
            stats.std(row) = std(selected);
            stats.median(row) = median(selected);
            stats.best(row) = min(selected);
            stats.worst(row) = max(selected);
        end
    end
end


function [obj,grad,info] = poisson_objective_feature_seed( ...
    p,problem,basis,lambda,ls_opts)

    [M,y,dM] = build_system(p,problem,basis);

    [obj,grad,info] = reduced_objective_gradient( ...
        M,y,dM,lambda,ls_opts);
end


function save_study(file_name)

    names = evalin('caller','who');

    excluded = { ...
        'M','M_m1000','M_grid','M_both', ...
        'coef','coef_m1000','coef_grid','coef_both', ...
        'cache','objective_fun','ME','names'};

    names = setdiff(names,excluded);

    save_args = [{file_name},names.',{'-v7.3'}];
    evalin('caller',sprintf( ...
        'save(%s);',join_save_arguments(save_args)));
end


function command = join_save_arguments(arguments)

    quoted = cell(size(arguments));

    for k = 1:numel(arguments)
        quoted{k} = ['''',strrep(arguments{k},'''',''''''),''''];
    end

    command = strjoin(quoted,',');
end
