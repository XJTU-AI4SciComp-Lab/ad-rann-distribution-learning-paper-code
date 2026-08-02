clear;
clc;
close all;

warning('off','MATLAB:rankDeficientMatrix');

%% ========================================================================
%  Path
% =========================================================================

this_file = mfilename('fullpath');

lambda_dir = fileparts(this_file);
example_dir = fileparts(lambda_dir);

root = fileparts(fileparts(example_dir));

addpath(genpath(fullfile(root,'src')));
addpath(example_dir);
addpath(lambda_dir);


%% ========================================================================
%  Base configuration
% =========================================================================

cfg_base = config();

% No Adam iteration print.
cfg_base.optimizer.verbose = false;

% Keep the same optimizer settings.
cfg_base.optimizer.store_moments = true;
cfg_base.optimizer.store_full_info = false;

% Do not compute SVD inside Adam.
cfg_base.compute_spectrum = false;


%% ========================================================================
%  GPU
% =========================================================================

if cfg_base.linear_solver.use_gpu

    if ~isfield(cfg_base.linear_solver,'gpu_id')
        error('Please define cfg.linear_solver.gpu_id in config.m.');
    end

    gpu_id = cfg_base.linear_solver.gpu_id;
    num_gpu = gpuDeviceCount;

    if gpu_id > num_gpu
        error( ...
            'GPU %d is not available. MATLAB sees only %d GPU(s).', ...
            gpu_id,num_gpu);
    end

    g = gpuDevice(gpu_id);

    fprintf('\n');
    fprintf('Using GPU %d: %s\n',g.Index,g.Name);
    fprintf('Available memory: %.2f GB\n\n', ...
        g.AvailableMemory/1024^3);
end


%% ========================================================================
%  Lambda values
% =========================================================================

lambda_list = [ ...
    1e1, ...
    1e0, ...
    1e-1, ...
    1e-2, ...
    1e-3, ...
    1e-4, ...
    1e-5, ...
    1e-6, ...
    1e-7, ...
    1e-8, ...
    1e-9, ...
    1e-10, ...
    1e-11, ...
    1e-12, ...
    1e-13, ...
    1e-14, ...
    0];

num_lambda = numel(lambda_list);


%% ========================================================================
%  Seeds
% =========================================================================

seeds = 1:100;
num_seeds = numel(seeds);


%% ========================================================================
%  Fixed initialization
% =========================================================================

p0 = [4;8];


%% ========================================================================
%  Conditioning diagnostics
%
%  SVD is computed after the method timer has stopped.
%  Therefore SVD time is NOT included in total_time_all.
% =========================================================================

compute_conditioning = true;


%% ========================================================================
%  Test grid
%
%  Constructed only once and not included in per-run timing.
% =========================================================================

Xtest = tensor_grid( ...
    cfg_base.domain, ...
    cfg_base.test_grid, ...
    0);

ref_test = exact_solution(Xtest);


%% ========================================================================
%  Allocate
%
%  row    = lambda
%  column = seed
% =========================================================================

p1_opt_all = nan(num_lambda,num_seeds);
p2_opt_all = nan(num_lambda,num_seeds);

best_iteration_all = nan(num_lambda,num_seeds);
best_mse_all = nan(num_lambda,num_seeds);

rel_l2_all = nan(num_lambda,num_seeds);
rel_linf_all = nan(num_lambda,num_seeds);

success_all = false(num_lambda,num_seeds);
error_message = cell(num_lambda,num_seeds);


%% ========================================================================
%  Detailed timing
% =========================================================================

basis_time_all = nan(num_lambda,num_seeds);
pde_time_all = nan(num_lambda,num_seeds);
cache_time_all = nan(num_lambda,num_seeds);

training_time_all = nan(num_lambda,num_seeds);

assembly_time_all = nan(num_lambda,num_seeds);
final_ls_time_all = nan(num_lambda,num_seeds);
test_time_all = nan(num_lambda,num_seeds);

other_time_all = nan(num_lambda,num_seeds);
total_time_all = nan(num_lambda,num_seeds);


%% ========================================================================
%  Conditioning data
% =========================================================================

sigma_min_all = nan(num_lambda,num_seeds);
sigma_max_all = nan(num_lambda,num_seeds);

kappa_M_all = nan(num_lambda,num_seeds);
kappa_ridge_all = nan(num_lambda,num_seeds);

conditioning_time_all = nan(num_lambda,num_seeds);


%% ========================================================================
%  Header
% =========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('           AD-RaNN lambda x random-seed experiment\n');
fprintf('====================================================================\n');

fprintf('Initial p       = [%.6f, %.6f]\n',p0(1),p0(2));
fprintf('Lambda values   = %d\n',num_lambda);
fprintf('Seeds           = %d : %d\n',seeds(1),seeds(end));
fprintf('Total runs      = %d\n',num_lambda*num_seeds);
fprintf('Features        = %d\n',cfg_base.num_features);

fprintf('\nLambda list:\n');

for j = 1:num_lambda
    fprintf('  %2d: %.1e\n',j,lambda_list(j));
end

fprintf('====================================================================\n\n');


%% ========================================================================
%  Main experiment
% =========================================================================

for j = 1:num_lambda

    lambda_j = lambda_list(j);

    fprintf('\n');
    fprintf('####################################################################\n');
    fprintf('Lambda %d / %d\n',j,num_lambda);
    fprintf('lambda = %.1e\n',lambda_j);
    fprintf('####################################################################\n\n');


    for i = 1:num_seeds

        seed_i = seeds(i);

        fprintf( ...
            '[lambda %2d/%2d | seed %3d/%3d] ... ', ...
            j,num_lambda,seed_i,num_seeds);

        total_timer = tic;

        try

            %% ============================================================
            %  Configuration
            % =============================================================

            cfg = cfg_base;

            cfg.seed = seed_i;
            cfg.lambda = lambda_j;


            %% ============================================================
            %  Random basis
            % =============================================================

            section_timer = tic;

            basis = build_random_weights( ...
                cfg.num_features, ...
                cfg.domain, ...
                cfg.seed);

            basis_time_all(j,i) = toc(section_timer);


            %% ============================================================
            %  PDE data
            % =============================================================

            section_timer = tic;

            problem.domain = cfg.domain;
            problem.boundary_penalty = cfg.boundary_penalty;

            problem.Xi = tensor_grid( ...
                cfg.domain, ...
                cfg.interior_grid, ...
                1e-6);

            problem.fi = rhs(problem.Xi);

            nB = cfg.boundary_points_per_side;

            x = linspace( ...
                cfg.domain(1,1), ...
                cfg.domain(1,2), ...
                nB)';

            y = linspace( ...
                cfg.domain(2,1), ...
                cfg.domain(2,2), ...
                nB)';

            problem.Xb = [ ...
                cfg.domain(1,1)*ones(nB,1), y; ...
                cfg.domain(1,2)*ones(nB,1), y; ...
                x, cfg.domain(2,1)*ones(nB,1); ...
                x, cfg.domain(2,2)*ones(nB,1)];

            problem.gb = exact_solution(problem.Xb);

            problem.y = [ ...
                problem.fi; ...
                cfg.boundary_penalty*problem.gb];

            pde_time_all(j,i) = toc(section_timer);


            %% ============================================================
            %  Least-squares options
            % =============================================================

            ls_opts = cfg.linear_solver;
            ls_opts.compute_spectrum = false;


            %% ============================================================
            %  Cache / objective preparation
            % =============================================================

            section_timer = tic;

            if cfg.use_fast_evaluator

                cache = prepare_poisson_cache( ...
                    problem, ...
                    basis);

                objective_fun = @(p) ...
                    evaluate_poisson_reduced_fast( ...
                        p, ...
                        cache, ...
                        cfg.lambda, ...
                        ls_opts);

            else

                objective_fun = @(p) ...
                    poisson_objective_lambda_seed( ...
                        p, ...
                        problem, ...
                        basis, ...
                        cfg.lambda, ...
                        ls_opts);

            end

            cache_time_all(j,i) = toc(section_timer);


            %% ============================================================
            %  Distribution optimization
            %
            %  This includes repeated objective / gradient evaluations
            %  and the ridge least-squares solves inside Adam.
            % =============================================================

            section_timer = tic;

            [p_opt,history] = ...
                optimize_distribution_adam( ...
                    p0, ...
                    objective_fun, ...
                    cfg.optimizer);

            training_time_all(j,i) = toc(section_timer);


            %% ============================================================
            %  Final system assembly
            % =============================================================

            section_timer = tic;

            [M,y_rhs,~] = ...
                build_system( ...
                    p_opt, ...
                    problem, ...
                    basis);

            assembly_time_all(j,i) = toc(section_timer);


            %% ============================================================
            %  Final unregularized least-squares refit
            % =============================================================

            section_timer = tic;

            [coef,final_ls_info] = ...
                solve_least_squares( ...
                    M, ...
                    y_rhs, ...
                    cfg.linear_solver);

            final_ls_time_all(j,i) = toc(section_timer);


            %% ============================================================
            %  Independent test error
            % =============================================================

            section_timer = tic;

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

            test_time_all(j,i) = toc(section_timer);


            %% ============================================================
            %  Store algorithm results
            % =============================================================

            p1_opt_all(j,i) = p_opt(1);
            p2_opt_all(j,i) = p_opt(2);

            best_iteration_all(j,i) = ...
                history.best_iteration;

            best_mse_all(j,i) = ...
                history.best_selection_value;

            rel_l2_all(j,i) = err_l2;
            rel_linf_all(j,i) = err_linf;


            %% ============================================================
            %  Total method time
            %
            %  Stop BEFORE conditioning/SVD diagnostics.
            % =============================================================

            total_time_all(j,i) = toc(total_timer);


            %% ============================================================
            %  Small unclassified overhead
            % =============================================================

            tracked_time = ...
                basis_time_all(j,i) + ...
                pde_time_all(j,i) + ...
                cache_time_all(j,i) + ...
                training_time_all(j,i) + ...
                assembly_time_all(j,i) + ...
                final_ls_time_all(j,i) + ...
                test_time_all(j,i);

            other_time_all(j,i) = ...
                max(total_time_all(j,i) - tracked_time,0);


            success_all(j,i) = true;


            %% ============================================================
            %  Conditioning diagnostics
            %
            %  NOT included in total_time_all.
            % =============================================================

            if compute_conditioning

                conditioning_timer = tic;

                try

                    singular_values = svd(M,'econ');

                    sigma_max = singular_values(1);
                    sigma_min = singular_values(end);

                    sigma_max_all(j,i) = sigma_max;
                    sigma_min_all(j,i) = sigma_min;

                    if sigma_min > 0

                        kappa_M_all(j,i) = ...
                            sigma_max/sigma_min;

                    else

                        kappa_M_all(j,i) = Inf;

                    end

                    if lambda_j > 0

                        kappa_ridge_all(j,i) = ...
                            (sigma_max^2 + lambda_j) / ...
                            (sigma_min^2 + lambda_j);

                    else

                        if sigma_min > 0

                            kappa_ridge_all(j,i) = ...
                                (sigma_max/sigma_min)^2;

                        else

                            kappa_ridge_all(j,i) = Inf;

                        end

                    end

                    conditioning_time_all(j,i) = ...
                        toc(conditioning_timer);

                catch ME_cond

                    conditioning_time_all(j,i) = ...
                        toc(conditioning_timer);

                    fprintf('\n');
                    fprintf( ...
                        'Conditioning diagnostic failed: %s\n', ...
                        ME_cond.message);

                end

            end


            %% ============================================================
            %  Print current run
            % =============================================================

            fprintf( ...
                ['p*=[%.6f, %.6f] | ', ...
                 'MSE=%.3e | ', ...
                 'L2=%.3e | ', ...
                 'Linf=%.3e | ', ...
                 'time=%.3f s\n'], ...
                p_opt(1), ...
                p_opt(2), ...
                history.best_selection_value, ...
                err_l2, ...
                err_linf, ...
                total_time_all(j,i));


        catch ME

            success_all(j,i) = false;

            total_time_all(j,i) = toc(total_timer);

            error_message{j,i} = ...
                getReport( ...
                    ME, ...
                    'extended', ...
                    'hyperlinks', ...
                    'off');

            fprintf('FAILED\n');
            fprintf('%s\n',error_message{j,i});

        end

    end


    %% ====================================================================
    %  Summary for current lambda
    % =====================================================================

    valid = success_all(j,:);

    fprintf('\n');
    fprintf('--------------------------------------------------------------------\n');
    fprintf('Summary for lambda = %.1e\n',lambda_j);
    fprintf('--------------------------------------------------------------------\n');

    fprintf('Successful runs = %d / %d\n', ...
        nnz(valid),num_seeds);


    if any(valid)

        current_l2 = rel_l2_all(j,valid);
        current_linf = rel_linf_all(j,valid);

        fprintf('\nRelative L2\n');
        fprintf('Mean   = %.6e\n',mean(current_l2));
        fprintf('Std    = %.6e\n',std(current_l2));
        fprintf('Median = %.6e\n',median(current_l2));
        fprintf('Best   = %.6e\n',min(current_l2));
        fprintf('Worst  = %.6e\n',max(current_l2));

        fprintf('\nRelative Linf\n');
        fprintf('Mean   = %.6e\n',mean(current_linf));
        fprintf('Std    = %.6e\n',std(current_linf));
        fprintf('Median = %.6e\n',median(current_linf));
        fprintf('Best   = %.6e\n',min(current_linf));
        fprintf('Worst  = %.6e\n',max(current_linf));

        fprintf('\np1*\n');
        fprintf('Mean = %.6f\n',mean(p1_opt_all(j,valid)));
        fprintf('Std  = %.6f\n',std(p1_opt_all(j,valid)));

        fprintf('\np2*\n');
        fprintf('Mean = %.6f\n',mean(p2_opt_all(j,valid)));
        fprintf('Std  = %.6f\n',std(p2_opt_all(j,valid)));

        fprintf('\nMean selected iteration = %.3f\n', ...
            mean(best_iteration_all(j,valid)));

        fprintf('\nMean timing\n');

        fprintf('Basis        = %.6f s\n', ...
            mean(basis_time_all(j,valid)));

        fprintf('PDE data     = %.6f s\n', ...
            mean(pde_time_all(j,valid)));

        fprintf('Cache/setup  = %.6f s\n', ...
            mean(cache_time_all(j,valid)));

        fprintf('Optimization = %.6f s\n', ...
            mean(training_time_all(j,valid)));

        fprintf('Assembly     = %.6f s\n', ...
            mean(assembly_time_all(j,valid)));

        fprintf('Final LS     = %.6f s\n', ...
            mean(final_ls_time_all(j,valid)));

        fprintf('Test         = %.6f s\n', ...
            mean(test_time_all(j,valid)));

        fprintf('Other        = %.6f s\n', ...
            mean(other_time_all(j,valid)));

        fprintf('Total        = %.6f s\n', ...
            mean(total_time_all(j,valid)));


        if compute_conditioning

            valid_cond = ...
                valid & ...
                isfinite(sigma_min_all(j,:)) & ...
                isfinite(kappa_ridge_all(j,:));

            if any(valid_cond)

                fprintf('\nConditioning\n');

                fprintf('Mean sigma_min       = %.6e\n', ...
                    mean(sigma_min_all(j,valid_cond)));

                fprintf('Median sigma_min     = %.6e\n', ...
                    median(sigma_min_all(j,valid_cond)));

                fprintf('Mean kappa(M)        = %.6e\n', ...
                    mean(kappa_M_all(j,valid_cond)));

                fprintf('Median kappa(M)      = %.6e\n', ...
                    median(kappa_M_all(j,valid_cond)));

                fprintf('Mean ridge condition = %.6e\n', ...
                    mean(kappa_ridge_all(j,valid_cond)));

                fprintf('Median ridge cond.   = %.6e\n', ...
                    median(kappa_ridge_all(j,valid_cond)));

            end

            valid_svd = ...
                valid & ...
                isfinite(conditioning_time_all(j,:));

            if any(valid_svd)

                fprintf('Mean SVD time         = %.6f s\n', ...
                    mean(conditioning_time_all(j,valid_svd)));

            end

        end

    end

    fprintf('--------------------------------------------------------------------\n');

end


%% ========================================================================
%  Final statistics
% =========================================================================

L2_mean = nan(num_lambda,1);
L2_std = nan(num_lambda,1);
L2_median = nan(num_lambda,1);
L2_best = nan(num_lambda,1);
L2_worst = nan(num_lambda,1);

Linf_mean = nan(num_lambda,1);
Linf_std = nan(num_lambda,1);
Linf_median = nan(num_lambda,1);
Linf_best = nan(num_lambda,1);
Linf_worst = nan(num_lambda,1);

MSE_mean = nan(num_lambda,1);
MSE_std = nan(num_lambda,1);

p1_mean = nan(num_lambda,1);
p1_std = nan(num_lambda,1);

p2_mean = nan(num_lambda,1);
p2_std = nan(num_lambda,1);

best_iter_mean = nan(num_lambda,1);

training_mean = nan(num_lambda,1);
training_std = nan(num_lambda,1);

total_mean = nan(num_lambda,1);
total_std = nan(num_lambda,1);

sigma_min_mean = nan(num_lambda,1);
sigma_min_median = nan(num_lambda,1);
sigma_min_min = nan(num_lambda,1);

kappa_M_mean = nan(num_lambda,1);
kappa_M_median = nan(num_lambda,1);

kappa_ridge_mean = nan(num_lambda,1);
kappa_ridge_median = nan(num_lambda,1);

num_success = zeros(num_lambda,1);


%% Detailed timing means

basis_time_mean = nan(num_lambda,1);
pde_time_mean = nan(num_lambda,1);
cache_time_mean = nan(num_lambda,1);

assembly_time_mean = nan(num_lambda,1);
final_ls_time_mean = nan(num_lambda,1);
test_time_mean = nan(num_lambda,1);

other_time_mean = nan(num_lambda,1);

conditioning_time_mean = nan(num_lambda,1);


%% ========================================================================
%  Compute statistics
% =========================================================================

for j = 1:num_lambda

    valid = success_all(j,:);

    num_success(j) = nnz(valid);

    if ~any(valid)
        continue;
    end


    %% Error

    L2_mean(j) = mean(rel_l2_all(j,valid));
    L2_std(j) = std(rel_l2_all(j,valid));
    L2_median(j) = median(rel_l2_all(j,valid));
    L2_best(j) = min(rel_l2_all(j,valid));
    L2_worst(j) = max(rel_l2_all(j,valid));

    Linf_mean(j) = mean(rel_linf_all(j,valid));
    Linf_std(j) = std(rel_linf_all(j,valid));
    Linf_median(j) = median(rel_linf_all(j,valid));
    Linf_best(j) = min(rel_linf_all(j,valid));
    Linf_worst(j) = max(rel_linf_all(j,valid));


    %% Training MSE

    MSE_mean(j) = mean(best_mse_all(j,valid));
    MSE_std(j) = std(best_mse_all(j,valid));


    %% Optimized p

    p1_mean(j) = mean(p1_opt_all(j,valid));
    p1_std(j) = std(p1_opt_all(j,valid));

    p2_mean(j) = mean(p2_opt_all(j,valid));
    p2_std(j) = std(p2_opt_all(j,valid));


    %% Selected checkpoint

    best_iter_mean(j) = ...
        mean(best_iteration_all(j,valid));


    %% Detailed timing

    basis_time_mean(j) = ...
        mean(basis_time_all(j,valid));

    pde_time_mean(j) = ...
        mean(pde_time_all(j,valid));

    cache_time_mean(j) = ...
        mean(cache_time_all(j,valid));

    training_mean(j) = ...
        mean(training_time_all(j,valid));

    training_std(j) = ...
        std(training_time_all(j,valid));

    assembly_time_mean(j) = ...
        mean(assembly_time_all(j,valid));

    final_ls_time_mean(j) = ...
        mean(final_ls_time_all(j,valid));

    test_time_mean(j) = ...
        mean(test_time_all(j,valid));

    other_time_mean(j) = ...
        mean(other_time_all(j,valid));

    total_mean(j) = ...
        mean(total_time_all(j,valid));

    total_std(j) = ...
        std(total_time_all(j,valid));


    %% Conditioning

    valid_cond = ...
        valid & ...
        isfinite(sigma_min_all(j,:)) & ...
        isfinite(kappa_ridge_all(j,:));

    if any(valid_cond)

        sigma_min_mean(j) = ...
            mean(sigma_min_all(j,valid_cond));

        sigma_min_median(j) = ...
            median(sigma_min_all(j,valid_cond));

        sigma_min_min(j) = ...
            min(sigma_min_all(j,valid_cond));

        kappa_M_mean(j) = ...
            mean(kappa_M_all(j,valid_cond));

        kappa_M_median(j) = ...
            median(kappa_M_all(j,valid_cond));

        kappa_ridge_mean(j) = ...
            mean(kappa_ridge_all(j,valid_cond));

        kappa_ridge_median(j) = ...
            median(kappa_ridge_all(j,valid_cond));

    end


    valid_svd = ...
        valid & ...
        isfinite(conditioning_time_all(j,:));

    if any(valid_svd)

        conditioning_time_mean(j) = ...
            mean(conditioning_time_all(j,valid_svd));

    end

end


%% ========================================================================
%  Main summary table
% =========================================================================

Lambda = lambda_list(:);

SummaryTable = table( ...
    Lambda, ...
    num_success, ...
    L2_mean, ...
    L2_std, ...
    L2_median, ...
    L2_best, ...
    L2_worst, ...
    Linf_mean, ...
    Linf_std, ...
    MSE_mean, ...
    p1_mean, ...
    p1_std, ...
    p2_mean, ...
    p2_std, ...
    best_iter_mean, ...
    training_mean, ...
    training_std, ...
    total_mean, ...
    total_std, ...
    sigma_min_mean, ...
    sigma_min_median, ...
    sigma_min_min, ...
    kappa_M_mean, ...
    kappa_M_median, ...
    kappa_ridge_mean, ...
    kappa_ridge_median);


%% ========================================================================
%  Detailed runtime table
% =========================================================================

TimingTable = table( ...
    Lambda, ...
    num_success, ...
    basis_time_mean, ...
    pde_time_mean, ...
    cache_time_mean, ...
    training_mean, ...
    assembly_time_mean, ...
    final_ls_time_mean, ...
    test_time_mean, ...
    other_time_mean, ...
    total_mean, ...
    conditioning_time_mean, ...
    'VariableNames',{ ...
        'Lambda', ...
        'Success', ...
        'Basis_s', ...
        'PDE_s', ...
        'Cache_s', ...
        'Optimization_s', ...
        'Assembly_s', ...
        'FinalLS_s', ...
        'Test_s', ...
        'Other_s', ...
        'Total_s', ...
        'SVD_s'});


%% ========================================================================
%  Print full summary
% =========================================================================

fprintf('\n\n');
fprintf('====================================================================\n');
fprintf('                         FINAL SUMMARY\n');
fprintf('====================================================================\n\n');

disp(SummaryTable);


%% ========================================================================
%  Compact accuracy table
% =========================================================================

fprintf('\n');
fprintf('=============================================================================================================================\n');

fprintf( ...
    [' lambda       Mean L2      Std L2       Best L2       Worst L2     ', ...
     'Mean iter   Opt./s     Total/s    Ridge cond\n']);

fprintf('=============================================================================================================================\n');

for j = 1:num_lambda

    fprintf( ...
        ['%-10.1e   %.3e   %.3e   %.3e   %.3e   ', ...
         '%8.2f   %8.3f   %8.3f   %.3e\n'], ...
        lambda_list(j), ...
        L2_mean(j), ...
        L2_std(j), ...
        L2_best(j), ...
        L2_worst(j), ...
        best_iter_mean(j), ...
        training_mean(j), ...
        total_mean(j), ...
        kappa_ridge_mean(j));

end

fprintf('=============================================================================================================================\n');


%% ========================================================================
%  Print detailed timing table
% =========================================================================

fprintf('\n\n');
fprintf('====================================================================\n');
fprintf('                    MEAN RUNTIME BREAKDOWN\n');
fprintf('====================================================================\n\n');

disp(TimingTable);


%% ========================================================================
%  Compact timing table
% =========================================================================

fprintf('\n');
fprintf('=====================================================================================================================================\n');
fprintf('                                                MEAN RUNTIME BREAKDOWN (s)\n');
fprintf('=====================================================================================================================================\n');

fprintf([ ...
    ' lambda       Basis       PDE      Cache       Opt.    Assembly    FinalLS      Test     Other     Total       SVD\n']);

fprintf('-------------------------------------------------------------------------------------------------------------------------------------\n');

for j = 1:num_lambda

    fprintf( ...
        ['%-10.1e  ', ...
         '%8.4f  ', ...
         '%8.4f  ', ...
         '%8.4f  ', ...
         '%8.4f  ', ...
         '%8.4f  ', ...
         '%8.4f  ', ...
         '%8.4f  ', ...
         '%8.4f  ', ...
         '%8.4f  ', ...
         '%8.4f\n'], ...
        lambda_list(j), ...
        basis_time_mean(j), ...
        pde_time_mean(j), ...
        cache_time_mean(j), ...
        training_mean(j), ...
        assembly_time_mean(j), ...
        final_ls_time_mean(j), ...
        test_time_mean(j), ...
        other_time_mean(j), ...
        total_mean(j), ...
        conditioning_time_mean(j));

end

fprintf('=====================================================================================================================================\n');


%% ========================================================================
%  Additional compact timing table for paper
%
%  Setup  = basis + PDE data + cache
%  Final  = final assembly + final least-squares refit
% =========================================================================

setup_time_mean = ...
    basis_time_mean + ...
    pde_time_mean + ...
    cache_time_mean;

final_time_mean = ...
    assembly_time_mean + ...
    final_ls_time_mean;


PaperTimingTable = table( ...
    Lambda, ...
    setup_time_mean, ...
    training_mean, ...
    final_time_mean, ...
    test_time_mean, ...
    total_mean, ...
    'VariableNames',{ ...
        'Lambda', ...
        'Setup_s', ...
        'Optimization_s', ...
        'FinalRefit_s', ...
        'Test_s', ...
        'Total_s'});


fprintf('\n\n');
fprintf('====================================================================\n');
fprintf('                   COMPACT PAPER TIMING TABLE\n');
fprintf('====================================================================\n\n');

disp(PaperTimingTable);


fprintf('\n');
fprintf('=======================================================================================\n');
fprintf('                              PAPER TIMING TABLE (s)\n');
fprintf('=======================================================================================\n');
fprintf(' lambda       Setup       Optimization    Final refit      Test        Total\n');
fprintf('---------------------------------------------------------------------------------------\n');

for j = 1:num_lambda

    fprintf( ...
        '%-10.1e  %10.4f  %12.4f  %12.4f  %10.4f  %10.4f\n', ...
        lambda_list(j), ...
        setup_time_mean(j), ...
        training_mean(j), ...
        final_time_mean(j), ...
        test_time_mean(j), ...
        total_mean(j));

end

fprintf('=======================================================================================\n');


%% ========================================================================
%  Figures
%
%  lambda = 0 is excluded from logarithmic x-axes.
% =========================================================================

positive_lambda = lambda_list > 0;

lambda_plot = lambda_list(positive_lambda);

L2_mean_plot = L2_mean(positive_lambda);
L2_best_plot = L2_best(positive_lambda);
L2_worst_plot = L2_worst(positive_lambda);

p1_mean_plot = p1_mean(positive_lambda);
p1_std_plot = p1_std(positive_lambda);

p2_mean_plot = p2_mean(positive_lambda);
p2_std_plot = p2_std(positive_lambda);

kappa_plot = kappa_ridge_mean(positive_lambda);


%% ========================================================================
%  Figure 1: Mean / worst / best relative L2 error
% =========================================================================

figure( ...
    'Position',[100 100 760 500], ...
    'Color','w');

h1 = loglog( ...
    lambda_plot, ...
    L2_mean_plot, ...
    'o-', ...
    'LineWidth',1.3, ...
    'MarkerSize',5);

hold on;

h2 = loglog( ...
    lambda_plot, ...
    L2_worst_plot, ...
    's-', ...
    'LineWidth',1.3, ...
    'MarkerSize',5);

h3 = loglog( ...
    lambda_plot, ...
    L2_best_plot, ...
    '^-', ...
    'LineWidth',1.3, ...
    'MarkerSize',5);

hold off;

xlabel( ...
    'Ridge parameter $\lambda$', ...
    'Interpreter','latex', ...
    'FontSize',14);

ylabel( ...
    'Relative $\ell_2$ error', ...
    'Interpreter','latex', ...
    'FontSize',14);

title( ...
    'Accuracy versus ridge parameter', ...
    'Interpreter','latex', ...
    'FontSize',12);

legend( ...
    [h1,h2,h3], ...
    {'Mean','Worst','Best'}, ...
    'Interpreter','latex', ...
    'Location','best', ...
    'Box','off');

set(gca, ...
    'XScale','log', ...
    'YScale','log', ...
    'XDir','reverse', ...
    'FontSize',12, ...
    'LineWidth',1.0, ...
    'Color','w');

box on;
grid off;


%% ========================================================================
%  Figure 2: Optimized distribution parameters
% =========================================================================

figure( ...
    'Position',[900 100 760 500], ...
    'Color','w');

h4 = errorbar( ...
    lambda_plot, ...
    p1_mean_plot, ...
    p1_std_plot, ...
    'o-', ...
    'LineWidth',1.3, ...
    'MarkerSize',5);

hold on;

h5 = errorbar( ...
    lambda_plot, ...
    p2_mean_plot, ...
    p2_std_plot, ...
    's-', ...
    'LineWidth',1.3, ...
    'MarkerSize',5);

hold off;

xlabel( ...
    'Ridge parameter $\lambda$', ...
    'Interpreter','latex', ...
    'FontSize',14);

ylabel( ...
    'Optimized distribution parameter', ...
    'Interpreter','latex', ...
    'FontSize',14);

title( ...
    'Optimized distribution parameters', ...
    'Interpreter','latex', ...
    'FontSize',12);

legend( ...
    [h4,h5], ...
    {'Mean $p_1^*$','Mean $p_2^*$'}, ...
    'Interpreter','latex', ...
    'Location','best', ...
    'Box','off');

set(gca, ...
    'XScale','log', ...
    'XDir','reverse', ...
    'FontSize',12, ...
    'LineWidth',1.0, ...
    'Color','w');

box on;
grid off;


%% ========================================================================
%  Figure 3: Ridge condition number
% =========================================================================

if compute_conditioning

    figure( ...
        'Position',[500 650 760 500], ...
        'Color','w');

    loglog( ...
        lambda_plot, ...
        kappa_plot, ...
        'o-', ...
        'LineWidth',1.3, ...
        'MarkerSize',5);

    xlabel( ...
        'Ridge parameter $\lambda$', ...
        'Interpreter','latex', ...
        'FontSize',14);

    ylabel( ...
        'Mean ridge condition number', ...
        'Interpreter','latex', ...
        'FontSize',14);

    title( ...
        'Conditioning versus ridge parameter', ...
        'Interpreter','latex', ...
        'FontSize',12);

    set(gca, ...
        'XScale','log', ...
        'YScale','log', ...
        'XDir','reverse', ...
        'FontSize',12, ...
        'LineWidth',1.0, ...
        'Color','w');

    box on;
    grid off;

end


%% ========================================================================
%  Generic objective
% =========================================================================

function [obj,grad,info] = ...
    poisson_objective_lambda_seed( ...
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