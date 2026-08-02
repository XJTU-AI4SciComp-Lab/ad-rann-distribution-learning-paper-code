clear;
clc;

warning('off','MATLAB:rankDeficientMatrix');

%% ========================================================================
%  Path
% =========================================================================

this_file = mfilename('fullpath');
seed_dir = fileparts(this_file);
example_dir = fileparts(seed_dir);
root = fileparts(fileparts(example_dir));

addpath(genpath(fullfile(root,'src')));
addpath(example_dir);
addpath(seed_dir);


%% ========================================================================
%  Base configuration
% =========================================================================

cfg_base = config();


%% ========================================================================
%  GPU
% =========================================================================

if cfg_base.linear_solver.use_gpu

    gpu_id = cfg_base.linear_solver.gpu_id;

    if gpu_id > gpuDeviceCount
        error( ...
            'GPU %d is not available. MATLAB sees only %d GPU(s).', ...
            gpu_id, gpuDeviceCount);
    end

    g = gpuDevice(gpu_id);

    fprintf('\nUsing GPU %d: %s\n', ...
        g.Index,g.Name);

    fprintf('Available memory: %.2f GB\n\n', ...
        g.AvailableMemory/1024^3);
end


%% ========================================================================
%  Optimizer settings
% =========================================================================

cfg_base.optimizer.verbose = false;
cfg_base.optimizer.store_moments = true;
cfg_base.optimizer.store_full_info = false;

cfg_base.compute_spectrum = false;


%% ========================================================================
%  Initializations
%
%  For EVERY initialization below, two experiments are performed:
%
%  (1) AD-RaNN:
%      p0 -> Adam optimization -> p* -> final unregularized LS
%
%  (2) Fixed RaNN:
%      p = p0 is kept fixed -> NO Adam optimization
%      -> final unregularized LS
%
%  Thus every initialization has a direct fixed-vs-optimized comparison.
% =========================================================================

p0_list = [ ...
     1,  1; ...
     2,  2; ...
     4,  4; ...
     8,  8; ...
     1,  2; ...
     2,  4; ...
     4,  8; ...   % Frequency-based initialization
     6, 12; ...
    10, 20; ...
     8,  4];      % Reversed anisotropy

num_init = size(p0_list,1);


%% ========================================================================
%  Fixed-p baselines
%
%  IMPORTANT:
%  ALL initializations are used as fixed-p baselines.
%
%  This is the key modification:
%       fixed_p_list = p0_list
%
%  Therefore there are 10 fixed baselines, not only (1,1) and (4,8).
% =========================================================================

fixed_p_list = p0_list;
num_fixed = size(fixed_p_list,1);

if num_fixed ~= num_init
    error('Fixed-p baseline list must contain all initializations.');
end


%% ========================================================================
%  Seeds
% =========================================================================

seeds = 1:100;
num_seeds = numel(seeds);


%% ========================================================================
%  Test grid
% =========================================================================

Xtest = tensor_grid( ...
    cfg_base.domain, ...
    cfg_base.test_grid, ...
    0);

ref_test = exact_solution(Xtest);


%% ========================================================================
%  Allocate AD-RaNN results
%
%  row    = initialization
%  column = random seed
% =========================================================================

p1_opt_all = nan(num_init,num_seeds);
p2_opt_all = nan(num_init,num_seeds);

best_iteration_all = nan(num_init,num_seeds);
best_mse_all = nan(num_init,num_seeds);

rel_l2_all = nan(num_init,num_seeds);
rel_linf_all = nan(num_init,num_seeds);

training_time_all = nan(num_init,num_seeds);
total_time_all = nan(num_init,num_seeds);

success_all = false(num_init,num_seeds);
error_message = cell(num_init,num_seeds);


%% ========================================================================
%  Allocate fixed-p baseline results
%
%  NO timing arrays are defined intentionally.
%  Fixed-p baseline cost is NOT included in the reported AD-RaNN runtime.
% =========================================================================

fixed_rel_l2_all = nan(num_fixed,num_seeds);
fixed_rel_linf_all = nan(num_fixed,num_seeds);

fixed_success_all = false(num_fixed,num_seeds);
fixed_error_message = cell(num_fixed,num_seeds);


%% ========================================================================
%  Header
% =========================================================================

fprintf('\n');
fprintf('========================================================================\n');
fprintf('          AD-RaNN initialization x random-seed experiment\n');
fprintf('========================================================================\n');

fprintf('Initializations        = %d\n',num_init);
fprintf('Seeds                  = 1 : 100\n');
fprintf('AD-RaNN runs           = %d\n',num_init*num_seeds);
fprintf('Fixed-p baseline runs  = %d\n',num_fixed*num_seeds);
fprintf('lambda                 = %.3e\n',cfg_base.lambda);
fprintf('Features               = %d\n',cfg_base.num_features);

fprintf('\nInitializations / fixed baselines:\n');

for j = 1:num_init

    fprintf('  %2d: [%.6f, %.6f]\n', ...
        j, ...
        p0_list(j,1), ...
        p0_list(j,2));

end

fprintf('\n');
fprintf('Every initialization above is also evaluated WITHOUT optimization.\n');
fprintf('Fixed-p baseline calculations are excluded from all timing results.\n');
fprintf('========================================================================\n\n');


%% ========================================================================
%  MAIN AD-RaNN EXPERIMENT
%
%  IMPORTANT:
%  This entire experiment is completed and timed FIRST.
%
%  Fixed-p baselines are evaluated only AFTER all AD-RaNN timing is finished.
%  Therefore the baseline calculation cannot affect training_time_all or
%  total_time_all.
% =========================================================================

for j = 1:num_init

    p0 = p0_list(j,:)';

    fprintf('\n');
    fprintf('########################################################################\n');
    fprintf('AD-RaNN initialization %d / %d\n',j,num_init);
    fprintf('Initial p = [%.6f, %.6f]\n',p0(1),p0(2));
    fprintf('########################################################################\n\n');


    for i = 1:num_seeds

        seed_i = seeds(i);

        fprintf( ...
            '[init %2d/%2d | seed %3d/%3d] ... ', ...
            j, ...
            num_init, ...
            seed_i, ...
            num_seeds);

        total_timer = tic;

        try

            %% ----------------------------------------------------------
            %  Configuration
            % -----------------------------------------------------------

            cfg = cfg_base;
            cfg.seed = seed_i;


            %% ----------------------------------------------------------
            %  Random basis
            % -----------------------------------------------------------

            basis = build_random_weights( ...
                cfg.num_features, ...
                cfg.domain, ...
                cfg.seed);


            %% ----------------------------------------------------------
            %  PDE data
            % -----------------------------------------------------------

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


            %% ----------------------------------------------------------
            %  Least-squares options
            % -----------------------------------------------------------

            ls_opts = cfg.linear_solver;
            ls_opts.compute_spectrum = cfg.compute_spectrum;


            %% ----------------------------------------------------------
            %  Reduced objective
            % -----------------------------------------------------------

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
                    poisson_objective_init_seed( ...
                        p, ...
                        problem, ...
                        basis, ...
                        cfg.lambda, ...
                        ls_opts);

            end


            %% ----------------------------------------------------------
            %  Distribution optimization
            % -----------------------------------------------------------

            train_timer = tic;

            [p_opt,history] = ...
                optimize_distribution_adam( ...
                    p0, ...
                    objective_fun, ...
                    cfg.optimizer);

            training_time = toc(train_timer);


            %% ----------------------------------------------------------
            %  Final UNREGULARIZED least-squares solve
            % -----------------------------------------------------------

            [M,y_rhs,~] = ...
                build_system( ...
                    p_opt, ...
                    problem, ...
                    basis);

            [coef,~] = ...
                solve_least_squares( ...
                    M, ...
                    y_rhs, ...
                    cfg.linear_solver);


            %% ----------------------------------------------------------
            %  Test error
            % -----------------------------------------------------------

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


            %% ----------------------------------------------------------
            %  Store AD-RaNN result
            % -----------------------------------------------------------

            p1_opt_all(j,i) = p_opt(1);
            p2_opt_all(j,i) = p_opt(2);

            best_iteration_all(j,i) = ...
                history.best_iteration;

            best_mse_all(j,i) = ...
                history.best_selection_value;

            rel_l2_all(j,i) = err_l2;
            rel_linf_all(j,i) = err_linf;

            training_time_all(j,i) = training_time;

            total_time_all(j,i) = toc(total_timer);

            success_all(j,i) = true;


            %% ----------------------------------------------------------
            %  Print result
            % -----------------------------------------------------------

            fprintf( ...
                ['p*=[%.6f, %.6f] | ', ...
                 'MSE=%.3e | ', ...
                 'L2=%.3e | ', ...
                 'Linf=%.3e | ', ...
                 'checkpoint=%d | ', ...
                 'time=%.3f s\n'], ...
                p_opt(1), ...
                p_opt(2), ...
                history.best_selection_value, ...
                err_l2, ...
                err_linf, ...
                history.best_iteration, ...
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
    %  Summary for current AD-RaNN initialization
    % =====================================================================

    valid = success_all(j,:);

    fprintf('\n');
    fprintf('------------------------------------------------------------------------\n');
    fprintf('AD-RaNN summary for initial p = [%.6f, %.6f]\n', ...
        p0(1),p0(2));
    fprintf('------------------------------------------------------------------------\n');

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
        fprintf('Mean = %.6f\n', ...
            mean(p1_opt_all(j,valid)));
        fprintf('Std  = %.6f\n', ...
            std(p1_opt_all(j,valid)));

        fprintf('\np2*\n');
        fprintf('Mean = %.6f\n', ...
            mean(p2_opt_all(j,valid)));
        fprintf('Std  = %.6f\n', ...
            std(p2_opt_all(j,valid)));

        fprintf('\nMean selected checkpoint = %.3f\n', ...
            mean(best_iteration_all(j,valid)));

        fprintf('Mean training time = %.6f s\n', ...
            mean(training_time_all(j,valid)));

        fprintf('Mean total time    = %.6f s\n', ...
            mean(total_time_all(j,valid)));

    end

    fprintf('------------------------------------------------------------------------\n');

end


%% ========================================================================
%  FIXED-p BASELINES
%
%  ALL 10 initializations are evaluated here.
%
%  NO distribution optimization is performed.
%
%  These calculations occur AFTER all AD-RaNN timings have been stored.
%  Hence they cannot enter the AD-RaNN runtime statistics.
% =========================================================================

fprintf('\n\n');
fprintf('========================================================================\n');
fprintf('              FIXED-p BASELINES -- NO OPTIMIZATION\n');
fprintf('========================================================================\n');

fprintf('Number of fixed distributions = %d\n',num_fixed);

fprintf('\nFixed distributions:\n');

for k = 1:num_fixed

    fprintf('  %2d: [%.6f, %.6f]\n', ...
        k, ...
        fixed_p_list(k,1), ...
        fixed_p_list(k,2));

end

fprintf('\n');
fprintf('Baseline timing is intentionally excluded.\n');
fprintf('========================================================================\n\n');


%% ========================================================================
%  Loop over seeds
%
%  For a given seed, the random basis is generated ONCE and reused for
%  all ten fixed distributions.
%
%  Since build_random_weights is called with the same cfg.seed as in the
%  AD-RaNN experiment, each fixed-p result uses the same random-feature
%  realization as the corresponding AD-RaNN run for that seed.
% =========================================================================

for i = 1:num_seeds

    seed_i = seeds(i);

    %% --------------------------------------------------------------------
    %  Configuration
    % ---------------------------------------------------------------------

    cfg = cfg_base;
    cfg.seed = seed_i;


    %% --------------------------------------------------------------------
    %  Same random basis for this seed
    % ---------------------------------------------------------------------

    basis = build_random_weights( ...
        cfg.num_features, ...
        cfg.domain, ...
        cfg.seed);


    %% --------------------------------------------------------------------
    %  PDE data
    % ---------------------------------------------------------------------

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


    %% --------------------------------------------------------------------
    %  ALL ten fixed distributions
    % ---------------------------------------------------------------------

    for k = 1:num_fixed

        p_fixed = fixed_p_list(k,:)';

        fprintf( ...
            '[fixed %2d/%2d | p=[%5.1f,%5.1f] | seed %3d/%3d] ... ', ...
            k, ...
            num_fixed, ...
            p_fixed(1), ...
            p_fixed(2), ...
            seed_i, ...
            num_seeds);

        try

            %% ----------------------------------------------------------
            %  IMPORTANT:
            %  p_fixed is NEVER passed to Adam.
            %
            %  Directly assemble the PDE system at this distribution.
            % -----------------------------------------------------------

            [M_fixed,y_fixed,~] = ...
                build_system( ...
                    p_fixed, ...
                    problem, ...
                    basis);


            %% ----------------------------------------------------------
            %  Final UNREGULARIZED least-squares solve
            % -----------------------------------------------------------

            [coef_fixed,~] = ...
                solve_least_squares( ...
                    M_fixed, ...
                    y_fixed, ...
                    cfg.linear_solver);


            %% ----------------------------------------------------------
            %  Test error
            % -----------------------------------------------------------

            Phi_test_fixed = gaussian_features( ...
                Xtest, ...
                p_fixed, ...
                basis);

            pred_fixed = Phi_test_fixed*coef_fixed;


            fixed_err_l2 = relative_l2( ...
                pred_fixed, ...
                ref_test);

            fixed_err_linf = relative_linf( ...
                pred_fixed, ...
                ref_test);


            %% ----------------------------------------------------------
            %  Store
            % -----------------------------------------------------------

            fixed_rel_l2_all(k,i) = fixed_err_l2;
            fixed_rel_linf_all(k,i) = fixed_err_linf;

            fixed_success_all(k,i) = true;


            %% ----------------------------------------------------------
            %  Print
            % -----------------------------------------------------------

            fprintf( ...
                'L2=%.3e | Linf=%.3e\n', ...
                fixed_err_l2, ...
                fixed_err_linf);


        catch ME

            fixed_success_all(k,i) = false;

            fixed_error_message{k,i} = ...
                getReport( ...
                    ME, ...
                    'extended', ...
                    'hyperlinks', ...
                    'off');

            fprintf('FAILED\n');
            fprintf('%s\n',fixed_error_message{k,i});

        end

    end

end


%% ========================================================================
%  Fixed-p baseline statistics
% =========================================================================

Fixed_L2_mean = nan(num_fixed,1);
Fixed_L2_std = nan(num_fixed,1);
Fixed_L2_median = nan(num_fixed,1);
Fixed_L2_best = nan(num_fixed,1);
Fixed_L2_worst = nan(num_fixed,1);

Fixed_Linf_mean = nan(num_fixed,1);
Fixed_Linf_std = nan(num_fixed,1);
Fixed_Linf_median = nan(num_fixed,1);
Fixed_Linf_best = nan(num_fixed,1);
Fixed_Linf_worst = nan(num_fixed,1);

Fixed_num_success = zeros(num_fixed,1);


for k = 1:num_fixed

    valid = fixed_success_all(k,:);

    Fixed_num_success(k) = nnz(valid);

    if ~any(valid)
        continue;
    end


    current_l2 = fixed_rel_l2_all(k,valid);
    current_linf = fixed_rel_linf_all(k,valid);


    Fixed_L2_mean(k) = mean(current_l2);
    Fixed_L2_std(k) = std(current_l2);
    Fixed_L2_median(k) = median(current_l2);
    Fixed_L2_best(k) = min(current_l2);
    Fixed_L2_worst(k) = max(current_l2);


    Fixed_Linf_mean(k) = mean(current_linf);
    Fixed_Linf_std(k) = std(current_linf);
    Fixed_Linf_median(k) = median(current_linf);
    Fixed_Linf_best(k) = min(current_linf);
    Fixed_Linf_worst(k) = max(current_linf);

end


%% ========================================================================
%  Print fixed-p baseline summary
% =========================================================================

fprintf('\n\n');
fprintf('========================================================================\n');
fprintf('             FIXED-p FINAL SUMMARY -- NO OPTIMIZATION\n');
fprintf('========================================================================\n');

fprintf('\n');
fprintf(' Fixed p          Mean L2        Std L2         Median L2      Best L2        Worst L2\n');
fprintf('----------------------------------------------------------------------------------------\n');

for k = 1:num_fixed

    fprintf( ...
        '[%5.1f,%5.1f]   %.6e   %.6e   %.6e   %.6e   %.6e\n', ...
        fixed_p_list(k,1), ...
        fixed_p_list(k,2), ...
        Fixed_L2_mean(k), ...
        Fixed_L2_std(k), ...
        Fixed_L2_median(k), ...
        Fixed_L2_best(k), ...
        Fixed_L2_worst(k));

end

fprintf('----------------------------------------------------------------------------------------\n');
fprintf('Timing for all fixed-p baselines is intentionally excluded.\n');
fprintf('========================================================================\n');


%% ========================================================================
%  Final AD-RaNN statistics
% =========================================================================

L2_mean = nan(num_init,1);
L2_std = nan(num_init,1);
L2_median = nan(num_init,1);
L2_best = nan(num_init,1);
L2_worst = nan(num_init,1);

Linf_mean = nan(num_init,1);
Linf_std = nan(num_init,1);
Linf_median = nan(num_init,1);
Linf_best = nan(num_init,1);
Linf_worst = nan(num_init,1);

p1_mean = nan(num_init,1);
p1_std = nan(num_init,1);

p2_mean = nan(num_init,1);
p2_std = nan(num_init,1);

best_iter_mean = nan(num_init,1);

training_mean = nan(num_init,1);
training_std = nan(num_init,1);

total_mean = nan(num_init,1);
total_std = nan(num_init,1);

num_success = zeros(num_init,1);


for j = 1:num_init

    valid = success_all(j,:);

    num_success(j) = nnz(valid);

    if ~any(valid)
        continue;
    end


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


    p1_mean(j) = mean(p1_opt_all(j,valid));
    p1_std(j) = std(p1_opt_all(j,valid));

    p2_mean(j) = mean(p2_opt_all(j,valid));
    p2_std(j) = std(p2_opt_all(j,valid));


    best_iter_mean(j) = ...
        mean(best_iteration_all(j,valid));


    training_mean(j) = ...
        mean(training_time_all(j,valid));

    training_std(j) = ...
        std(training_time_all(j,valid));


    total_mean(j) = ...
        mean(total_time_all(j,valid));

    total_std(j) = ...
        std(total_time_all(j,valid));

end


%% ========================================================================
%  Error-reduction factors
%
%  Fixed / Optimized
%
%  A value > 1 means distribution optimization improves the mean error.
% =========================================================================

L2_reduction_factor = ...
    Fixed_L2_mean ./ L2_mean;

Linf_reduction_factor = ...
    Fixed_Linf_mean ./ Linf_mean;


%% ========================================================================
%  Main combined summary table
% =========================================================================

Initial_p1 = p0_list(:,1);
Initial_p2 = p0_list(:,2);

SummaryTable = table( ...
    Initial_p1, ...
    Initial_p2, ...
    Fixed_num_success, ...
    num_success, ...
    Fixed_L2_mean, ...
    Fixed_L2_std, ...
    L2_mean, ...
    L2_std, ...
    L2_reduction_factor, ...
    Fixed_Linf_mean, ...
    Fixed_Linf_std, ...
    Linf_mean, ...
    Linf_std, ...
    Linf_reduction_factor, ...
    p1_mean, ...
    p1_std, ...
    p2_mean, ...
    p2_std, ...
    best_iter_mean, ...
    training_mean, ...
    training_std, ...
    total_mean, ...
    total_std);


fprintf('\n\n');
fprintf('========================================================================\n');
fprintf('                  COMBINED FINAL SUMMARY\n');
fprintf('========================================================================\n\n');

disp(SummaryTable);


%% ========================================================================
%  Fixed baseline MATLAB table
% =========================================================================

Fixed_p1 = fixed_p_list(:,1);
Fixed_p2 = fixed_p_list(:,2);

FixedBaselineTable = table( ...
    Fixed_p1, ...
    Fixed_p2, ...
    Fixed_num_success, ...
    Fixed_L2_mean, ...
    Fixed_L2_std, ...
    Fixed_L2_median, ...
    Fixed_L2_best, ...
    Fixed_L2_worst, ...
    Fixed_Linf_mean, ...
    Fixed_Linf_std, ...
    Fixed_Linf_median, ...
    Fixed_Linf_best, ...
    Fixed_Linf_worst);


fprintf('\n\n');
fprintf('========================================================================\n');
fprintf('              FIXED-p MATLAB SUMMARY TABLE\n');
fprintf('========================================================================\n\n');

disp(FixedBaselineTable);


%% ========================================================================
%  AD-RaNN MATLAB table
% =========================================================================

ADRannTable = table( ...
    Initial_p1, ...
    Initial_p2, ...
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
    best_iter_mean, ...
    training_mean, ...
    training_std, ...
    total_mean, ...
    total_std);


fprintf('\n\n');
fprintf('========================================================================\n');
fprintf('                    AD-RaNN FINAL SUMMARY\n');
fprintf('========================================================================\n\n');

disp(ADRannTable);


%% ========================================================================
%  Compact paper-style table
%
%  This directly shows:
%
%      p0 -> mean p*
%      Fixed error -> optimized error
% =========================================================================

fprintf('\n');
fprintf('=============================================================================================================================\n');
fprintf(' Initial p -> Mean p*                 Fixed Mean L2 -> AD-RaNN Mean L2       Reduction       Mean ckpt    Train/s    Total/s\n');
fprintf('=============================================================================================================================\n');

for j = 1:num_init

    fprintf( ...
        ['[%5.1f,%5.1f] -> [%6.3f,%6.3f]       ', ...
         '%.3e -> %.3e                 ', ...
         '%9.2e x      ', ...
         '%8.2f    ', ...
         '%8.3f    ', ...
         '%8.3f\n'], ...
        p0_list(j,1), ...
        p0_list(j,2), ...
        p1_mean(j), ...
        p2_mean(j), ...
        Fixed_L2_mean(j), ...
        L2_mean(j), ...
        L2_reduction_factor(j), ...
        best_iter_mean(j), ...
        training_mean(j), ...
        total_mean(j));

end

fprintf('=============================================================================================================================\n');


%% ========================================================================
%  Explicit 10-group fixed -> optimized comparison
% =========================================================================

fprintf('\n\n');
fprintf('========================================================================\n');
fprintf('               EFFECT OF DISTRIBUTION OPTIMIZATION\n');
fprintf('========================================================================\n');


for j = 1:num_init

    fprintf('\n');

    fprintf('Initial p = [%.1f, %.1f]\n', ...
        p0_list(j,1), ...
        p0_list(j,2));

    fprintf('Mean optimized p*         = [%.6f, %.6f]\n', ...
        p1_mean(j), ...
        p2_mean(j));

    fprintf('Fixed p, no optimization  : Mean L2 = %.6e\n', ...
        Fixed_L2_mean(j));

    fprintf('After AD-RaNN optimization: Mean L2 = %.6e\n', ...
        L2_mean(j));

    fprintf('Fixed / AD-RaNN            : %.6e x\n', ...
        L2_reduction_factor(j));

end

fprintf('========================================================================\n');


%% ========================================================================
%  Save all results
% =========================================================================

result_file = fullfile( ...
    seed_dir, ...
    'ad_rann_initialization_100seeds_ALL_fixed_baselines.mat');


save( ...
    result_file, ...
    ...
    'cfg_base', ...
    'p0_list', ...
    'fixed_p_list', ...
    'seeds', ...
    ...
    'p1_opt_all', ...
    'p2_opt_all', ...
    'best_iteration_all', ...
    'best_mse_all', ...
    ...
    'rel_l2_all', ...
    'rel_linf_all', ...
    'training_time_all', ...
    'total_time_all', ...
    'success_all', ...
    'error_message', ...
    ...
    'fixed_rel_l2_all', ...
    'fixed_rel_linf_all', ...
    'fixed_success_all', ...
    'fixed_error_message', ...
    ...
    'L2_mean', ...
    'L2_std', ...
    'L2_median', ...
    'L2_best', ...
    'L2_worst', ...
    ...
    'Linf_mean', ...
    'Linf_std', ...
    'Linf_median', ...
    'Linf_best', ...
    'Linf_worst', ...
    ...
    'Fixed_L2_mean', ...
    'Fixed_L2_std', ...
    'Fixed_L2_median', ...
    'Fixed_L2_best', ...
    'Fixed_L2_worst', ...
    ...
    'Fixed_Linf_mean', ...
    'Fixed_Linf_std', ...
    'Fixed_Linf_median', ...
    'Fixed_Linf_best', ...
    'Fixed_Linf_worst', ...
    ...
    'L2_reduction_factor', ...
    'Linf_reduction_factor', ...
    ...
    'p1_mean', ...
    'p1_std', ...
    'p2_mean', ...
    'p2_std', ...
    ...
    'best_iter_mean', ...
    'training_mean', ...
    'training_std', ...
    'total_mean', ...
    'total_std', ...
    ...
    'SummaryTable', ...
    'FixedBaselineTable', ...
    'ADRannTable');


fprintf('\nResults saved to:\n%s\n\n',result_file);


%% ========================================================================
%  Save combined summary CSV
% =========================================================================

csv_file = fullfile( ...
    seed_dir, ...
    'ad_rann_initialization_100seeds_ALL_fixed_baselines.csv');

writetable( ...
    SummaryTable, ...
    csv_file);

fprintf('Combined summary CSV saved to:\n%s\n\n',csv_file);


%% ========================================================================
%  Labels
% =========================================================================

init_labels = { ...
    '(1,1)', ...
    '(2,2)', ...
    '(4,4)', ...
    '(8,8)', ...
    '(1,2)', ...
    '(2,4)', ...
    '(4,8)', ...
    '(6,12)', ...
    '(10,20)', ...
    '(8,4)'};


%% ========================================================================
%  Figure 1:
%  Optimized AD-RaNN error boxplot
% =========================================================================

figure;

error_data = [];
group_data = [];

for j = 1:num_init

    valid = success_all(j,:);

    values = rel_l2_all(j,valid)';

    error_data = [ ...
        error_data; ...
        values];

    group_data = [ ...
        group_data; ...
        j*ones(numel(values),1)];

end


boxchart( ...
    group_data, ...
    error_data);

set(gca, ...
    'YScale','log', ...
    'FontSize',12, ...
    'LineWidth',1.0);

grid off;
box on;

xticks(1:num_init);
xticklabels(init_labels);

xlabel( ...
    'Initial parameter $\mathbf{p}_0$', ...
    'Interpreter','latex', ...
    'FontSize',14);

ylabel( ...
    'Relative $\ell_2$ error', ...
    'Interpreter','latex', ...
    'FontSize',14);

title( ...
    'AD-RaNN initialization robustness over 100 random seeds', ...
    'FontSize',12);


%% ========================================================================
%  Figure 2:
%  Fixed vs optimized mean relative L2
%
%  No error bars are used here because mean - std can become negative,
%  which causes warnings on a logarithmic axis.
% =========================================================================

figure;

semilogy( ...
    1:num_init, ...
    Fixed_L2_mean, ...
    's-', ...
    'LineWidth',1.2, ...
    'MarkerSize',6);

hold on;

semilogy( ...
    1:num_init, ...
    L2_mean, ...
    'o-', ...
    'LineWidth',1.2, ...
    'MarkerSize',6);

hold off;

set(gca, ...
    'FontSize',12, ...
    'LineWidth',1.0);

grid off;
box on;

xticks(1:num_init);
xticklabels(init_labels);

xlabel( ...
    'Initial parameter $\mathbf{p}_0$', ...
    'Interpreter','latex', ...
    'FontSize',14);

ylabel( ...
    'Mean relative $\ell_2$ error', ...
    'Interpreter','latex', ...
    'FontSize',14);

legend( ...
    'Fixed distribution', ...
    'After AD-RaNN optimization', ...
    'Location','best');

title( ...
    'Effect of distribution optimization', ...
    'FontSize',12);


%% ========================================================================
%  Figure 3:
%  Mean AD-RaNN runtime
%
%  Fixed-p baseline cost is NOT included.
% =========================================================================

figure;

plot( ...
    1:num_init, ...
    total_mean, ...
    'o-', ...
    'LineWidth',1.2, ...
    'MarkerSize',6);

set(gca, ...
    'FontSize',12, ...
    'LineWidth',1.0);

grid off;
box on;

xticks(1:num_init);
xticklabels(init_labels);

xlabel( ...
    'Initial parameter $\mathbf{p}_0$', ...
    'Interpreter','latex', ...
    'FontSize',14);

ylabel( ...
    'Mean total time / s', ...
    'FontSize',14);

title( ...
    'Mean AD-RaNN runtime over 100 random seeds', ...
    'FontSize',12);


%% ========================================================================
%  Generic objective
% =========================================================================

function [obj,grad,info] = ...
    poisson_objective_init_seed( ...
        p, ...
        problem, ...
        basis, ...
        lambda, ...
        ls_opts)

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