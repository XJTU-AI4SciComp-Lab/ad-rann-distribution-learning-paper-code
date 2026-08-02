clear;
clc;

warning('off','MATLAB:rankDeficientMatrix');

%% ========================================================================
%  Path
% =========================================================================

this_file = mfilename('fullpath');
seed_dir = fileparts(this_file);
root = fileparts(fileparts(seed_dir));
example_dir = fullfile(root,'examples','poisson_2d');
output_dir = fullfile(root,'results','poisson_2d','seed_total');

if exist(output_dir,'dir') ~= 7
    mkdir(output_dir);
end

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
            gpu_id,gpuDeviceCount);
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
%  IMPORTANT:
%  For EVERY p0 below we compute both:
%
%  1. Fixed RaNN:
%       p = p0, no distribution optimization,
%       followed directly by final unregularized LS.
%
%  2. AD-RaNN:
%       start from p0, optimize p,
%       then perform final unregularized LS.
%
%  Therefore every row has a direct fixed-vs-adaptive comparison.
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
%  Allocate: optimized AD-RaNN results
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
%  Allocate: fixed-p baseline results
%
%  These are the errors obtained with p = p0 and NO Adam optimization.
%
%  No timing array is introduced for the fixed-p baseline because its cost
%  is deliberately excluded from the reported AD-RaNN runtime.
% =========================================================================

fixed_rel_l2_all = nan(num_init,num_seeds);
fixed_rel_linf_all = nan(num_init,num_seeds);

fixed_success_all = false(num_init,num_seeds);
fixed_error_message = cell(num_init,num_seeds);


%% ========================================================================
%  Header
% =========================================================================

fprintf('\n');
fprintf('========================================================================\n');
fprintf('          AD-RaNN initialization x random-seed experiment\n');
fprintf('========================================================================\n');

fprintf('Initializations       = %d\n',num_init);
fprintf('Seeds                 = 1 : 100\n');
fprintf('AD-RaNN runs          = %d\n',num_init*num_seeds);
fprintf('Fixed-p baseline runs = %d\n',num_init*num_seeds);
fprintf('lambda                = %.3e\n',cfg_base.lambda);
fprintf('Features              = %d\n',cfg_base.num_features);

fprintf('\nInitial p list:\n');

for j = 1:num_init

    fprintf('  %2d: [%.6f, %.6f]\n', ...
        j, ...
        p0_list(j,1), ...
        p0_list(j,2));

end

fprintf('\n');
fprintf('For every p0, the corresponding fixed-p baseline is also evaluated.\n');
fprintf('Fixed-p baseline cost is EXCLUDED from all reported timing.\n');

fprintf('========================================================================\n\n');


%% ========================================================================
%  Main experiment
% =========================================================================

for j = 1:num_init

    p0 = p0_list(j,:)';

    fprintf('\n');
    fprintf('########################################################################\n');
    fprintf('Initialization %d / %d\n',j,num_init);
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


        %% ----------------------------------------------------------------
        %  Flags
        % -----------------------------------------------------------------

        baseline_ready = false;
        ad_success = false;


        %% ----------------------------------------------------------------
        %  IMPORTANT:
        %
        %  The timer starts here and measures the ORIGINAL AD-RaNN workflow.
        %
        %  The fixed-p baseline is evaluated only AFTER total_time_all(j,i)
        %  has already been stored.
        %
        %  Therefore fixed-p evaluation cannot enter the reported runtime.
        % -----------------------------------------------------------------

        total_timer = tic;


        try

            %% ============================================================
            %  Configuration
            % =============================================================

            cfg = cfg_base;
            cfg.seed = seed_i;


            %% ============================================================
            %  Random basis
            % =============================================================

            basis = build_random_weights( ...
                cfg.num_features, ...
                cfg.domain, ...
                cfg.seed);


            %% ============================================================
            %  PDE data
            % =============================================================

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


            %% ============================================================
            %  The fixed baseline can now be evaluated later using
            %  exactly the SAME basis and PDE data.
            % =============================================================

            baseline_ready = true;


            %% ============================================================
            %  Least-squares options
            % =============================================================

            ls_opts = cfg.linear_solver;
            ls_opts.compute_spectrum = cfg.compute_spectrum;


            %% ============================================================
            %  Reduced objective
            % =============================================================

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


            %% ============================================================
            %  Distribution optimization
            % =============================================================

            train_timer = tic;

            [p_opt,history] = ...
                optimize_distribution_adam( ...
                    p0, ...
                    objective_fun, ...
                    cfg.optimizer);

            training_time = toc(train_timer);


            %% ============================================================
            %  Final UNREGULARIZED least-squares solve
            % =============================================================

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


            %% ============================================================
            %  Optimized AD-RaNN error
            % =============================================================

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


            %% ============================================================
            %  Store AD-RaNN results
            % =============================================================

            p1_opt_all(j,i) = p_opt(1);
            p2_opt_all(j,i) = p_opt(2);

            best_iteration_all(j,i) = ...
                history.best_iteration;

            best_mse_all(j,i) = ...
                history.best_selection_value;

            rel_l2_all(j,i) = err_l2;
            rel_linf_all(j,i) = err_linf;

            training_time_all(j,i) = ...
                training_time;


            %% ============================================================
            %  STOP THE REPORTED AD-RaNN TIME HERE
            %
            %  Everything below this point is excluded from total_time_all.
            % =============================================================

            total_time_all(j,i) = ...
                toc(total_timer);

            success_all(j,i) = true;
            ad_success = true;


        catch ME

            success_all(j,i) = false;

            total_time_all(j,i) = ...
                toc(total_timer);

            error_message{j,i} = ...
                getReport( ...
                    ME, ...
                    'extended', ...
                    'hyperlinks', ...
                    'off');

        end


        %% =================================================================
        %  Fixed-p baseline
        %
        %  p is held exactly at p0.
        %
        %  NO distribution optimization.
        %  NO ridge optimization.
        %
        %  We directly perform the same final UNREGULARIZED LS solve.
        %
        %  This is computed AFTER the AD-RaNN time was already recorded.
        % ==================================================================

        if baseline_ready

            try

                %% --------------------------------------------------------
                %  Fixed p = original initialization p0
                % ---------------------------------------------------------

                p_fixed = p0;


                %% --------------------------------------------------------
                %  Assemble with fixed distribution
                % ---------------------------------------------------------

                [M_fixed,y_fixed,~] = ...
                    build_system( ...
                        p_fixed, ...
                        problem, ...
                        basis);


                %% --------------------------------------------------------
                %  Unregularized LS
                % ---------------------------------------------------------

                [coef_fixed,~] = ...
                    solve_least_squares( ...
                        M_fixed, ...
                        y_fixed, ...
                        cfg.linear_solver);


                %% --------------------------------------------------------
                %  Fixed-p error
                % ---------------------------------------------------------

                Phi_test_fixed = gaussian_features( ...
                    Xtest, ...
                    p_fixed, ...
                    basis);

                pred_fixed = ...
                    Phi_test_fixed*coef_fixed;


                fixed_err_l2 = relative_l2( ...
                    pred_fixed, ...
                    ref_test);

                fixed_err_linf = relative_linf( ...
                    pred_fixed, ...
                    ref_test);


                %% --------------------------------------------------------
                %  Store baseline
                % ---------------------------------------------------------

                fixed_rel_l2_all(j,i) = ...
                    fixed_err_l2;

                fixed_rel_linf_all(j,i) = ...
                    fixed_err_linf;

                fixed_success_all(j,i) = true;


            catch ME_fixed

                fixed_success_all(j,i) = false;

                fixed_error_message{j,i} = ...
                    getReport( ...
                        ME_fixed, ...
                        'extended', ...
                        'hyperlinks', ...
                        'off');

            end

        end


        %% =================================================================
        %  Print result
        % ==================================================================

        if ad_success

            if fixed_success_all(j,i)

                fprintf( ...
                    ['p*=[%.6f, %.6f] | ', ...
                     'Opt L2=%.3e | ', ...
                     'Fixed L2=%.3e | ', ...
                     'Linf=%.3e | ', ...
                     'checkpoint=%d | ', ...
                     'time=%.3f s\n'], ...
                    p1_opt_all(j,i), ...
                    p2_opt_all(j,i), ...
                    rel_l2_all(j,i), ...
                    fixed_rel_l2_all(j,i), ...
                    rel_linf_all(j,i), ...
                    best_iteration_all(j,i), ...
                    total_time_all(j,i));

            else

                fprintf( ...
                    ['p*=[%.6f, %.6f] | ', ...
                     'Opt L2=%.3e | ', ...
                     'Fixed L2=FAILED | ', ...
                     'Linf=%.3e | ', ...
                     'checkpoint=%d | ', ...
                     'time=%.3f s\n'], ...
                    p1_opt_all(j,i), ...
                    p2_opt_all(j,i), ...
                    rel_l2_all(j,i), ...
                    rel_linf_all(j,i), ...
                    best_iteration_all(j,i), ...
                    total_time_all(j,i));

            end

        else

            if fixed_success_all(j,i)

                fprintf( ...
                    'AD-RaNN FAILED | Fixed L2=%.3e\n', ...
                    fixed_rel_l2_all(j,i));

            else

                fprintf('AD-RaNN FAILED | Fixed baseline FAILED\n');

            end

            fprintf('%s\n',error_message{j,i});

        end

    end


    %% ====================================================================
    %  Summary for current initialization
    % =====================================================================

    valid_opt = success_all(j,:);
    valid_fixed = fixed_success_all(j,:);

    fprintf('\n');
    fprintf('------------------------------------------------------------------------\n');
    fprintf('Summary for initial p = [%.6f, %.6f]\n', ...
        p0(1), ...
        p0(2));
    fprintf('------------------------------------------------------------------------\n');

    fprintf('Successful AD-RaNN runs = %d / %d\n', ...
        nnz(valid_opt), ...
        num_seeds);

    fprintf('Successful fixed-p runs = %d / %d\n', ...
        nnz(valid_fixed), ...
        num_seeds);


    if any(valid_fixed)

        current_fixed_l2 = ...
            fixed_rel_l2_all(j,valid_fixed);

        current_fixed_linf = ...
            fixed_rel_linf_all(j,valid_fixed);

        fprintf('\nFIXED p = [%.6f, %.6f], NO optimization\n', ...
            p0(1), ...
            p0(2));

        fprintf('Relative L2\n');
        fprintf('Mean   = %.6e\n',mean(current_fixed_l2));
        fprintf('Std    = %.6e\n',std(current_fixed_l2));
        fprintf('Median = %.6e\n',median(current_fixed_l2));
        fprintf('Best   = %.6e\n',min(current_fixed_l2));
        fprintf('Worst  = %.6e\n',max(current_fixed_l2));

        fprintf('\nRelative Linf\n');
        fprintf('Mean   = %.6e\n',mean(current_fixed_linf));
        fprintf('Std    = %.6e\n',std(current_fixed_linf));
        fprintf('Median = %.6e\n',median(current_fixed_linf));
        fprintf('Best   = %.6e\n',min(current_fixed_linf));
        fprintf('Worst  = %.6e\n',max(current_fixed_linf));

    end


    if any(valid_opt)

        current_l2 = ...
            rel_l2_all(j,valid_opt);

        current_linf = ...
            rel_linf_all(j,valid_opt);

        fprintf('\nAFTER AD-RaNN optimization\n');

        fprintf('Relative L2\n');
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
            mean(p1_opt_all(j,valid_opt)));

        fprintf('Std  = %.6f\n', ...
            std(p1_opt_all(j,valid_opt)));

        fprintf('\np2*\n');
        fprintf('Mean = %.6f\n', ...
            mean(p2_opt_all(j,valid_opt)));

        fprintf('Std  = %.6f\n', ...
            std(p2_opt_all(j,valid_opt)));

        fprintf('\nMean selected checkpoint = %.3f\n', ...
            mean(best_iteration_all(j,valid_opt)));

        fprintf('Mean training time = %.6f s\n', ...
            mean(training_time_all(j,valid_opt)));

        fprintf('Mean total time    = %.6f s\n', ...
            mean(total_time_all(j,valid_opt)));

    end


    if any(valid_fixed) && any(valid_opt)

        fixed_mean_tmp = ...
            mean(fixed_rel_l2_all(j,valid_fixed));

        opt_mean_tmp = ...
            mean(rel_l2_all(j,valid_opt));

        fprintf('\nRatio of mean L2 errors\n');

        fprintf( ...
            'Fixed / AD-RaNN = %.6e x\n', ...
            fixed_mean_tmp/opt_mean_tmp);

    end

    fprintf('------------------------------------------------------------------------\n');

end


%% ========================================================================
%  Final statistics: optimized AD-RaNN
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


%% ========================================================================
%  Final statistics: fixed-p baseline
% =========================================================================

Fixed_L2_mean = nan(num_init,1);
Fixed_L2_std = nan(num_init,1);
Fixed_L2_median = nan(num_init,1);
Fixed_L2_best = nan(num_init,1);
Fixed_L2_worst = nan(num_init,1);

Fixed_Linf_mean = nan(num_init,1);
Fixed_Linf_std = nan(num_init,1);
Fixed_Linf_median = nan(num_init,1);
Fixed_Linf_best = nan(num_init,1);
Fixed_Linf_worst = nan(num_init,1);

fixed_num_success = zeros(num_init,1);


%% ========================================================================
%  Compute all statistics
% =========================================================================

for j = 1:num_init

    %% --------------------------------------------------------------------
    %  Optimized
    % ---------------------------------------------------------------------

    valid_opt = success_all(j,:);

    num_success(j) = nnz(valid_opt);

    if any(valid_opt)

        L2_mean(j) = ...
            mean(rel_l2_all(j,valid_opt));

        L2_std(j) = ...
            std(rel_l2_all(j,valid_opt));

        L2_median(j) = ...
            median(rel_l2_all(j,valid_opt));

        L2_best(j) = ...
            min(rel_l2_all(j,valid_opt));

        L2_worst(j) = ...
            max(rel_l2_all(j,valid_opt));


        Linf_mean(j) = ...
            mean(rel_linf_all(j,valid_opt));

        Linf_std(j) = ...
            std(rel_linf_all(j,valid_opt));

        Linf_median(j) = ...
            median(rel_linf_all(j,valid_opt));

        Linf_best(j) = ...
            min(rel_linf_all(j,valid_opt));

        Linf_worst(j) = ...
            max(rel_linf_all(j,valid_opt));


        p1_mean(j) = ...
            mean(p1_opt_all(j,valid_opt));

        p1_std(j) = ...
            std(p1_opt_all(j,valid_opt));

        p2_mean(j) = ...
            mean(p2_opt_all(j,valid_opt));

        p2_std(j) = ...
            std(p2_opt_all(j,valid_opt));


        best_iter_mean(j) = ...
            mean(best_iteration_all(j,valid_opt));


        training_mean(j) = ...
            mean(training_time_all(j,valid_opt));

        training_std(j) = ...
            std(training_time_all(j,valid_opt));


        total_mean(j) = ...
            mean(total_time_all(j,valid_opt));

        total_std(j) = ...
            std(total_time_all(j,valid_opt));

    end


    %% --------------------------------------------------------------------
    %  Fixed baseline
    % ---------------------------------------------------------------------

    valid_fixed = fixed_success_all(j,:);

    fixed_num_success(j) = ...
        nnz(valid_fixed);

    if any(valid_fixed)

        Fixed_L2_mean(j) = ...
            mean(fixed_rel_l2_all(j,valid_fixed));

        Fixed_L2_std(j) = ...
            std(fixed_rel_l2_all(j,valid_fixed));

        Fixed_L2_median(j) = ...
            median(fixed_rel_l2_all(j,valid_fixed));

        Fixed_L2_best(j) = ...
            min(fixed_rel_l2_all(j,valid_fixed));

        Fixed_L2_worst(j) = ...
            max(fixed_rel_l2_all(j,valid_fixed));


        Fixed_Linf_mean(j) = ...
            mean(fixed_rel_linf_all(j,valid_fixed));

        Fixed_Linf_std(j) = ...
            std(fixed_rel_linf_all(j,valid_fixed));

        Fixed_Linf_median(j) = ...
            median(fixed_rel_linf_all(j,valid_fixed));

        Fixed_Linf_best(j) = ...
            min(fixed_rel_linf_all(j,valid_fixed));

        Fixed_Linf_worst(j) = ...
            max(fixed_rel_linf_all(j,valid_fixed));

    end

end


%% ========================================================================
%  Error-reduction factor
%
%       reduction = mean fixed error / mean optimized error
%
%  > 1 means AD-RaNN improves over the corresponding fixed distribution.
% =========================================================================

L2_reduction_factor = ...
    Fixed_L2_mean ./ L2_mean;

Linf_reduction_factor = ...
    Fixed_Linf_mean ./ Linf_mean;


%% ========================================================================
%  Summary table
% =========================================================================

Initial_p1 = p0_list(:,1);
Initial_p2 = p0_list(:,2);

SummaryTable = table( ...
    Initial_p1, ...
    Initial_p2, ...
    fixed_num_success, ...
    num_success, ...
    Fixed_L2_mean, ...
    Fixed_L2_std, ...
    L2_mean, ...
    L2_std, ...
    L2_reduction_factor, ...
    Fixed_Linf_mean, ...
    Linf_mean, ...
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


%% ========================================================================
%  Print final summary table
% =========================================================================

fprintf('\n\n');
fprintf('========================================================================\n');
fprintf('                         FINAL SUMMARY\n');
fprintf('========================================================================\n\n');

disp(SummaryTable);


%% ========================================================================
%  Detailed fixed-p summary
% =========================================================================

fprintf('\n');
fprintf('========================================================================\n');
fprintf('             FIXED-p BASELINE SUMMARY -- NO OPTIMIZATION\n');
fprintf('========================================================================\n');

for j = 1:num_init

    fprintf('\n');
    fprintf('p = [%.1f, %.1f]\n', ...
        p0_list(j,1), ...
        p0_list(j,2));

    fprintf('Mean L2   = %.6e\n', ...
        Fixed_L2_mean(j));

    fprintf('Std L2    = %.6e\n', ...
        Fixed_L2_std(j));

    fprintf('Median L2 = %.6e\n', ...
        Fixed_L2_median(j));

    fprintf('Best L2   = %.6e\n', ...
        Fixed_L2_best(j));

    fprintf('Worst L2  = %.6e\n', ...
        Fixed_L2_worst(j));

end

fprintf('\n');
fprintf('Fixed-p baseline timing is intentionally NOT reported.\n');
fprintf('========================================================================\n');


%% ========================================================================
%  Compact paper-style comparison
% =========================================================================

fprintf('\n');
fprintf('====================================================================================================================\n');
fprintf(' Initial p       Fixed Mean L2    AD-RaNN Mean L2    Reduction      Mean p*              Mean ckpt    Train/s   Total/s\n');
fprintf('====================================================================================================================\n');

for j = 1:num_init

    fprintf( ...
        ['[%5.1f,%5.1f]   ', ...
         '%.3e        ', ...
         '%.3e          ', ...
         '%9.2e x    ', ...
         '[%6.3f,%6.3f]      ', ...
         '%8.2f    ', ...
         '%7.3f   ', ...
         '%7.3f\n'], ...
        p0_list(j,1), ...
        p0_list(j,2), ...
        Fixed_L2_mean(j), ...
        L2_mean(j), ...
        L2_reduction_factor(j), ...
        p1_mean(j), ...
        p2_mean(j), ...
        best_iter_mean(j), ...
        training_mean(j), ...
        total_mean(j));

end

fprintf('====================================================================================================================\n');


%% ========================================================================
%  Explicit fixed-vs-optimized comparison
% =========================================================================

fprintf('\n\n');
fprintf('========================================================================\n');
fprintf('                 EFFECT OF DISTRIBUTION OPTIMIZATION\n');
fprintf('========================================================================\n');


for j = 1:num_init

    fprintf('\n');

    fprintf('Initial p = [%.1f, %.1f]\n', ...
        p0_list(j,1), ...
        p0_list(j,2));

    fprintf('Fixed p, no optimization  : Mean L2 = %.6e\n', ...
        Fixed_L2_mean(j));

    fprintf('After AD-RaNN optimization: Mean L2 = %.6e\n', ...
        L2_mean(j));

    fprintf('Ratio Fixed / AD-RaNN      : %.6e x\n', ...
        L2_reduction_factor(j));

end

fprintf('========================================================================\n');


%% ========================================================================
%  Save results
% =========================================================================

result_file = fullfile( ...
    output_dir, ...
    'ad_rann_initialization_100seeds_all_fixed_baselines.mat');


save( ...
    result_file, ...
    'cfg_base', ...
    'p0_list', ...
    'seeds', ...
    ...
    'p1_opt_all', ...
    'p2_opt_all', ...
    'best_iteration_all', ...
    'best_mse_all', ...
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
    'best_iter_mean', ...
    'training_mean', ...
    'training_std', ...
    'total_mean', ...
    'total_std', ...
    ...
    'SummaryTable');


fprintf('\nResults saved to:\n%s\n',result_file);


%% ========================================================================
%  Save summary CSV
% =========================================================================

csv_file = fullfile( ...
    output_dir, ...
    'ad_rann_initialization_100seeds_all_fixed_baselines.csv');

writetable( ...
    SummaryTable, ...
    csv_file);

fprintf('\nSummary CSV saved to:\n%s\n\n',csv_file);


%% ========================================================================
%  Figure labels
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
%  Figure 1: optimized AD-RaNN error boxplot
% =========================================================================

figure;

error_data = [];
group_data = [];

for j = 1:num_init

    valid = success_all(j,:);

    values = ...
        rel_l2_all(j,valid)';

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
    'Initial parameter $\bm p_0$', ...
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
%  Figure 2: fixed vs optimized mean L2
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
    'Distribution parameter', ...
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
%  Figure 3: mean AD-RaNN runtime
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
    'Initial parameter $\bm p_0$', ...
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
