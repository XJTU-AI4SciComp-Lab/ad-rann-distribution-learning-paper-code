function results = activation_study_core(cfg)
%ACTIVATION_STUDY_CORE Tanh/sin AD-RaNN activation experiment.
%
% This file contains the experiment bookkeeping so that
% run_activation_experiment.m stays intentionally short.
%
% Existing project functions are reused for:
%   random basis generation,
%   Poisson cache,
%   tanh/sin fast reduced evaluators,
%   Adam optimization,
%   final system assembly,
%   least-squares solves,
%   test features and error metrics.
%
% All activations start from the same p0=(4,8), so the only intended
% experimental factor changed here is the activation function.
%
% There is NO fast/generic gradient consistency gate.

output_dir = cfg.study.output_dir;

if exist(output_dir,'dir') ~= 7
    mkdir(output_dir);
end

%% ========================================================================
% Required existing project functions
% =========================================================================
required = { ...
    'build_random_weights', ...
    'tensor_grid', ...
    'rhs', ...
    'exact_solution', ...
    'prepare_poisson_cache', ...
    'evaluate_poisson_reduced_fast_tanh', ...
    'evaluate_poisson_reduced_fast_sin', ...
    'build_system_activation', ...
    'optimize_distribution_adam', ...
    'solve_least_squares', ...
    'relative_l2', ...
    'relative_linf', ...
    'tanh_features', ...
    'sin_features'};

for k = 1:numel(required)

    if exist(required{k},'file') ~= 2
        error( ...
            'Required existing project function not found: %s.m', ...
            required{k});
    end
end

%% ========================================================================
% GPU information
% =========================================================================
if cfg.linear_solver.use_gpu

    try

        gpu_id = cfg.linear_solver.gpu_id;

        if gpu_id > gpuDeviceCount

            error( ...
                'GPU %d is unavailable; MATLAB sees %d GPU(s).', ...
                gpu_id, ...
                gpuDeviceCount);

        end

        g = gpuDevice(gpu_id);

        fprintf('\n');
        fprintf('Using GPU %d: %s\n',g.Index,g.Name);
        fprintf('Available memory: %.2f GB\n', ...
            g.AvailableMemory/1024^3);

    catch ME

        warning( ...
            'GPU setup failed: %s\nSwitching to CPU least squares.', ...
            ME.message);

        cfg.linear_solver.use_gpu = false;

    end
end

%% ========================================================================
% Common initialization for all activations
%
% This is an activation-function ablation.  To isolate the effect of the
% activation, tanh and sin start from exactly the same p0=(4,8).
% =========================================================================
activation_list = cfg.study.activations(:);
num_activation = numel(activation_list);

p0 = cfg.study.initial_p(:);

if numel(p0) ~= 2
    error('cfg.study.initial_p must contain exactly two parameters.');
end

if any(p0 < cfg.optimizer.lower_bound(:)) || ...
        any(p0 > cfg.optimizer.upper_bound(:))
    error('Common p0 lies outside the Adam bounds.');
end

p0_all = repmat(p0,1,num_activation);

fprintf('\n');
fprintf('========================================================================\n');
fprintf('COMMON INITIALIZATION FOR ACTIVATION ABLATION\n');
fprintf('========================================================================\n');
fprintf('All activations use p0 = [%.6f, %.6f]\n',p0(1),p0(2));
fprintf('No activation-specific frequency initialization is used.\n');
fprintf('========================================================================\n');

%% ========================================================================
% Common deterministic data
% =========================================================================
Xtest = tensor_grid( ...
    cfg.domain, ...
    cfg.test_grid, ...
    0);

ref_test = exact_solution(Xtest);

problem = make_poisson_problem_local(cfg);

seeds = cfg.study.seeds(:)';
num_seeds = numel(seeds);

%% ========================================================================
% Allocate arrays
% =========================================================================
p1_opt_all = nan(num_activation,num_seeds);
p2_opt_all = nan(num_activation,num_seeds);

selected_checkpoint_all = nan(num_activation,num_seeds);
best_mse_all = nan(num_activation,num_seeds);

rel_l2_all = nan(num_activation,num_seeds);
rel_linf_all = nan(num_activation,num_seeds);

fixed_rel_l2_all = nan(num_activation,num_seeds);
fixed_rel_linf_all = nan(num_activation,num_seeds);

optimization_time_all = nan(num_activation,num_seeds);
total_time_all = nan(num_activation,num_seeds);

success_all = false(num_activation,num_seeds);
fixed_success_all = false(num_activation,num_seeds);
completed_all = false(num_activation,num_seeds);

error_message = cell(num_activation,num_seeds);
fixed_error_message = cell(num_activation,num_seeds);

checkpoint_file = fullfile( ...
    output_dir, ...
    'activation_study_checkpoint.mat');

%% ========================================================================
% Resume
% =========================================================================
if cfg.study.resume && exist(checkpoint_file,'file') == 2

    try

        S = load(checkpoint_file,'state');

        if isfield(S,'state')

            state = S.state;

            compatible = ...
                isfield(state,'activation_list') && ...
                isfield(state,'seeds') && ...
                isequal(state.activation_list(:),activation_list(:)) && ...
                isequal(state.seeds(:),seeds(:));

            if compatible

                p1_opt_all = state.p1_opt_all;
                p2_opt_all = state.p2_opt_all;

                selected_checkpoint_all = ...
                    state.selected_checkpoint_all;

                best_mse_all = ...
                    state.best_mse_all;

                rel_l2_all = state.rel_l2_all;
                rel_linf_all = state.rel_linf_all;

                fixed_rel_l2_all = ...
                    state.fixed_rel_l2_all;

                fixed_rel_linf_all = ...
                    state.fixed_rel_linf_all;

                optimization_time_all = ...
                    state.optimization_time_all;

                total_time_all = ...
                    state.total_time_all;

                success_all = state.success_all;
                fixed_success_all = state.fixed_success_all;
                completed_all = state.completed_all;

                error_message = state.error_message;
                fixed_error_message = state.fixed_error_message;

                fprintf( ...
                    '\nResume checkpoint loaded: %d/%d runs complete.\n', ...
                    nnz(completed_all), ...
                    num_activation*num_seeds);
            end
        end

    catch ME

        warning( ...
            'Checkpoint could not be loaded; starting fresh: %s', ...
            ME.message);
    end
end

%% ========================================================================
% Header
% =========================================================================
fprintf('\n');
fprintf('========================================================================\n');
fprintf('AD-RaNN ACTIVATION STUDY\n');
fprintf('========================================================================\n');
fprintf('Activations              = tanh, sin\n');
fprintf('Seeds per activation     = %d\n',num_seeds);
fprintf('Features                 = %d\n',cfg.num_features);
fprintf('lambda                   = %.3e\n',cfg.lambda);
fprintf('Adam updates             = %d (no early stopping)\n', ...
    cfg.optimizer.maxit);
fprintf('Adam learning rate       = %.3f\n', ...
    cfg.optimizer.learning_rate);
fprintf('Common initialization      = [%.3f, %.3f]\n',p0(1),p0(2));
fprintf('Gradient consistency gate= OFF\n');
fprintf('Fixed baseline timing    = excluded\n');
fprintf('========================================================================\n');

%% ========================================================================
% Main study
% =========================================================================
for a = 1:num_activation

    activation = lower(activation_list{a});
    p0 = p0_all(:,a);

    fprintf('\n');
    fprintf('########################################################################\n');
    fprintf('Activation %d/%d: %s\n', ...
        a, ...
        num_activation, ...
        upper(activation));
    fprintf('Common p0 = [%.6f, %.6f]\n', ...
        p0(1), ...
        p0(2));
    fprintf('########################################################################\n');

    for i = 1:num_seeds

        seed_i = seeds(i);

        if completed_all(a,i)

            fprintf( ...
                '[%s | seed %3d/%3d] completed -> skipped\n', ...
                upper(activation), ...
                seed_i, ...
                num_seeds);

            continue;
        end

        fprintf( ...
            '[%s | seed %3d/%3d] ... ', ...
            upper(activation), ...
            seed_i, ...
            num_seeds);

        %% ----------------------------------------------------------------
        % Timed AD-RaNN run
        % -----------------------------------------------------------------
        total_timer = tic;

        try

            cfg_run = cfg;
            cfg_run.activation = activation;
            cfg_run.seed = seed_i;

            basis = build_random_weights( ...
                cfg_run.num_features, ...
                cfg_run.domain, ...
                cfg_run.seed);

            cache = prepare_poisson_cache( ...
                problem, ...
                basis);

            ls_opts = cfg_run.linear_solver;
            ls_opts.compute_spectrum = false;

            switch activation

                case 'tanh'

                    objective_fun = @(p) ...
                        evaluate_poisson_reduced_fast_tanh( ...
                            p, ...
                            cache, ...
                            cfg_run.lambda, ...
                            ls_opts);

                case 'sin'

                    objective_fun = @(p) ...
                        evaluate_poisson_reduced_fast_sin( ...
                            p, ...
                            cache, ...
                            cfg_run.lambda, ...
                            ls_opts);
            end

            optimization_timer = tic;

            [p_opt,history] = ...
                optimize_distribution_adam( ...
                    p0, ...
                    objective_fun, ...
                    cfg_run.optimizer);

            optimization_time = ...
                toc(optimization_timer);

            % Final unregularized LS refit.
            [M,y_rhs,~] = ...
                build_system_activation( ...
                    p_opt, ...
                    problem, ...
                    basis, ...
                    activation);

            [coef,~] = ...
                solve_least_squares( ...
                    M, ...
                    y_rhs, ...
                    cfg_run.linear_solver);

            Phi_test = ...
                test_features_local( ...
                    Xtest, ...
                    p_opt, ...
                    basis, ...
                    activation);

            pred = Phi_test*coef;

            err_l2 = ...
                relative_l2(pred,ref_test);

            err_linf = ...
                relative_linf(pred,ref_test);

            p1_opt_all(a,i) = p_opt(1);
            p2_opt_all(a,i) = p_opt(2);

            selected_checkpoint_all(a,i) = ...
                history.best_iteration;

            best_mse_all(a,i) = ...
                history.best_selection_value;

            rel_l2_all(a,i) = err_l2;
            rel_linf_all(a,i) = err_linf;

            optimization_time_all(a,i) = ...
                optimization_time;

            total_time_all(a,i) = ...
                toc(total_timer);

            success_all(a,i) = true;

        catch ME

            total_time_all(a,i) = toc(total_timer);

            success_all(a,i) = false;

            error_message{a,i} = ...
                getReport( ...
                    ME, ...
                    'extended', ...
                    'hyperlinks', ...
                    'off');
        end

        %% ----------------------------------------------------------------
        % Fixed-p baseline, outside reported AD-RaNN timer
        % -----------------------------------------------------------------
        if cfg.study.compute_fixed_baseline

            try

                cfg_fixed = cfg;
                cfg_fixed.activation = activation;
                cfg_fixed.seed = seed_i;

                basis_fixed = ...
                    build_random_weights( ...
                        cfg_fixed.num_features, ...
                        cfg_fixed.domain, ...
                        cfg_fixed.seed);

                [M_fixed,y_fixed,~] = ...
                    build_system_activation( ...
                        p0, ...
                        problem, ...
                        basis_fixed, ...
                        activation);

                [coef_fixed,~] = ...
                    solve_least_squares( ...
                        M_fixed, ...
                        y_fixed, ...
                        cfg_fixed.linear_solver);

                Phi_fixed = ...
                    test_features_local( ...
                        Xtest, ...
                        p0, ...
                        basis_fixed, ...
                        activation);

                pred_fixed = ...
                    Phi_fixed*coef_fixed;

                fixed_rel_l2_all(a,i) = ...
                    relative_l2( ...
                        pred_fixed, ...
                        ref_test);

                fixed_rel_linf_all(a,i) = ...
                    relative_linf( ...
                        pred_fixed, ...
                        ref_test);

                fixed_success_all(a,i) = true;

            catch ME_fixed

                fixed_success_all(a,i) = false;

                fixed_error_message{a,i} = ...
                    getReport( ...
                        ME_fixed, ...
                        'extended', ...
                        'hyperlinks', ...
                        'off');
            end

        else

            fixed_success_all(a,i) = true;
        end

        completed_all(a,i) = ...
            success_all(a,i) && ...
            fixed_success_all(a,i);

        %% ----------------------------------------------------------------
        % Print
        % -----------------------------------------------------------------
        if success_all(a,i) && fixed_success_all(a,i)

            if cfg.study.compute_fixed_baseline

                fprintf( ...
                    ['p*=[%.6f, %.6f] | ', ...
                     'Fixed L2=%.3e -> Opt L2=%.3e | ', ...
                     'ckpt=%d | opt=%.3f s | total=%.3f s\n'], ...
                    p1_opt_all(a,i), ...
                    p2_opt_all(a,i), ...
                    fixed_rel_l2_all(a,i), ...
                    rel_l2_all(a,i), ...
                    selected_checkpoint_all(a,i), ...
                    optimization_time_all(a,i), ...
                    total_time_all(a,i));

            else

                fprintf( ...
                    ['p*=[%.6f, %.6f] | ', ...
                     'Opt L2=%.3e | ckpt=%d | ', ...
                     'opt=%.3f s | total=%.3f s\n'], ...
                    p1_opt_all(a,i), ...
                    p2_opt_all(a,i), ...
                    rel_l2_all(a,i), ...
                    selected_checkpoint_all(a,i), ...
                    optimization_time_all(a,i), ...
                    total_time_all(a,i));
            end

        else

            fprintf('FAILED\n');

            if ~success_all(a,i)
                fprintf('%s\n',error_message{a,i});
            end

            if ~fixed_success_all(a,i)
                fprintf('%s\n',fixed_error_message{a,i});
            end
        end

        %% ----------------------------------------------------------------
        % Checkpoint
        % -----------------------------------------------------------------
        state = struct();

        state.activation_list = activation_list;
        state.seeds = seeds;

        state.p1_opt_all = p1_opt_all;
        state.p2_opt_all = p2_opt_all;

        state.selected_checkpoint_all = ...
            selected_checkpoint_all;

        state.best_mse_all = ...
            best_mse_all;

        state.rel_l2_all = ...
            rel_l2_all;

        state.rel_linf_all = ...
            rel_linf_all;

        state.fixed_rel_l2_all = ...
            fixed_rel_l2_all;

        state.fixed_rel_linf_all = ...
            fixed_rel_linf_all;

        state.optimization_time_all = ...
            optimization_time_all;

        state.total_time_all = ...
            total_time_all;

        state.success_all = success_all;
        state.fixed_success_all = fixed_success_all;
        state.completed_all = completed_all;

        state.error_message = error_message;
        state.fixed_error_message = fixed_error_message;

        save(checkpoint_file,'state');

        clear basis basis_fixed cache objective_fun;
        clear M y_rhs coef Phi_test pred;
        clear M_fixed y_fixed coef_fixed Phi_fixed pred_fixed;
    end
end

%% ========================================================================
% Statistics
% =========================================================================
num_success = zeros(num_activation,1);
fixed_num_success = zeros(num_activation,1);

L2_mean = nan(num_activation,1);
L2_std = nan(num_activation,1);
L2_median = nan(num_activation,1);
L2_best = nan(num_activation,1);
L2_worst = nan(num_activation,1);

Linf_mean = nan(num_activation,1);
Linf_std = nan(num_activation,1);

Fixed_L2_mean = nan(num_activation,1);
Fixed_L2_std = nan(num_activation,1);

p1_mean = nan(num_activation,1);
p1_std = nan(num_activation,1);

p2_mean = nan(num_activation,1);
p2_std = nan(num_activation,1);

selected_checkpoint_mean = ...
    nan(num_activation,1);

optimization_mean = nan(num_activation,1);
optimization_std = nan(num_activation,1);

total_mean = nan(num_activation,1);
total_std = nan(num_activation,1);

for a = 1:num_activation

    valid = success_all(a,:);
    valid_fixed = fixed_success_all(a,:);

    num_success(a) = nnz(valid);
    fixed_num_success(a) = nnz(valid_fixed);

    if any(valid)

        values = rel_l2_all(a,valid);

        L2_mean(a) = mean(values);
        L2_std(a) = std(values);
        L2_median(a) = median(values);
        L2_best(a) = min(values);
        L2_worst(a) = max(values);

        Linf_mean(a) = ...
            mean(rel_linf_all(a,valid));

        Linf_std(a) = ...
            std(rel_linf_all(a,valid));

        p1_mean(a) = ...
            mean(p1_opt_all(a,valid));

        p1_std(a) = ...
            std(p1_opt_all(a,valid));

        p2_mean(a) = ...
            mean(p2_opt_all(a,valid));

        p2_std(a) = ...
            std(p2_opt_all(a,valid));

        selected_checkpoint_mean(a) = ...
            mean(selected_checkpoint_all(a,valid));

        optimization_mean(a) = ...
            mean(optimization_time_all(a,valid));

        optimization_std(a) = ...
            std(optimization_time_all(a,valid));

        total_mean(a) = ...
            mean(total_time_all(a,valid));

        total_std(a) = ...
            std(total_time_all(a,valid));
    end

    if cfg.study.compute_fixed_baseline && any(valid_fixed)

        Fixed_L2_mean(a) = ...
            mean(fixed_rel_l2_all(a,valid_fixed));

        Fixed_L2_std(a) = ...
            std(fixed_rel_l2_all(a,valid_fixed));
    end
end

L2_reduction_factor = ...
    Fixed_L2_mean ./ L2_mean;

%% ========================================================================
% Final summary
% =========================================================================
Activation = activation_list;

Initial_p1 = p0_all(1,:)';
Initial_p2 = p0_all(2,:)';

ActivationSummaryTable = table( ...
    Activation, ...
    Initial_p1, ...
    Initial_p2, ...
    num_success, ...
    fixed_num_success, ...
    Fixed_L2_mean, ...
    Fixed_L2_std, ...
    L2_mean, ...
    L2_std, ...
    L2_median, ...
    L2_best, ...
    L2_worst, ...
    L2_reduction_factor, ...
    Linf_mean, ...
    Linf_std, ...
    p1_mean, ...
    p1_std, ...
    p2_mean, ...
    p2_std, ...
    selected_checkpoint_mean, ...
    optimization_mean, ...
    optimization_std, ...
    total_mean, ...
    total_std);

fprintf('\n\n');
fprintf('========================================================================\n');
fprintf('FINAL ACTIVATION SUMMARY\n');
fprintf('========================================================================\n\n');

disp(ActivationSummaryTable);

fprintf('\n');
fprintf('================================================================================================================\n');
fprintf(' Activation   p0 -> Mean p*                  Fixed Mean L2 -> AD-RaNN Mean L2    Reduction   Mean ckpt   Opt/s   Total/s\n');
fprintf('================================================================================================================\n');

for a = 1:num_activation

    fprintf( ...
        [' %-8s ', ...
         '[%5.1f,%5.1f] -> [%7.3f,%7.3f]    ', ...
         '%.3e -> %.3e        ', ...
         '%8.2e x   ', ...
         '%7.2f   ', ...
         '%6.3f  ', ...
         '%7.3f\n'], ...
        activation_list{a}, ...
        p0_all(1,a), ...
        p0_all(2,a), ...
        p1_mean(a), ...
        p2_mean(a), ...
        Fixed_L2_mean(a), ...
        L2_mean(a), ...
        L2_reduction_factor(a), ...
        selected_checkpoint_mean(a), ...
        optimization_mean(a), ...
        total_mean(a));
end

fprintf('================================================================================================================\n');

%% ========================================================================
% Save
% =========================================================================
results = struct();

results.cfg = cfg;
results.activation_list = activation_list;
results.seeds = seeds;

results.p0_all = p0_all;

results.p1_opt_all = p1_opt_all;
results.p2_opt_all = p2_opt_all;

results.selected_checkpoint_all = ...
    selected_checkpoint_all;

results.best_mse_all = best_mse_all;

results.rel_l2_all = rel_l2_all;
results.rel_linf_all = rel_linf_all;

results.fixed_rel_l2_all = fixed_rel_l2_all;
results.fixed_rel_linf_all = fixed_rel_linf_all;

results.optimization_time_all = ...
    optimization_time_all;

results.total_time_all = ...
    total_time_all;

results.success_all = success_all;
results.fixed_success_all = fixed_success_all;
results.completed_all = completed_all;

results.L2_mean = L2_mean;
results.L2_std = L2_std;
results.L2_median = L2_median;
results.L2_best = L2_best;
results.L2_worst = L2_worst;

results.Linf_mean = Linf_mean;
results.Linf_std = Linf_std;

results.Fixed_L2_mean = Fixed_L2_mean;
results.Fixed_L2_std = Fixed_L2_std;

results.L2_reduction_factor = ...
    L2_reduction_factor;

results.p1_mean = p1_mean;
results.p1_std = p1_std;
results.p2_mean = p2_mean;
results.p2_std = p2_std;

results.selected_checkpoint_mean = ...
    selected_checkpoint_mean;

results.optimization_mean = ...
    optimization_mean;

results.optimization_std = ...
    optimization_std;

results.total_mean = ...
    total_mean;

results.total_std = ...
    total_std;

results.ActivationSummaryTable = ...
    ActivationSummaryTable;

result_file = fullfile( ...
    output_dir, ...
    'activation_study_tanh_sin_results.mat');

csv_file = fullfile( ...
    output_dir, ...
    'activation_study_tanh_sin_summary.csv');

save(result_file,'results');

writetable( ...
    ActivationSummaryTable, ...
    csv_file);

fprintf('\nResults saved to:\n%s\n',result_file);
fprintf('\nCSV saved to:\n%s\n',csv_file);

end


%% =========================================================================
% Local Poisson problem
% =========================================================================
function problem = make_poisson_problem_local(cfg)

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

end


%% =========================================================================
% Local activation-specific test features
% =========================================================================
function Phi = test_features_local( ...
        X, ...
        p, ...
        basis, ...
        activation)

switch activation

    case 'tanh'

        Phi = tanh_features( ...
            X, ...
            p, ...
            basis);

    case 'sin'

        Phi = sin_features( ...
            X, ...
            p, ...
            basis);

    otherwise

        error( ...
            'Unsupported activation: %s', ...
            activation);
end

end
