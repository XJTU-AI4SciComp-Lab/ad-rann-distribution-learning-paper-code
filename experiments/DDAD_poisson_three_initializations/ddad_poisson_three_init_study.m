function results = ddad_poisson_three_init_study(cfg)
%DDAD_POISSON_THREE_INIT_STUDY
%
% Three-initialization DDAD study:
%
%   p0 = (1,1), (4,4), (4,8)
%
% Correct DDAD logic:
%
%   noisy solution data
%       -> optimize p by data-driven reduced problem
%       -> select historical best p*
%       -> return to ORIGINAL Poisson PDE
%       -> final unregularized PDE least-squares solve at p*
%       -> report final PDE relative L2/Linf error
%
% Timing convention:
%
%   OptimizationTime = optimize_distribution_adam(...) only
%   PDESolveTime      = final build_system + unregularized LS only
%   MethodTime        = OptimizationTime + PDESolveTime
%
% NOT included:
%   random point generation, exact solution, noise generation/addition,
%   feature generation, cache construction, diagnostics, test error,
%   fixed baseline, statistics, checkpoint/file I/O.

output_dir = cfg.ddad.output_dir;

if exist(output_dir,'dir') ~= 7
    mkdir(output_dir);
end

warning('off','MATLAB:rankDeficientMatrix');

%% ========================================================================
% Required functions from the existing project
% =========================================================================
required = { ...
    'build_random_weights', ...
    'prepare_data_cache', ...
    'evaluate_data_reduced_fast', ...
    'build_data_system', ...
    'optimize_distribution_adam', ...
    'build_system', ...
    'solve_least_squares', ...
    'gaussian_features', ...
    'relative_l2', ...
    'relative_linf', ...
    'tensor_grid', ...
    'exact_solution', ...
    'rhs'};

for k = 1:numel(required)
    if exist(required{k},'file') ~= 2
        error('Required project function not found: %s.m',required{k});
    end
end

%% ========================================================================
% GPU
% =========================================================================
if isfield(cfg,'linear_solver') && ...
        isfield(cfg.linear_solver,'use_gpu') && ...
        cfg.linear_solver.use_gpu

    try
        gpu_id = cfg.linear_solver.gpu_id;

        if gpu_id > gpuDeviceCount
            error('GPU %d unavailable; MATLAB sees %d GPU(s).', ...
                gpu_id,gpuDeviceCount);
        end

        g = gpuDevice(gpu_id);

        fprintf('\nUsing GPU %d: %s\n',g.Index,g.Name);
        fprintf('Available memory: %.2f GB\n', ...
            g.AvailableMemory/1024^3);

    catch ME
        warning('GPU setup failed: %s\nFalling back to CPU.',ME.message);
        cfg.linear_solver.use_gpu = false;
    end
end

%% ========================================================================
% Constants
% =========================================================================
initial_p_list = cfg.ddad.initial_p_list;

if size(initial_p_list,2) ~= 2
    error('cfg.ddad.initial_p_list must be nInit-by-2.');
end

num_init = size(initial_p_list,1);

noise_levels = cfg.ddad.noise_levels(:);
num_noise = numel(noise_levels);

seeds = cfg.ddad.seeds(:)';
num_seeds = numel(seeds);

Ndata = cfg.ddad.num_data_points;
activation = cfg.activation;

%% ========================================================================
% Build ORIGINAL Poisson PDE data once
% =========================================================================
problem = struct();

problem.domain = cfg.domain;
problem.boundary_penalty = cfg.boundary_penalty;

problem.Xi = tensor_grid( ...
    cfg.domain, ...
    cfg.interior_grid, ...
    1e-6);

problem.fi = rhs(problem.Xi);

nB = cfg.boundary_points_per_side;

xB = linspace( ...
    cfg.domain(1,1), ...
    cfg.domain(1,2), ...
    nB)';

yB = linspace( ...
    cfg.domain(2,1), ...
    cfg.domain(2,2), ...
    nB)';

problem.Xb = [ ...
    cfg.domain(1,1)*ones(nB,1), yB; ...
    cfg.domain(1,2)*ones(nB,1), yB; ...
    xB, cfg.domain(2,1)*ones(nB,1); ...
    xB, cfg.domain(2,2)*ones(nB,1)];

problem.gb = exact_solution(problem.Xb);

problem.y = [ ...
    problem.fi; ...
    cfg.boundary_penalty*problem.gb];

%% ========================================================================
% Exact test grid
% =========================================================================
Xtest = tensor_grid( ...
    cfg.domain, ...
    cfg.test_grid, ...
    0);

u_test_exact = exact_solution(Xtest);

%% ========================================================================
% Allocate
%
% Dimensions:
%   first  = initialization
%   second = noise level
%   third  = seed
% =========================================================================
sz = [num_init,num_noise,num_seeds];

p1_opt_all = nan(sz);
p2_opt_all = nan(sz);

selected_checkpoint_all = nan(sz);
best_data_mse_all = nan(sz);

target_rel_l2_all = nan(sz);
data_fit_rel_l2_all = nan(sz);

fixed_pde_rel_l2_all = nan(sz);

pde_rel_l2_all = nan(sz);
pde_rel_linf_all = nan(sz);

optimization_time_all = nan(sz);
pde_solve_time_all = nan(sz);
method_time_all = nan(sz);

success_all = false(sz);
completed_all = false(sz);

error_message = cell(sz);

checkpoint_file = fullfile( ...
    output_dir, ...
    'ddad_three_init_checkpoint.mat');

%% ========================================================================
% Resume
% =========================================================================
if cfg.ddad.resume && exist(checkpoint_file,'file') == 2

    try
        S = load(checkpoint_file,'state');

        if isfield(S,'state')
            state = S.state;

            compatible = ...
                isfield(state,'initial_p_list') && ...
                isfield(state,'noise_levels') && ...
                isfield(state,'seeds') && ...
                isequal(state.initial_p_list,initial_p_list) && ...
                isequal(state.noise_levels(:),noise_levels(:)) && ...
                isequal(state.seeds(:),seeds(:));

            if compatible
                p1_opt_all = state.p1_opt_all;
                p2_opt_all = state.p2_opt_all;

                selected_checkpoint_all = state.selected_checkpoint_all;
                best_data_mse_all = state.best_data_mse_all;

                target_rel_l2_all = state.target_rel_l2_all;
                data_fit_rel_l2_all = state.data_fit_rel_l2_all;

                fixed_pde_rel_l2_all = state.fixed_pde_rel_l2_all;

                pde_rel_l2_all = state.pde_rel_l2_all;
                pde_rel_linf_all = state.pde_rel_linf_all;

                optimization_time_all = state.optimization_time_all;
                pde_solve_time_all = state.pde_solve_time_all;
                method_time_all = state.method_time_all;

                success_all = state.success_all;
                completed_all = state.completed_all;
                error_message = state.error_message;

                fprintf('\nCheckpoint loaded: %d/%d successful runs complete.\n', ...
                    nnz(completed_all),num_init*num_noise*num_seeds);
            end
        end

    catch ME
        warning('Checkpoint load failed; starting fresh: %s',ME.message);
    end
end

%% ========================================================================
% Header
% =========================================================================
fprintf('\n');
fprintf('====================================================================================\n');
fprintf('DDAD POISSON -- THREE INITIALIZATIONS -- FINAL ERROR FROM ORIGINAL PDE\n');
fprintf('====================================================================================\n');
fprintf('Initializations          = (1,1), (4,4), (4,8)\n');
fprintf('Noise amplitudes         = %s\n',mat2str(noise_levels.'));
fprintf('Seeds                    = %d:%d\n',seeds(1),seeds(end));
fprintf('Random data points       = %d / seed\n',Ndata);
fprintf('Randomized features      = %d\n',cfg.num_features);
fprintf('Activation               = %s\n',activation);
fprintf('DDAD ridge lambda        = %.3e\n',cfg.lambda);
fprintf('Adam updates             = %d\n',cfg.optimizer.maxit);
fprintf('Adam learning rate       = %.3f\n',cfg.optimizer.learning_rate);
fprintf('FINAL ERROR              = ORIGINAL PDE least-squares solution\n');
fprintf('Method time              = Adam + final PDE solve only\n');
fprintf('====================================================================================\n\n');

%% ========================================================================
% Main loop
%
% Seed is outermost so ALL p0/delta cases use the same Xdata, eta and basis.
% =========================================================================
for i = 1:num_seeds

    seed_i = seeds(i);

    if all(reshape(completed_all(:,:,i),[],1))
        fprintf('[seed %3d/%3d] all cases completed -> skipped\n', ...
            seed_i,num_seeds);
        continue;
    end

    %% --------------------------------------------------------------------
    % Shared random realization for this seed -- NOT TIMED
    % ---------------------------------------------------------------------
    rng(seed_i,'twister');

    Xdata = zeros(Ndata,2);

    Xdata(:,1) = ...
        cfg.domain(1,1) + ...
        (cfg.domain(1,2)-cfg.domain(1,1))*rand(Ndata,1);

    Xdata(:,2) = ...
        cfg.domain(2,1) + ...
        (cfg.domain(2,2)-cfg.domain(2,1))*rand(Ndata,1);

    u_data_exact = exact_solution(Xdata);

    eta = rand(Ndata,1);

    basis = build_random_weights( ...
        cfg.num_features, ...
        cfg.domain, ...
        seed_i);

    %% --------------------------------------------------------------------
    % Initialization loop
    % ---------------------------------------------------------------------
    for q = 1:num_init

        p0 = initial_p_list(q,:).';

        %% ----------------------------------------------------------------
        % "Before DDAD" = ORIGINAL-PDE solution using fixed p0.
        %
        % This baseline does NOT depend on delta, so compute once per
        % initialization/seed and copy it across noise levels.
        % NOT included in DDAD method time.
        % -----------------------------------------------------------------
        fixed_pde_l2_this = NaN;

        if cfg.ddad.compute_fixed_pde_baseline

            [M_fixed,y_fixed,~] = ...
                build_system( ...
                    p0, ...
                    problem, ...
                    basis);

            [coef_fixed,~] = ...
                solve_least_squares( ...
                    M_fixed, ...
                    y_fixed, ...
                    cfg.linear_solver);

            Phi_fixed_test = ...
                gaussian_features( ...
                    Xtest, ...
                    p0, ...
                    basis);

            u_fixed_test = ...
                Phi_fixed_test*coef_fixed;

            fixed_pde_l2_this = ...
                relative_l2( ...
                    u_fixed_test, ...
                    u_test_exact);
        end

        %% ----------------------------------------------------------------
        % Noise loop
        % -----------------------------------------------------------------
        for a = 1:num_noise

            delta = noise_levels(a);

            if completed_all(q,a,i)
                fprintf( ...
                    '[p0=(%g,%g) | delta=%8.1e | seed %3d/%3d] completed -> skipped\n', ...
                    p0(1),p0(2),delta,seed_i,num_seeds);
                continue;
            end

            fprintf( ...
                '[p0=(%g,%g) | delta=%8.1e | seed %3d/%3d] ... ', ...
                p0(1),p0(2),delta,seed_i,num_seeds);

            try
                %% --------------------------------------------------------
                % Noisy target -- NOT TIMED
                % ---------------------------------------------------------
                ydata = ...
                    u_data_exact + ...
                    delta*eta;

                target_rel_l2 = ...
                    relative_l2( ...
                        ydata, ...
                        u_data_exact);

                %% --------------------------------------------------------
                % Data-driven reduced objective -- NOT TIMED
                % ---------------------------------------------------------
                cache_data = ...
                    prepare_data_cache( ...
                        Xdata, ...
                        ydata, ...
                        basis);

                ls_opts = cfg.linear_solver;
                ls_opts.compute_spectrum = false;

                objective_fun = @(p) ...
                    evaluate_data_reduced_fast( ...
                        p, ...
                        cache_data, ...
                        cfg.lambda, ...
                        ls_opts, ...
                        activation);

                [obj0,grad0,~] = objective_fun(p0);

                if ~isfinite(obj0) || any(~isfinite(grad0(:)))
                    error('DDAD reduced objective/gradient is non-finite at p0.');
                end

                %% --------------------------------------------------------
                % TIMING 1: Adam only
                % ---------------------------------------------------------
                t_opt = tic;

                [p_opt,history] = ...
                    optimize_distribution_adam( ...
                        p0, ...
                        objective_fun, ...
                        cfg.optimizer);

                optimization_time = toc(t_opt);

                %% --------------------------------------------------------
                % Data-fit diagnostic -- NOT final PDE solution
                % NOT TIMED
                % ---------------------------------------------------------
                [M_data,y_data_final] = ...
                    build_data_system( ...
                        p_opt, ...
                        cache_data, ...
                        activation);

                [coef_data,~] = ...
                    solve_least_squares( ...
                        M_data, ...
                        y_data_final, ...
                        cfg.linear_solver);

                pred_data = M_data*coef_data;

                data_fit_rel_l2 = ...
                    relative_l2( ...
                        pred_data, ...
                        ydata);

                %% --------------------------------------------------------
                % TIMING 2: final ORIGINAL-PDE solve at p*
                % ---------------------------------------------------------
                t_pde = tic;

                [M_pde,y_pde,~] = ...
                    build_system( ...
                        p_opt, ...
                        problem, ...
                        basis);

                [coef_pde,~] = ...
                    solve_least_squares( ...
                        M_pde, ...
                        y_pde, ...
                        cfg.linear_solver);

                pde_solve_time = toc(t_pde);

                method_time = ...
                    optimization_time + ...
                    pde_solve_time;

                %% --------------------------------------------------------
                % Final PDE test error -- NOT TIMED
                % ---------------------------------------------------------
                Phi_test = ...
                    gaussian_features( ...
                        Xtest, ...
                        p_opt, ...
                        basis);

                u_pde_test = ...
                    Phi_test*coef_pde;

                pde_l2 = ...
                    relative_l2( ...
                        u_pde_test, ...
                        u_test_exact);

                pde_linf = ...
                    relative_linf( ...
                        u_pde_test, ...
                        u_test_exact);

                %% --------------------------------------------------------
                % Store
                % ---------------------------------------------------------
                p1_opt_all(q,a,i) = p_opt(1);
                p2_opt_all(q,a,i) = p_opt(2);

                selected_checkpoint_all(q,a,i) = ...
                    history.best_iteration;

                best_data_mse_all(q,a,i) = ...
                    history.best_selection_value;

                target_rel_l2_all(q,a,i) = ...
                    target_rel_l2;

                data_fit_rel_l2_all(q,a,i) = ...
                    data_fit_rel_l2;

                fixed_pde_rel_l2_all(q,a,i) = ...
                    fixed_pde_l2_this;

                pde_rel_l2_all(q,a,i) = ...
                    pde_l2;

                pde_rel_linf_all(q,a,i) = ...
                    pde_linf;

                optimization_time_all(q,a,i) = ...
                    optimization_time;

                pde_solve_time_all(q,a,i) = ...
                    pde_solve_time;

                method_time_all(q,a,i) = ...
                    method_time;

                success_all(q,a,i) = true;
                completed_all(q,a,i) = true;
                error_message{q,a,i} = '';

                fprintf( ...
                    ['target=%.3e | p*=[%.5f,%.5f] | ', ...
                     'PDE before=%.3e -> after=%.3e | ', ...
                     'data-fit=%.3e | ckpt=%d | ', ...
                     'Adam=%.3f s | PDE=%.3f s | method=%.3f s\n'], ...
                    target_rel_l2, ...
                    p_opt(1), ...
                    p_opt(2), ...
                    fixed_pde_l2_this, ...
                    pde_l2, ...
                    data_fit_rel_l2, ...
                    history.best_iteration, ...
                    optimization_time, ...
                    pde_solve_time, ...
                    method_time);

            catch ME

                success_all(q,a,i) = false;
                completed_all(q,a,i) = false;

                error_message{q,a,i} = ...
                    getReport( ...
                        ME, ...
                        'extended', ...
                        'hyperlinks', ...
                        'off');

                fprintf('\nFAILED: p0=(%g,%g), delta=%.3e, seed=%d\n', ...
                    p0(1),p0(2),delta,seed_i);

                fprintf('%s\n',error_message{q,a,i});

                state = make_state();
                save(checkpoint_file,'state','-v7.3');

                if cfg.ddad.stop_on_error
                    rethrow(ME);
                end
            end

            %% ------------------------------------------------------------
            % Checkpoint
            % -------------------------------------------------------------
            state = make_state();
            save(checkpoint_file,'state','-v7.3');

            clear objective_fun cache_data;
            clear M_data y_data_final coef_data pred_data;
            clear M_pde y_pde coef_pde Phi_test u_pde_test;
        end

        clear M_fixed y_fixed coef_fixed Phi_fixed_test u_fixed_test;
    end

    clear basis Xdata u_data_exact eta;
end

%% ========================================================================
% Summary statistics: one row per (initialization, noise)
% =========================================================================
n_summary = num_init*num_noise;

Initial_p1 = nan(n_summary,1);
Initial_p2 = nan(n_summary,1);
NoiseAmplitude = nan(n_summary,1);
num_success = zeros(n_summary,1);

TargetRelL2_mean = nan(n_summary,1);

DataFitRelL2_mean = nan(n_summary,1);

FixedPDEL2_mean = nan(n_summary,1);
FixedPDEL2_std = nan(n_summary,1);

PDEL2_mean = nan(n_summary,1);
PDEL2_std = nan(n_summary,1);
PDEL2_median = nan(n_summary,1);
PDEL2_best = nan(n_summary,1);
PDEL2_worst = nan(n_summary,1);

PDEL2_reduction_factor = nan(n_summary,1);

PDELinf_mean = nan(n_summary,1);

p1_mean = nan(n_summary,1);
p1_std = nan(n_summary,1);
p2_mean = nan(n_summary,1);
p2_std = nan(n_summary,1);

selected_checkpoint_mean = nan(n_summary,1);

OptimizationTime_mean = nan(n_summary,1);
OptimizationTime_std = nan(n_summary,1);

PDESolveTime_mean = nan(n_summary,1);
PDESolveTime_std = nan(n_summary,1);

MethodTime_mean = nan(n_summary,1);
MethodTime_std = nan(n_summary,1);

row = 0;

for q = 1:num_init

    for a = 1:num_noise

        row = row+1;

        Initial_p1(row) = initial_p_list(q,1);
        Initial_p2(row) = initial_p_list(q,2);
        NoiseAmplitude(row) = noise_levels(a);

        valid = reshape(success_all(q,a,:),1,[]);

        num_success(row) = nnz(valid);

        if ~any(valid)
            continue;
        end

        target_vals = reshape(target_rel_l2_all(q,a,valid),1,[]);
        datafit_vals = reshape(data_fit_rel_l2_all(q,a,valid),1,[]);
        fixed_vals = reshape(fixed_pde_rel_l2_all(q,a,valid),1,[]);
        pde_vals = reshape(pde_rel_l2_all(q,a,valid),1,[]);
        linf_vals = reshape(pde_rel_linf_all(q,a,valid),1,[]);

        p1_vals = reshape(p1_opt_all(q,a,valid),1,[]);
        p2_vals = reshape(p2_opt_all(q,a,valid),1,[]);

        ckpt_vals = reshape(selected_checkpoint_all(q,a,valid),1,[]);

        opt_vals = reshape(optimization_time_all(q,a,valid),1,[]);
        pdet_vals = reshape(pde_solve_time_all(q,a,valid),1,[]);
        method_vals = reshape(method_time_all(q,a,valid),1,[]);

        TargetRelL2_mean(row) = mean(target_vals);

        DataFitRelL2_mean(row) = mean(datafit_vals);

        FixedPDEL2_mean(row) = mean(fixed_vals);
        FixedPDEL2_std(row) = std(fixed_vals);

        PDEL2_mean(row) = mean(pde_vals);
        PDEL2_std(row) = std(pde_vals);
        PDEL2_median(row) = median(pde_vals);
        PDEL2_best(row) = min(pde_vals);
        PDEL2_worst(row) = max(pde_vals);

        PDEL2_reduction_factor(row) = ...
            FixedPDEL2_mean(row) / ...
            PDEL2_mean(row);

        PDELinf_mean(row) = mean(linf_vals);

        p1_mean(row) = mean(p1_vals);
        p1_std(row) = std(p1_vals);

        p2_mean(row) = mean(p2_vals);
        p2_std(row) = std(p2_vals);

        selected_checkpoint_mean(row) = mean(ckpt_vals);

        OptimizationTime_mean(row) = mean(opt_vals);
        OptimizationTime_std(row) = std(opt_vals);

        PDESolveTime_mean(row) = mean(pdet_vals);
        PDESolveTime_std(row) = std(pdet_vals);

        MethodTime_mean(row) = mean(method_vals);
        MethodTime_std(row) = std(method_vals);
    end
end

SummaryTable = table( ...
    Initial_p1, ...
    Initial_p2, ...
    NoiseAmplitude, ...
    num_success, ...
    TargetRelL2_mean, ...
    DataFitRelL2_mean, ...
    FixedPDEL2_mean, ...
    FixedPDEL2_std, ...
    PDEL2_mean, ...
    PDEL2_std, ...
    PDEL2_median, ...
    PDEL2_best, ...
    PDEL2_worst, ...
    PDEL2_reduction_factor, ...
    PDELinf_mean, ...
    p1_mean, ...
    p1_std, ...
    p2_mean, ...
    p2_std, ...
    selected_checkpoint_mean, ...
    OptimizationTime_mean, ...
    OptimizationTime_std, ...
    PDESolveTime_mean, ...
    PDESolveTime_std, ...
    MethodTime_mean, ...
    MethodTime_std);

%% ========================================================================
% Compact printed summary
% =========================================================================
fprintf('\n\n');
fprintf('================================================================================================================================================\n');
fprintf('THREE-INITIALIZATION DDAD FINAL SUMMARY\n');
fprintf('FINAL ERROR = ORIGINAL POISSON PDE LEAST-SQUARES SOLUTION\n');
fprintf('================================================================================================================================================\n');
fprintf(' p0       delta      success   mean p*             PDE before -> after        improve     mean ckpt   Adam/s   PDE/s   method/s\n');
fprintf('================================================================================================================================================\n');

for r = 1:height(SummaryTable)

    fprintf( ...
        '(%g,%g)   %8.1e   %3d/%3d   [%7.3f,%7.3f]   %.3e -> %.3e   %8.2fx    %7.2f    %6.3f   %6.3f   %7.3f\n', ...
        SummaryTable.Initial_p1(r), ...
        SummaryTable.Initial_p2(r), ...
        SummaryTable.NoiseAmplitude(r), ...
        SummaryTable.num_success(r), ...
        num_seeds, ...
        SummaryTable.p1_mean(r), ...
        SummaryTable.p2_mean(r), ...
        SummaryTable.FixedPDEL2_mean(r), ...
        SummaryTable.PDEL2_mean(r), ...
        SummaryTable.PDEL2_reduction_factor(r), ...
        SummaryTable.selected_checkpoint_mean(r), ...
        SummaryTable.OptimizationTime_mean(r), ...
        SummaryTable.PDESolveTime_mean(r), ...
        SummaryTable.MethodTime_mean(r));
end

fprintf('================================================================================================================================================\n\n');

disp(SummaryTable);

%% ========================================================================
% Additional initialization-only average summary
%
% This averages across ALL requested noise levels and successful seeds.
% It is useful for a quick overall timing comparison, while the main paper
% table should normally use the noise-resolved SummaryTable above.
% =========================================================================
Init_p1 = initial_p_list(:,1);
Init_p2 = initial_p_list(:,2);

OverallFixedPDEL2_mean = nan(num_init,1);
OverallPDEL2_mean = nan(num_init,1);
OverallOptimizationTime_mean = nan(num_init,1);
OverallPDESolveTime_mean = nan(num_init,1);
OverallMethodTime_mean = nan(num_init,1);

for q = 1:num_init

    valid = reshape(success_all(q,:,:),[],1);

    fixed_vals = reshape(fixed_pde_rel_l2_all(q,:,:),[],1);
    pde_vals = reshape(pde_rel_l2_all(q,:,:),[],1);

    opt_vals = reshape(optimization_time_all(q,:,:),[],1);
    pdet_vals = reshape(pde_solve_time_all(q,:,:),[],1);
    method_vals = reshape(method_time_all(q,:,:),[],1);

    OverallFixedPDEL2_mean(q) = mean(fixed_vals(valid));
    OverallPDEL2_mean(q) = mean(pde_vals(valid));

    OverallOptimizationTime_mean(q) = mean(opt_vals(valid));
    OverallPDESolveTime_mean(q) = mean(pdet_vals(valid));
    OverallMethodTime_mean(q) = mean(method_vals(valid));
end

InitializationOverallTable = table( ...
    Init_p1, ...
    Init_p2, ...
    OverallFixedPDEL2_mean, ...
    OverallPDEL2_mean, ...
    OverallOptimizationTime_mean, ...
    OverallPDESolveTime_mean, ...
    OverallMethodTime_mean);

fprintf('\n');
fprintf('===============================================================================================\n');
fprintf('INITIALIZATION-ONLY OVERALL AVERAGE (averaged across all requested noise levels)\n');
fprintf('===============================================================================================\n');
disp(InitializationOverallTable);

%% ========================================================================
% Detailed all-run table
% =========================================================================
nrows = num_init*num_noise*num_seeds;

detail_init_p1 = nan(nrows,1);
detail_init_p2 = nan(nrows,1);

detail_delta = nan(nrows,1);
detail_seed = nan(nrows,1);

detail_target_l2 = nan(nrows,1);
detail_data_fit_l2 = nan(nrows,1);

detail_p1 = nan(nrows,1);
detail_p2 = nan(nrows,1);

detail_ckpt = nan(nrows,1);
detail_best_data_mse = nan(nrows,1);

detail_fixed_pde_l2 = nan(nrows,1);
detail_pde_l2 = nan(nrows,1);
detail_pde_linf = nan(nrows,1);

detail_opt_time = nan(nrows,1);
detail_pde_time = nan(nrows,1);
detail_method_time = nan(nrows,1);

detail_success = false(nrows,1);

row = 0;

for q = 1:num_init
    for a = 1:num_noise
        for i = 1:num_seeds

            row = row+1;

            detail_init_p1(row) = initial_p_list(q,1);
            detail_init_p2(row) = initial_p_list(q,2);

            detail_delta(row) = noise_levels(a);
            detail_seed(row) = seeds(i);

            detail_target_l2(row) = target_rel_l2_all(q,a,i);
            detail_data_fit_l2(row) = data_fit_rel_l2_all(q,a,i);

            detail_p1(row) = p1_opt_all(q,a,i);
            detail_p2(row) = p2_opt_all(q,a,i);

            detail_ckpt(row) = selected_checkpoint_all(q,a,i);
            detail_best_data_mse(row) = best_data_mse_all(q,a,i);

            detail_fixed_pde_l2(row) = fixed_pde_rel_l2_all(q,a,i);
            detail_pde_l2(row) = pde_rel_l2_all(q,a,i);
            detail_pde_linf(row) = pde_rel_linf_all(q,a,i);

            detail_opt_time(row) = optimization_time_all(q,a,i);
            detail_pde_time(row) = pde_solve_time_all(q,a,i);
            detail_method_time(row) = method_time_all(q,a,i);

            detail_success(row) = success_all(q,a,i);
        end
    end
end

AllRunsTable = table( ...
    detail_init_p1, ...
    detail_init_p2, ...
    detail_delta, ...
    detail_seed, ...
    detail_target_l2, ...
    detail_data_fit_l2, ...
    detail_p1, ...
    detail_p2, ...
    detail_ckpt, ...
    detail_best_data_mse, ...
    detail_fixed_pde_l2, ...
    detail_pde_l2, ...
    detail_pde_linf, ...
    detail_opt_time, ...
    detail_pde_time, ...
    detail_method_time, ...
    detail_success, ...
    'VariableNames',{ ...
        'Initial_p1', ...
        'Initial_p2', ...
        'NoiseAmplitude', ...
        'Seed', ...
        'TargetRelativeL2', ...
        'DataFitRelativeL2', ...
        'p1_opt', ...
        'p2_opt', ...
        'SelectedCheckpoint', ...
        'BestDataResidualMSE', ...
        'FixedPDERelativeL2', ...
        'DDADPDERelativeL2', ...
        'DDADPDERelativeLinf', ...
        'OptimizationTimeSec', ...
        'PDESolveTimeSec', ...
        'MethodTimeSec', ...
        'Success'});

%% ========================================================================
% Save
% =========================================================================
results = struct();

results.cfg = cfg;

results.initial_p_list = initial_p_list;
results.noise_levels = noise_levels;
results.seeds = seeds;

results.p1_opt_all = p1_opt_all;
results.p2_opt_all = p2_opt_all;

results.selected_checkpoint_all = selected_checkpoint_all;
results.best_data_mse_all = best_data_mse_all;

results.target_rel_l2_all = target_rel_l2_all;
results.data_fit_rel_l2_all = data_fit_rel_l2_all;

results.fixed_pde_rel_l2_all = fixed_pde_rel_l2_all;

results.pde_rel_l2_all = pde_rel_l2_all;
results.pde_rel_linf_all = pde_rel_linf_all;

results.optimization_time_all = optimization_time_all;
results.pde_solve_time_all = pde_solve_time_all;
results.method_time_all = method_time_all;

results.success_all = success_all;
results.completed_all = completed_all;
results.error_message = error_message;

results.SummaryTable = SummaryTable;
results.InitializationOverallTable = InitializationOverallTable;
results.AllRunsTable = AllRunsTable;

mat_file = fullfile( ...
    output_dir, ...
    'ddad_three_init_results.mat');

summary_csv_file = fullfile( ...
    output_dir, ...
    'ddad_three_init_summary.csv');

init_csv_file = fullfile( ...
    output_dir, ...
    'ddad_three_init_overall.csv');

allruns_csv_file = fullfile( ...
    output_dir, ...
    'ddad_three_init_all_runs.csv');

save(mat_file,'results','-v7.3');

writetable(SummaryTable,summary_csv_file);
writetable(InitializationOverallTable,init_csv_file);
writetable(AllRunsTable,allruns_csv_file);

fprintf('\nSaved MAT:\n%s\n',mat_file);
fprintf('\nSaved noise-resolved summary CSV:\n%s\n',summary_csv_file);
fprintf('\nSaved initialization-overall CSV:\n%s\n',init_csv_file);
fprintf('\nSaved all-runs CSV:\n%s\n',allruns_csv_file);

%% ========================================================================
% Nested checkpoint helper
% =========================================================================
    function state = make_state()

        state = struct();

        state.cfg = cfg;

        state.initial_p_list = initial_p_list;
        state.noise_levels = noise_levels;
        state.seeds = seeds;

        state.p1_opt_all = p1_opt_all;
        state.p2_opt_all = p2_opt_all;

        state.selected_checkpoint_all = selected_checkpoint_all;
        state.best_data_mse_all = best_data_mse_all;

        state.target_rel_l2_all = target_rel_l2_all;
        state.data_fit_rel_l2_all = data_fit_rel_l2_all;

        state.fixed_pde_rel_l2_all = fixed_pde_rel_l2_all;

        state.pde_rel_l2_all = pde_rel_l2_all;
        state.pde_rel_linf_all = pde_rel_linf_all;

        state.optimization_time_all = optimization_time_all;
        state.pde_solve_time_all = pde_solve_time_all;
        state.method_time_all = method_time_all;

        state.success_all = success_all;
        state.completed_all = completed_all;
        state.error_message = error_message;
    end
end
