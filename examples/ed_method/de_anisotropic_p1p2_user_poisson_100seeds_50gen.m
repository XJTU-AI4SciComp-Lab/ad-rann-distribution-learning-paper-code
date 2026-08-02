function results = de_anisotropic_p1p2_user_poisson_100seeds_50gen()
% =========================================================================
% ANISOTROPIC 2-PARAMETER DIFFERENTIAL-EVOLUTION BASELINE
%
% User Poisson benchmark:
%
%       -Delta u = f,       Omega = [-1,1]^2
%
%       u(x,y) = sin(pi*x) sin(5*pi*y)
%
% hence
%
%       f(x,y) = 26*pi^2*sin(pi*x)*sin(5*pi*y).
%
% -------------------------------------------------------------------------
% Random-feature parameterization
% -------------------------------------------------------------------------
%
% For seed s we draw once:
%
%       w_j = (w1_j,w2_j),        w1_j,w2_j ~ U[-1,1],
%       c_j = (c1_j,c2_j),        c1_j,c2_j ~ U[-1,1].
%
% These random quantities remain FIXED during the whole DE search.
%
% The anisotropic Gaussian feature is
%
%       phi_j(x,y;p1,p2)
%       =
%       exp( -S_j(x,y;p1,p2)^2 ),
%
% where
%
%       S_j
%       =
%       p1*w1_j*(x-c1_j)
%       +
%       p2*w2_j*(y-c2_j).
%
% The optimized distribution parameter is
%
%       p = (p1,p2).
%
% The frequency-informed point
%
%       p0 = (4,8)
%
% is inserted as the FIRST member of the initial DE population.
% The other individuals are generated globally by Latin hypercube sampling.
%
% -------------------------------------------------------------------------
% DE objective: Dong-Yang-style residual minimization
% -------------------------------------------------------------------------
%
% For each candidate p,
%
%       beta_LS(p)
%       =
%       argmin_beta ||A(p) beta - rhs||_2,
%
% and DE minimizes
%
%       K(p)
%       =
%       ||A(p) beta_LS(p) - rhs||_2.
%
% No ridge penalty.
% No rank penalty.
% No local polishing.
%
% -------------------------------------------------------------------------
% Differential-evolution configuration
% -------------------------------------------------------------------------
%
%       strategy             : best1bin
%       population           : 10
%       bounds               : [1e-3,100]^2
%       maximum generations  : 50
%       relative tolerance   : 0.1
%       mutation             : F ~ U[0.5,1.0) once per generation
%       recombination        : 0.7
%       initialization       : p0=(4,8) + Latin hypercube
%       updating             : immediate
%       polishing            : OFF
%
% -------------------------------------------------------------------------
% User benchmark settings retained
% -------------------------------------------------------------------------
%
%       Interior grid          : 30 x 80
%       Interior offset        : 1e-6
%       Boundary points/side   : 100
%       Boundary weight        : 100
%       Hidden features        : 600
%       Evaluation grid        : 101 x 101
%       Random seeds           : 1:100
%
% -------------------------------------------------------------------------
% Timing
% -------------------------------------------------------------------------
%
% Per seed:
%
%       Setup
%           random basis + p-independent cache
%
%       DE search
%           all DE objective evaluations
%
%       Final refit
%           assemble A(p*) + final unregularized least-squares solve
%
%       Test
%           evaluate the final approximation on 101 x 101 grid
%
%       Algorithm time
%           Setup + DE search + Final refit
%
%       Total time
%           Algorithm time + Test
%
% Plotting, file saving, common deterministic point generation, and
% one global GPU warm-up are excluded.
%
% -------------------------------------------------------------------------
% Reliability
% -------------------------------------------------------------------------
%
% Results are checkpointed after every completed seed.
% With cfg.resume=true, rerunning resumes unfinished seeds.
%
% =========================================================================

clc;
close all;

warning('off','MATLAB:rankDeficientMatrix');


%% ========================================================================
% 1. USER POISSON PROBLEM
% =========================================================================

cfg.xmin = -1.0;
cfg.xmax =  1.0;

cfg.ymin = -1.0;
cfg.ymax =  1.0;

cfg.Nx_int = 30;
cfg.Ny_int = 80;

cfg.interior_offset = 1.0e-6;

cfg.Nb_side = 100;

cfg.M = 600;

cfg.eta_bnd = 100.0;

cfg.test_Nx = 101;
cfg.test_Ny = 101;

cfg.seeds = 1:100;

num_seeds = numel(cfg.seeds);


%% ========================================================================
% 2. TWO-DIMENSIONAL DISTRIBUTION PARAMETER
% =========================================================================

cfg.p0 = [4.0, 8.0];

% Same broad bounds as the current AD-RaNN study.
cfg.p_lower = [1.0e-3, 1.0e-3];
cfg.p_upper = [100.0,    100.0];


%% ========================================================================
% 3. DIFFERENTIAL EVOLUTION
% =========================================================================

de.strategy = 'best1bin';

de.population_size = 10;

de.max_generations = 50;

de.relative_tolerance = 0.10;
de.absolute_tolerance = 0.0;

de.mutation_min = 0.50;
de.mutation_max = 1.00;

de.recombination = 0.70;

de.initialization = 'p0_plus_latinhypercube';

de.updating = 'immediate';

de.polish = false;


%% ========================================================================
% 4. GPU / LEAST-SQUARES SETTINGS
% =========================================================================

cfg.use_gpu_lsq = false;

% MATLAB GPU indexing is 1-based.
cfg.gpu_device_id = 1;

if cfg.use_gpu_lsq

    try

        if cfg.gpu_device_id > gpuDeviceCount
            error( ...
                'Requested GPU %d but MATLAB sees only %d GPU(s).', ...
                cfg.gpu_device_id, ...
                gpuDeviceCount);
        end

        g = gpuDevice(cfg.gpu_device_id);

        fprintf('\n');
        fprintf('============================================================\n');
        fprintf('GPU\n');
        fprintf('============================================================\n');
        fprintf('Index            = %d\n',g.Index);
        fprintf('Name             = %s\n',g.Name);
        fprintf('Available memory = %.3f GB\n', ...
            g.AvailableMemory/1024^3);
        fprintf('============================================================\n');

    catch ME

        warning( ...
            'GPU initialization failed; falling back to CPU.\n%s', ...
            ME.message);

        cfg.use_gpu_lsq = false;

    end

end


%% ========================================================================
% 5. OUTPUT
% =========================================================================

cfg.output_dir = ...
    'outputs_DE_anisotropic_p1p2_USER_Poisson_100seeds_50gen';

if ~exist(cfg.output_dir,'dir')
    mkdir(cfg.output_dir);
end

cfg.checkpoint_file = fullfile( ...
    cfg.output_dir, ...
    'checkpoint_DE_anisotropic_p1p2_100seeds_50gen.mat');

cfg.final_mat_file = fullfile( ...
    cfg.output_dir, ...
    'results_DE_anisotropic_p1p2_100seeds_50gen.mat');

cfg.per_seed_csv_file = fullfile( ...
    cfg.output_dir, ...
    'per_seed_DE_anisotropic_p1p2_100seeds_50gen.csv');

cfg.summary_csv_file = fullfile( ...
    cfg.output_dir, ...
    'summary_DE_anisotropic_p1p2_100seeds_50gen.csv');

cfg.resume = true;

cfg.make_summary_plots = true;


%% ========================================================================
% 6. DETERMINISTIC PDE / TEST DATA
%
% Common to all seeds and therefore generated only once.
% =========================================================================

Xint = make_interior_points(cfg);

Xbnd = make_boundary_points(cfg);

f_int = rhs_f( ...
    Xint(:,1), ...
    Xint(:,2));

g_bnd = exact_u( ...
    Xbnd(:,1), ...
    Xbnd(:,2));

rhs = [ ...
    f_int; ...
    cfg.eta_bnd*g_bnd];

[Xtest,Ytest,Xtest_points] = ...
    make_test_grid(cfg);

u_test = exact_u( ...
    Xtest_points(:,1), ...
    Xtest_points(:,2));

Nint = size(Xint,1);
Nbnd = size(Xbnd,1);
Neq = Nint + Nbnd;


%% ========================================================================
% 7. RESULT ARRAYS
% =========================================================================

p1_star_all = nan(num_seeds,1);
p2_star_all = nan(num_seeds,1);

best_K_all = nan(num_seeds,1);

rel_l2_all = nan(num_seeds,1);
rel_linf_all = nan(num_seeds,1);

rms_error_all = nan(num_seeds,1);
max_error_all = nan(num_seeds,1);

setup_time_all = nan(num_seeds,1);
search_time_all = nan(num_seeds,1);
final_refit_time_all = nan(num_seeds,1);
test_time_all = nan(num_seeds,1);

algorithm_time_all = nan(num_seeds,1);
total_time_all = nan(num_seeds,1);

generation_all = nan(num_seeds,1);
function_evaluations_all = nan(num_seeds,1);

final_training_residual_all = nan(num_seeds,1);

success_all = false(num_seeds,1);
completed_all = false(num_seeds,1);

error_message = cell(num_seeds,1);
de_history_all = cell(num_seeds,1);


%% ========================================================================
% 8. RESUME CHECKPOINT
% =========================================================================

if cfg.resume && exist(cfg.checkpoint_file,'file') == 2

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('RESUME CHECKPOINT\n');
    fprintf('============================================================\n');

    S = load(cfg.checkpoint_file);

    if isfield(S,'state')

        state = S.state;

        if isfield(state,'seeds')
            if ~isequal(state.seeds(:),cfg.seeds(:))
                error('Checkpoint seed list differs from current cfg.seeds.');
            end
        end

        p1_star_all = restore_field( ...
            state,'p1_star_all',p1_star_all);

        p2_star_all = restore_field( ...
            state,'p2_star_all',p2_star_all);

        best_K_all = restore_field( ...
            state,'best_K_all',best_K_all);

        rel_l2_all = restore_field( ...
            state,'rel_l2_all',rel_l2_all);

        rel_linf_all = restore_field( ...
            state,'rel_linf_all',rel_linf_all);

        rms_error_all = restore_field( ...
            state,'rms_error_all',rms_error_all);

        max_error_all = restore_field( ...
            state,'max_error_all',max_error_all);

        setup_time_all = restore_field( ...
            state,'setup_time_all',setup_time_all);

        search_time_all = restore_field( ...
            state,'search_time_all',search_time_all);

        final_refit_time_all = restore_field( ...
            state,'final_refit_time_all',final_refit_time_all);

        test_time_all = restore_field( ...
            state,'test_time_all',test_time_all);

        algorithm_time_all = restore_field( ...
            state,'algorithm_time_all',algorithm_time_all);

        total_time_all = restore_field( ...
            state,'total_time_all',total_time_all);

        generation_all = restore_field( ...
            state,'generation_all',generation_all);

        function_evaluations_all = restore_field( ...
            state,'function_evaluations_all',function_evaluations_all);

        final_training_residual_all = restore_field( ...
            state, ...
            'final_training_residual_all', ...
            final_training_residual_all);

        success_all = restore_field( ...
            state,'success_all',success_all);

        completed_all = restore_field( ...
            state,'completed_all',completed_all);

        error_message = restore_field( ...
            state,'error_message',error_message);

        de_history_all = restore_field( ...
            state,'de_history_all',de_history_all);

        fprintf('Completed seeds found = %d / %d\n', ...
            nnz(completed_all), ...
            num_seeds);

    else

        warning('Checkpoint exists but contains no state variable.');

    end

    fprintf('============================================================\n');

end


%% ========================================================================
% 9. ONE GLOBAL UNTIMED GPU WARM-UP
% =========================================================================

if cfg.use_gpu_lsq

    fprintf('\nPerforming one untimed GPU warm-up...\n');

    warm_seed = cfg.seeds(1);

    [W0_warm,C_warm] = ...
        build_random_basis(cfg,warm_seed);

    warm_cache = prepare_anisotropic_cache( ...
        Xint, ...
        Xbnd, ...
        W0_warm, ...
        C_warm, ...
        rhs, ...
        cfg.eta_bnd, ...
        cfg.use_gpu_lsq);

    de_cost_function( ...
        cfg.p0, ...
        warm_cache, ...
        cfg);

    wait(g);

    clear warm_cache W0_warm C_warm;

    fprintf('GPU warm-up completed.\n');

end


%% ========================================================================
% 10. HEADER
% =========================================================================

fprintf('\n\n');
fprintf('========================================================================\n');
fprintf(' ANISOTROPIC p=(p1,p2) DE ON USER POISSON: 100-SEED ROBUSTNESS\n');
fprintf('========================================================================\n');

fprintf('PDE domain              = [-1,1] x [-1,1]\n');
fprintf('Exact solution          = sin(pi*x) sin(5*pi*y)\n');

fprintf('Interior points         = %d x %d = %d\n', ...
    cfg.Nx_int, ...
    cfg.Ny_int, ...
    Nint);

fprintf('Boundary points         = %d\n',Nbnd);

fprintf('Least-squares equations = %d\n',Neq);

fprintf('Hidden features         = %d\n',cfg.M);

fprintf('Evaluation grid         = %d x %d\n', ...
    cfg.test_Nx, ...
    cfg.test_Ny);

fprintf('Activation              = Gaussian exp(-S^2)\n');

fprintf('Optimized parameter     = p=(p1,p2)\n');

fprintf('Reference initial p0    = (%.3f, %.3f)\n', ...
    cfg.p0(1), ...
    cfg.p0(2));

fprintf('DE strategy             = %s\n',de.strategy);

fprintf('DE population           = %d\n',de.population_size);

fprintf('DE p1 bounds            = [%.3e, %.3f]\n', ...
    cfg.p_lower(1), ...
    cfg.p_upper(1));

fprintf('DE p2 bounds            = [%.3e, %.3f]\n', ...
    cfg.p_lower(2), ...
    cfg.p_upper(2));

fprintf('DE max generations      = %d\n',de.max_generations);

fprintf('DE relative tolerance   = %.3f\n', ...
    de.relative_tolerance);

fprintf('DE recombination        = %.3f\n', ...
    de.recombination);

fprintf('DE mutation             = U[%.2f,%.2f) per generation\n', ...
    de.mutation_min, ...
    de.mutation_max);

fprintf('DE initialization       = p0 + Latin hypercube\n');

fprintf('DE updating             = immediate\n');

fprintf('Local polishing         = OFF\n');

fprintf('Seeds                   = 1 : 100\n');

fprintf('GPU least squares       = %d\n',cfg.use_gpu_lsq);

fprintf('Checkpoint/resume       = ON\n');

fprintf('========================================================================\n\n');


%% ========================================================================
% 11. MAIN 100-SEED LOOP
% =========================================================================

for i = 1:num_seeds

    seed_i = cfg.seeds(i);

    if completed_all(i)

        fprintf( ...
            '[seed %3d/%3d] already completed -> skipped\n', ...
            seed_i, ...
            num_seeds);

        continue;

    end

    fprintf('\n');
    fprintf('------------------------------------------------------------------------\n');
    fprintf('Seed %d / %d\n',seed_i,num_seeds);
    fprintf('------------------------------------------------------------------------\n');

    try

        %% =================================================================
        % SETUP
        %
        % Random basis is fixed during the whole DE search for this seed.
        % =================================================================

        setup_timer = tic;

        [W0,C] = ...
            build_random_basis( ...
                cfg, ...
                seed_i);

        cache = prepare_anisotropic_cache( ...
            Xint, ...
            Xbnd, ...
            W0, ...
            C, ...
            rhs, ...
            cfg.eta_bnd, ...
            cfg.use_gpu_lsq);

        if cfg.use_gpu_lsq
            wait(g);
        end

        setup_time = toc(setup_timer);


        %% =================================================================
        % DIFFERENTIAL EVOLUTION
        % =================================================================

        search_timer = tic;

        [p_star,K_star,de_history] = ...
            anisotropic_differential_evolution( ...
                @(p) de_cost_function( ...
                    p, ...
                    cache, ...
                    cfg), ...
                cfg.p0, ...
                cfg.p_lower, ...
                cfg.p_upper, ...
                de, ...
                seed_i);

        if cfg.use_gpu_lsq
            wait(g);
        end

        search_time = toc(search_timer);


        %% =================================================================
        % FINAL UNREGULARIZED REFIT
        % =================================================================

        refit_timer = tic;

        [A_final,b_final] = ...
            assemble_anisotropic_system( ...
                p_star, ...
                cache);

        [beta,final_training_residual] = ...
            stable_linear_least_squares( ...
                A_final, ...
                b_final, ...
                cfg);

        if cfg.use_gpu_lsq
            wait(g);
        end

        final_refit_time = toc(refit_timer);


        %% =================================================================
        % TEST
        % =================================================================

        test_timer = tic;

        Phi_test = ...
            anisotropic_gaussian_features( ...
                Xtest_points, ...
                p_star, ...
                W0, ...
                C);

        u_pred = ...
            Phi_test*beta;

        error_vec = ...
            u_pred-u_test;

        rel_l2 = ...
            norm(error_vec,2) / ...
            max(norm(u_test,2),eps);

        rel_linf = ...
            norm(error_vec,inf) / ...
            max(norm(u_test,inf),eps);

        rms_error = ...
            sqrt(mean(error_vec.^2));

        max_error = ...
            max(abs(error_vec));

        test_time = toc(test_timer);


        %% =================================================================
        % TOTAL TIMES
        % =================================================================

        algorithm_time = ...
            setup_time + ...
            search_time + ...
            final_refit_time;

        total_time = ...
            algorithm_time + ...
            test_time;


        %% =================================================================
        % STORE
        % =================================================================

        p1_star_all(i) = p_star(1);
        p2_star_all(i) = p_star(2);

        best_K_all(i) = K_star;

        rel_l2_all(i) = rel_l2;
        rel_linf_all(i) = rel_linf;

        rms_error_all(i) = rms_error;
        max_error_all(i) = max_error;

        setup_time_all(i) = setup_time;
        search_time_all(i) = search_time;
        final_refit_time_all(i) = final_refit_time;
        test_time_all(i) = test_time;

        algorithm_time_all(i) = algorithm_time;
        total_time_all(i) = total_time;

        generation_all(i) = de_history.generations;

        function_evaluations_all(i) = ...
            de_history.function_evaluations;

        final_training_residual_all(i) = ...
            final_training_residual;

        de_history_all{i} = de_history;

        success_all(i) = true;
        completed_all(i) = true;


        %% =================================================================
        % PRINT
        % =================================================================

        fprintf('\n');

        fprintf( ...
            ['[seed %3d/%3d] ', ...
             'p*=(%.8f, %.8f) | ', ...
             'K=%.3e | ', ...
             'max=%.3e | ', ...
             'rms=%.3e | ', ...
             'relL2=%.3e | ', ...
             'relLinf=%.3e | ', ...
             'gen=%d | ', ...
             'eval=%d | ', ...
             'search=%.3f s | ', ...
             'refit=%.3f s | ', ...
             'total=%.3f s\n'], ...
            seed_i, ...
            num_seeds, ...
            p_star(1), ...
            p_star(2), ...
            K_star, ...
            max_error, ...
            rms_error, ...
            rel_l2, ...
            rel_linf, ...
            de_history.generations, ...
            de_history.function_evaluations, ...
            search_time, ...
            final_refit_time, ...
            total_time);

        fprintf( ...
            'Final training residual = %.6e\n', ...
            final_training_residual);


    catch ME

        success_all(i) = false;

        % Leave unfinished so rerunning can retry this seed.
        completed_all(i) = false;

        error_message{i} = ...
            getReport( ...
                ME, ...
                'extended', ...
                'hyperlinks', ...
                'off');

        fprintf(2, ...
            '\nSEED %d FAILED\n', ...
            seed_i);

        fprintf(2,'%s\n', ...
            error_message{i});

    end


    %% ====================================================================
    % CHECKPOINT
    % =====================================================================

    state = build_checkpoint_state( ...
        cfg, ...
        de, ...
        p1_star_all, ...
        p2_star_all, ...
        best_K_all, ...
        rel_l2_all, ...
        rel_linf_all, ...
        rms_error_all, ...
        max_error_all, ...
        setup_time_all, ...
        search_time_all, ...
        final_refit_time_all, ...
        test_time_all, ...
        algorithm_time_all, ...
        total_time_all, ...
        generation_all, ...
        function_evaluations_all, ...
        final_training_residual_all, ...
        success_all, ...
        completed_all, ...
        error_message, ...
        de_history_all);

    save( ...
        cfg.checkpoint_file, ...
        'state', ...
        '-v7.3');

    fprintf( ...
        'Checkpoint saved: %d/%d successful-completed\n', ...
        nnz(completed_all), ...
        num_seeds);

    clear cache W0 C A_final b_final beta;

end


%% ========================================================================
% 12. FINAL STATISTICS
% =========================================================================

valid = success_all;

num_success = nnz(valid);
num_failed = num_seeds-num_success;

if num_success == 0
    error('No successful DE runs were obtained.');
end


%% ------------------------------------------------------------------------
% Optimized p
% -------------------------------------------------------------------------

p1_mean = mean(p1_star_all(valid));
p1_std = std(p1_star_all(valid));
p1_median = median(p1_star_all(valid));

p2_mean = mean(p2_star_all(valid));
p2_std = std(p2_star_all(valid));
p2_median = median(p2_star_all(valid));


%% ------------------------------------------------------------------------
% Objective
% -------------------------------------------------------------------------

K_mean = mean(best_K_all(valid));
K_std = std(best_K_all(valid));


%% ------------------------------------------------------------------------
% Relative L2
% -------------------------------------------------------------------------

L2_mean = mean(rel_l2_all(valid));
L2_std = std(rel_l2_all(valid));
L2_median = median(rel_l2_all(valid));
L2_best = min(rel_l2_all(valid));
L2_worst = max(rel_l2_all(valid));

valid_indices = find(valid);

[~,tmp_best] = min(rel_l2_all(valid));
[~,tmp_worst] = max(rel_l2_all(valid));

best_seed = cfg.seeds(valid_indices(tmp_best));
worst_seed = cfg.seeds(valid_indices(tmp_worst));


%% ------------------------------------------------------------------------
% Relative Linf
% -------------------------------------------------------------------------

Linf_mean = mean(rel_linf_all(valid));
Linf_std = std(rel_linf_all(valid));
Linf_median = median(rel_linf_all(valid));
Linf_best = min(rel_linf_all(valid));
Linf_worst = max(rel_linf_all(valid));


%% ------------------------------------------------------------------------
% RMS / maximum absolute error
% -------------------------------------------------------------------------

RMS_mean = mean(rms_error_all(valid));
RMS_std = std(rms_error_all(valid));

Max_mean = mean(max_error_all(valid));
Max_std = std(max_error_all(valid));


%% ------------------------------------------------------------------------
% DE statistics
% -------------------------------------------------------------------------

generation_mean = ...
    mean(generation_all(valid));

generation_std = ...
    std(generation_all(valid));

evaluation_mean = ...
    mean(function_evaluations_all(valid));

evaluation_std = ...
    std(function_evaluations_all(valid));


%% ------------------------------------------------------------------------
% Timing
% -------------------------------------------------------------------------

setup_mean = mean(setup_time_all(valid));
setup_std = std(setup_time_all(valid));

search_mean = mean(search_time_all(valid));
search_std = std(search_time_all(valid));

refit_mean = mean(final_refit_time_all(valid));
refit_std = std(final_refit_time_all(valid));

test_mean = mean(test_time_all(valid));
test_std = std(test_time_all(valid));

algorithm_mean = mean(algorithm_time_all(valid));
algorithm_std = std(algorithm_time_all(valid));

total_mean = mean(total_time_all(valid));
total_std = std(total_time_all(valid));


%% ========================================================================
% 13. FINAL REPORT
% =========================================================================

fprintf('\n\n');
fprintf('========================================================================\n');
fprintf(' ANISOTROPIC p=(p1,p2) DE: FINAL 100-SEED SUMMARY\n');
fprintf('========================================================================\n');

fprintf('Successful runs          = %d / %d\n', ...
    num_success, ...
    num_seeds);

fprintf('Failed/uncompleted runs  = %d / %d\n', ...
    num_failed, ...
    num_seeds);

fprintf('\nOptimized p*\n');

fprintf('Mean p*                  = (%.10f, %.10f)\n', ...
    p1_mean, ...
    p2_mean);

fprintf('Std p*                   = (%.10f, %.10f)\n', ...
    p1_std, ...
    p2_std);

fprintf('Median p*                = (%.10f, %.10f)\n', ...
    p1_median, ...
    p2_median);

fprintf('\nDE residual objective K\n');

fprintf('Mean                     = %.10e\n',K_mean);
fprintf('Std                      = %.10e\n',K_std);

fprintf('\nRelative L2\n');

fprintf('Mean                     = %.10e\n',L2_mean);
fprintf('Std                      = %.10e\n',L2_std);
fprintf('Median                   = %.10e\n',L2_median);
fprintf('Best                     = %.10e\n',L2_best);
fprintf('Worst                    = %.10e\n',L2_worst);
fprintf('Best seed                = %d\n',best_seed);
fprintf('Worst seed               = %d\n',worst_seed);

fprintf('\nRelative Linf\n');

fprintf('Mean                     = %.10e\n',Linf_mean);
fprintf('Std                      = %.10e\n',Linf_std);
fprintf('Median                   = %.10e\n',Linf_median);
fprintf('Best                     = %.10e\n',Linf_best);
fprintf('Worst                    = %.10e\n',Linf_worst);

fprintf('\nAbsolute-error metrics\n');

fprintf('Mean RMS error           = %.10e\n',RMS_mean);
fprintf('Std RMS error            = %.10e\n',RMS_std);
fprintf('Mean maximum error       = %.10e\n',Max_mean);
fprintf('Std maximum error        = %.10e\n',Max_std);

fprintf('\nDifferential evolution\n');

fprintf('Mean generations         = %.3f\n',generation_mean);
fprintf('Std generations          = %.3f\n',generation_std);
fprintf('Mean objective evals     = %.3f\n',evaluation_mean);
fprintf('Std objective evals      = %.3f\n',evaluation_std);

fprintf('\nMean timing breakdown\n');

fprintf('Setup                    = %.6f +/- %.6f s\n', ...
    setup_mean, ...
    setup_std);

fprintf('DE search                = %.6f +/- %.6f s\n', ...
    search_mean, ...
    search_std);

fprintf('Final refit              = %.6f +/- %.6f s\n', ...
    refit_mean, ...
    refit_std);

fprintf('Test evaluation          = %.6f +/- %.6f s\n', ...
    test_mean, ...
    test_std);

fprintf('Algorithm time           = %.6f +/- %.6f s\n', ...
    algorithm_mean, ...
    algorithm_std);

fprintf('Total incl. test         = %.6f +/- %.6f s\n', ...
    total_mean, ...
    total_std);

fprintf('========================================================================\n');


%% ========================================================================
% 14. PER-SEED TABLE
% =========================================================================

Seed = cfg.seeds(:);
Success = success_all;

p1_star = p1_star_all;
p2_star = p2_star_all;

Best_K = best_K_all;

Relative_L2 = rel_l2_all;
Relative_Linf = rel_linf_all;

RMSError = rms_error_all;
MaximumError = max_error_all;

Generations = generation_all;
ObjectiveEvaluations = function_evaluations_all;

SetupTimeSec = setup_time_all;
DESearchTimeSec = search_time_all;
FinalRefitTimeSec = final_refit_time_all;
TestTimeSec = test_time_all;

AlgorithmTimeSec = algorithm_time_all;
TotalTimeSec = total_time_all;

PerSeedTable = table( ...
    Seed, ...
    Success, ...
    p1_star, ...
    p2_star, ...
    Best_K, ...
    Relative_L2, ...
    Relative_Linf, ...
    RMSError, ...
    MaximumError, ...
    Generations, ...
    ObjectiveEvaluations, ...
    SetupTimeSec, ...
    DESearchTimeSec, ...
    FinalRefitTimeSec, ...
    TestTimeSec, ...
    AlgorithmTimeSec, ...
    TotalTimeSec);


%% ========================================================================
% 15. SUMMARY TABLE
% =========================================================================

Method = {'Anisotropic DE p=(p1,p2)'}';

Samples = num_success;

Initial_p1 = cfg.p0(1);
Initial_p2 = cfg.p0(2);

Mean_p1 = p1_mean;
Std_p1 = p1_std;

Mean_p2 = p2_mean;
Std_p2 = p2_std;

MeanRelL2 = L2_mean;
StdRelL2 = L2_std;
MedianRelL2 = L2_median;
BestRelL2 = L2_best;
WorstRelL2 = L2_worst;

MeanRelLinf = Linf_mean;
StdRelLinf = Linf_std;

MeanGenerations = generation_mean;
MeanObjectiveEvaluations = evaluation_mean;

MeanSetupTimeSec = setup_mean;
MeanOptimizationTimeSec = search_mean;
MeanFinalRefitTimeSec = refit_mean;
MeanTestTimeSec = test_mean;
MeanAlgorithmTimeSec = algorithm_mean;
MeanTotalTimeSec = total_mean;

SummaryTable = table( ...
    Method, ...
    Samples, ...
    Initial_p1, ...
    Initial_p2, ...
    Mean_p1, ...
    Std_p1, ...
    Mean_p2, ...
    Std_p2, ...
    MeanRelL2, ...
    StdRelL2, ...
    MedianRelL2, ...
    BestRelL2, ...
    WorstRelL2, ...
    MeanRelLinf, ...
    StdRelLinf, ...
    MeanGenerations, ...
    MeanObjectiveEvaluations, ...
    MeanSetupTimeSec, ...
    MeanOptimizationTimeSec, ...
    MeanFinalRefitTimeSec, ...
    MeanTestTimeSec, ...
    MeanAlgorithmTimeSec, ...
    MeanTotalTimeSec);

fprintf('\n');
fprintf('========================================================================\n');
fprintf('PER-SEED RESULTS\n');
fprintf('========================================================================\n\n');

disp(PerSeedTable);

fprintf('\n');
fprintf('========================================================================\n');
fprintf('COMPACT SUMMARY\n');
fprintf('========================================================================\n\n');

disp(SummaryTable);


%% ========================================================================
% 16. SAVE FINAL RESULTS
% =========================================================================

results = struct();

results.cfg = cfg;
results.de = de;

results.seeds = cfg.seeds;

results.success_all = success_all;
results.completed_all = completed_all;
results.error_message = error_message;

results.p1_star_all = p1_star_all;
results.p2_star_all = p2_star_all;

results.best_K_all = best_K_all;

results.rel_l2_all = rel_l2_all;
results.rel_linf_all = rel_linf_all;

results.rms_error_all = rms_error_all;
results.max_error_all = max_error_all;

results.setup_time_all = setup_time_all;
results.search_time_all = search_time_all;
results.final_refit_time_all = final_refit_time_all;
results.test_time_all = test_time_all;

results.algorithm_time_all = algorithm_time_all;
results.total_time_all = total_time_all;

results.generation_all = generation_all;
results.function_evaluations_all = function_evaluations_all;

results.final_training_residual_all = ...
    final_training_residual_all;

results.de_history_all = de_history_all;

results.p1_mean = p1_mean;
results.p1_std = p1_std;

results.p2_mean = p2_mean;
results.p2_std = p2_std;

results.K_mean = K_mean;
results.K_std = K_std;

results.L2_mean = L2_mean;
results.L2_std = L2_std;
results.L2_median = L2_median;
results.L2_best = L2_best;
results.L2_worst = L2_worst;

results.Linf_mean = Linf_mean;
results.Linf_std = Linf_std;
results.Linf_median = Linf_median;
results.Linf_best = Linf_best;
results.Linf_worst = Linf_worst;

results.RMS_mean = RMS_mean;
results.RMS_std = RMS_std;

results.Max_mean = Max_mean;
results.Max_std = Max_std;

results.generation_mean = generation_mean;
results.generation_std = generation_std;

results.evaluation_mean = evaluation_mean;
results.evaluation_std = evaluation_std;

results.setup_mean = setup_mean;
results.setup_std = setup_std;

results.search_mean = search_mean;
results.search_std = search_std;

results.refit_mean = refit_mean;
results.refit_std = refit_std;

results.test_mean = test_mean;
results.test_std = test_std;

results.algorithm_mean = algorithm_mean;
results.algorithm_std = algorithm_std;

results.total_mean = total_mean;
results.total_std = total_std;

results.best_seed = best_seed;
results.worst_seed = worst_seed;

results.PerSeedTable = PerSeedTable;
results.SummaryTable = SummaryTable;

results.Xint = Xint;
results.Xbnd = Xbnd;

results.Xtest = Xtest;
results.Ytest = Ytest;

save( ...
    cfg.final_mat_file, ...
    'results', ...
    '-v7.3');

writetable( ...
    PerSeedTable, ...
    cfg.per_seed_csv_file);

writetable( ...
    SummaryTable, ...
    cfg.summary_csv_file);

fprintf('\nFinal MAT saved to:\n%s\n',cfg.final_mat_file);
fprintf('\nPer-seed CSV saved to:\n%s\n',cfg.per_seed_csv_file);
fprintf('\nSummary CSV saved to:\n%s\n',cfg.summary_csv_file);


%% ========================================================================
% 17. SUMMARY FIGURES
% =========================================================================

if cfg.make_summary_plots

    make_summary_plots( ...
        cfg, ...
        p1_star_all, ...
        p2_star_all, ...
        rel_l2_all, ...
        rel_linf_all, ...
        search_time_all, ...
        total_time_all, ...
        generation_all, ...
        valid);

end

fprintf('\n');
fprintf('========================================================================\n');
fprintf('EXPERIMENT COMPLETED\n');
fprintf('========================================================================\n');

end


%% =========================================================================
% Random basis
%
% This is fixed within each DE search.
% =========================================================================

function [W0,C] = build_random_basis(cfg,seed)

stream = RandStream( ...
    'mt19937ar', ...
    'Seed', ...
    seed);

W0 = ...
    -1.0 + ...
    2.0*rand(stream,2,cfg.M);

C = zeros(2,cfg.M);

C(1,:) = ...
    cfg.xmin + ...
    (cfg.xmax-cfg.xmin)* ...
    rand(stream,1,cfg.M);

C(2,:) = ...
    cfg.ymin + ...
    (cfg.ymax-cfg.ymin)* ...
    rand(stream,1,cfg.M);

end


%% =========================================================================
% Interior points
% =========================================================================

function Xint = make_interior_points(cfg)

x = linspace( ...
    cfg.xmin + cfg.interior_offset, ...
    cfg.xmax - cfg.interior_offset, ...
    cfg.Nx_int);

y = linspace( ...
    cfg.ymin + cfg.interior_offset, ...
    cfg.ymax - cfg.interior_offset, ...
    cfg.Ny_int);

[X,Y] = meshgrid(x,y);

Xint = [X(:),Y(:)];

end


%% =========================================================================
% Boundary points
% =========================================================================

function Xbnd = make_boundary_points(cfg)

x = linspace( ...
    cfg.xmin, ...
    cfg.xmax, ...
    cfg.Nb_side).';

y = linspace( ...
    cfg.ymin, ...
    cfg.ymax, ...
    cfg.Nb_side).';

Xbnd = [ ...
    cfg.xmin*ones(cfg.Nb_side,1), y; ...
    cfg.xmax*ones(cfg.Nb_side,1), y; ...
    x, cfg.ymin*ones(cfg.Nb_side,1); ...
    x, cfg.ymax*ones(cfg.Nb_side,1)];

end


%% =========================================================================
% Test grid
% =========================================================================

function [X,Y,points] = make_test_grid(cfg)

x = linspace( ...
    cfg.xmin, ...
    cfg.xmax, ...
    cfg.test_Nx);

y = linspace( ...
    cfg.ymin, ...
    cfg.ymax, ...
    cfg.test_Ny);

[X,Y] = meshgrid(x,y);

points = [X(:),Y(:)];

end


%% =========================================================================
% Exact solution
% =========================================================================

function u = exact_u(x,y)

u = ...
    sin(pi*x) .* ...
    sin(5*pi*y);

end


%% =========================================================================
% RHS
% =========================================================================

function f = rhs_f(x,y)

f = ...
    26*pi^2 .* ...
    exact_u(x,y);

end


%% =========================================================================
% Fast p-independent cache
%
%   S = p1*Q1 + p2*Q2
%
% where Q1 and Q2 depend only on the random basis and collocation points.
% =========================================================================

function cache = prepare_anisotropic_cache( ...
    Xint, ...
    Xbnd, ...
    W0, ...
    C, ...
    rhs, ...
    eta, ...
    use_gpu)

w01 = W0(1,:);
w02 = W0(2,:);

Qi1 = bsxfun( ...
    @times, ...
    bsxfun(@minus,Xint(:,1),C(1,:)), ...
    w01);

Qi2 = bsxfun( ...
    @times, ...
    bsxfun(@minus,Xint(:,2),C(2,:)), ...
    w02);

Qb1 = bsxfun( ...
    @times, ...
    bsxfun(@minus,Xbnd(:,1),C(1,:)), ...
    w01);

Qb2 = bsxfun( ...
    @times, ...
    bsxfun(@minus,Xbnd(:,2),C(2,:)), ...
    w02);

w01_squared = w01.^2;
w02_squared = w02.^2;

if use_gpu

    cache.Qi1 = gpuArray(Qi1);
    cache.Qi2 = gpuArray(Qi2);

    cache.Qb1 = gpuArray(Qb1);
    cache.Qb2 = gpuArray(Qb2);

    cache.w01_squared = gpuArray(w01_squared);
    cache.w02_squared = gpuArray(w02_squared);

    cache.rhs = gpuArray(rhs);

else

    cache.Qi1 = Qi1;
    cache.Qi2 = Qi2;

    cache.Qb1 = Qb1;
    cache.Qb2 = Qb2;

    cache.w01_squared = w01_squared;
    cache.w02_squared = w02_squared;

    cache.rhs = rhs;

end

cache.eta = eta;
cache.on_gpu = use_gpu;

end


%% =========================================================================
% Assemble system at p=(p1,p2)
%
% Gaussian:
%
%       sigma(S)  = exp(-S^2)
%       sigma''(S)= (4S^2-2)exp(-S^2)
%
% and
%
%       -Delta phi_j
%       =
%       -sigma''(S_j)
%       [
%           p1^2 w1_j^2
%           +
%           p2^2 w2_j^2
%       ].
% =========================================================================

function [A,b] = ...
    assemble_anisotropic_system( ...
        p, ...
        cache)

p = double(p(:).');

p1 = p(1);
p2 = p(2);

S_int = ...
    p1*cache.Qi1 + ...
    p2*cache.Qi2;

Phi_int = ...
    exp(-(S_int.^2));

sigma_second = ...
    (4*S_int.^2-2).*Phi_int;

weight_norm_squared = ...
    p1^2*cache.w01_squared + ...
    p2^2*cache.w02_squared;

A_int = ...
    -bsxfun( ...
        @times, ...
        sigma_second, ...
        weight_norm_squared);

S_bnd = ...
    p1*cache.Qb1 + ...
    p2*cache.Qb2;

Phi_bnd = ...
    exp(-(S_bnd.^2));

A_bnd = ...
    cache.eta*Phi_bnd;

A = [ ...
    A_int; ...
    A_bnd];

b = cache.rhs;

end


%% =========================================================================
% DE residual objective
% =========================================================================

function K = ...
    de_cost_function( ...
        p, ...
        cache, ...
        cfg)

p = double(p(:).');

if numel(p) ~= 2 || ...
        any(~isfinite(p)) || ...
        any(p <= 0)

    K = inf;
    return;

end

try

    [A,b] = ...
        assemble_anisotropic_system( ...
            p, ...
            cache);

    beta = A\b;

    residual = ...
        A*beta-b;

    if cfg.use_gpu_lsq
        K = gather(norm(residual,2));
    else
        K = norm(residual,2);
    end

    if ~isfinite(K)
        K = inf;
    end

catch

    K = inf;

end

end


%% =========================================================================
% Final unregularized LS
% =========================================================================

function [beta,residual_norm] = ...
    stable_linear_least_squares( ...
        A, ...
        b, ...
        cfg)

beta_raw = A\b;

residual_raw = ...
    A*beta_raw-b;

if cfg.use_gpu_lsq

    residual_norm = ...
        gather(norm(residual_raw,2));

    beta = gather(beta_raw);

else

    residual_norm = ...
        norm(residual_raw,2);

    beta = beta_raw;

end

end


%% =========================================================================
% Test feature matrix
% =========================================================================

function Phi = ...
    anisotropic_gaussian_features( ...
        X, ...
        p, ...
        W0, ...
        C)

p = double(p(:).');

p1 = p(1);
p2 = p(2);

S = ...
    p1*bsxfun( ...
        @times, ...
        bsxfun(@minus,X(:,1),C(1,:)), ...
        W0(1,:)) ...
    + ...
    p2*bsxfun( ...
        @times, ...
        bsxfun(@minus,X(:,2),C(2,:)), ...
        W0(2,:));

Phi = ...
    exp(-(S.^2));

end


%% =========================================================================
% 2-D Differential evolution
%
% best1bin + immediate updating + mutation dithering.
%
% Initial population:
%   row 1       = p0 = (4,8)
%   rows 2:NP   = Latin-hypercube global samples in the whole search box
%
% Convergence:
%
%   std(fval)
%   <=
%   atol + tol*abs(mean(fval)).
% =========================================================================

function [best_x,best_f,history] = ...
    anisotropic_differential_evolution( ...
        fun, ...
        p0, ...
        lb, ...
        ub, ...
        de, ...
        random_seed)

stream = RandStream( ...
    'mt19937ar', ...
    'Seed', ...
    random_seed);

p0 = double(p0(:).');
lb = double(lb(:).');
ub = double(ub(:).');

D = numel(p0);
NP = de.population_size;

if D ~= 2
    error('This implementation expects D=2.');
end

if NP < 5
    error('Population size must be at least 5 for best1bin.');
end

if any(p0 < lb) || any(p0 > ub)
    error('p0 lies outside the DE bounds.');
end


%% ------------------------------------------------------------------------
% Initial population
% -------------------------------------------------------------------------

population = zeros(NP,D);

population(1,:) = p0;

population(2:end,:) = ...
    latin_hypercube_nd( ...
        NP-1, ...
        D, ...
        lb, ...
        ub, ...
        stream);


%% ------------------------------------------------------------------------
% Initial objective values
% -------------------------------------------------------------------------

fval = inf(NP,1);

function_evaluations = 0;

for i = 1:NP

    fval(i) = ...
        fun(population(i,:));

    function_evaluations = ...
        function_evaluations + 1;

end

[best_f,best_index] = ...
    min(fval);

best_x = ...
    population(best_index,:);


%% ------------------------------------------------------------------------
% History
% -------------------------------------------------------------------------

history.best_value = ...
    nan(de.max_generations+1,1);

history.mean_value = ...
    nan(de.max_generations+1,1);

history.std_value = ...
    nan(de.max_generations+1,1);

history.relative_spread = ...
    nan(de.max_generations+1,1);

history.best_p = ...
    nan(de.max_generations+1,D);

history.best_value(1) = ...
    best_f;

history.mean_value(1) = ...
    safe_mean(fval);

history.std_value(1) = ...
    safe_std(fval);

history.relative_spread(1) = ...
    convergence_measure( ...
        fval, ...
        de);

history.best_p(1,:) = ...
    best_x;

fprintf( ...
    ['Gen %3d | best K=%.6e | ', ...
     'p=(%.8f, %.8f) | spread=%.3e\n'], ...
    0, ...
    best_f, ...
    best_x(1), ...
    best_x(2), ...
    history.relative_spread(1));

generation_completed = 0;


%% ------------------------------------------------------------------------
% DE generations
% -------------------------------------------------------------------------

for generation = 1:de.max_generations

    % Mutation dithering: one F for the whole generation.
    F = ...
        de.mutation_min + ...
        (de.mutation_max-de.mutation_min)* ...
        rand(stream);

    for i = 1:NP

        %% ---------------------------------------------------------------
        % Pick r1,r2 different from target
        % ---------------------------------------------------------------

        pool = [1:i-1,i+1:NP];

        order = ...
            randperm(stream,numel(pool));

        r1 = pool(order(1));
        r2 = pool(order(2));


        %% ---------------------------------------------------------------
        % best1 mutation
        % ---------------------------------------------------------------

        mutant = ...
            best_x + ...
            F*( ...
                population(r1,:) - ...
                population(r2,:));


        %% ---------------------------------------------------------------
        % SciPy-style bound repair:
        % any infeasible coordinate is resampled uniformly.
        % ---------------------------------------------------------------

        for d = 1:D

            if mutant(d) < lb(d) || mutant(d) > ub(d)

                mutant(d) = ...
                    lb(d) + ...
                    (ub(d)-lb(d))* ...
                    rand(stream);

            end

        end


        %% ---------------------------------------------------------------
        % Binomial crossover
        % ---------------------------------------------------------------

        crossover_mask = ...
            rand(stream,1,D) <= ...
            de.recombination;

        jrand = ...
            randi(stream,D);

        crossover_mask(jrand) = true;

        trial = ...
            population(i,:);

        trial(crossover_mask) = ...
            mutant(crossover_mask);


        %% ---------------------------------------------------------------
        % Objective
        % ---------------------------------------------------------------

        trial_f = ...
            fun(trial);

        function_evaluations = ...
            function_evaluations + 1;


        %% ---------------------------------------------------------------
        % Immediate updating
        % ---------------------------------------------------------------

        if trial_f <= fval(i)

            population(i,:) = ...
                trial;

            fval(i) = ...
                trial_f;

            if trial_f < best_f

                best_f = trial_f;
                best_x = trial;

            end

        end

    end


    %% -------------------------------------------------------------------
    % Convergence
    % --------------------------------------------------------------------

    spread = ...
        convergence_measure( ...
            fval, ...
            de);

    history.best_value(generation+1) = ...
        best_f;

    history.mean_value(generation+1) = ...
        safe_mean(fval);

    history.std_value(generation+1) = ...
        safe_std(fval);

    history.relative_spread(generation+1) = ...
        spread;

    history.best_p(generation+1,:) = ...
        best_x;

    generation_completed = ...
        generation;

    fprintf( ...
        ['Gen %3d | best K=%.6e | ', ...
         'p=(%.8f, %.8f) | spread=%.3e\n'], ...
        generation, ...
        best_f, ...
        best_x(1), ...
        best_x(2), ...
        spread);

    if has_converged(fval,de)

        fprintf( ...
            'DE converged at generation %d.\n', ...
            generation);

        break;

    end

end


%% ------------------------------------------------------------------------
% Trim history
% -------------------------------------------------------------------------

last = ...
    generation_completed+1;

history.best_value = ...
    history.best_value(1:last);

history.mean_value = ...
    history.mean_value(1:last);

history.std_value = ...
    history.std_value(1:last);

history.relative_spread = ...
    history.relative_spread(1:last);

history.best_p = ...
    history.best_p(1:last,:);

history.generations = ...
    generation_completed;

history.function_evaluations = ...
    function_evaluations;

end


%% =========================================================================
% Latin hypercube in D dimensions
% =========================================================================

function population = ...
    latin_hypercube_nd( ...
        NP, ...
        D, ...
        lb, ...
        ub, ...
        stream)

population_unit = zeros(NP,D);

for d = 1:D

    strata_samples = ...
        ((0:NP-1)' + rand(stream,NP,1)) / NP;

    perm = ...
        randperm(stream,NP);

    population_unit(:,d) = ...
        strata_samples(perm);

end

population = ...
    lb + ...
    population_unit.*(ub-lb);

end


%% =========================================================================
% Convergence
% =========================================================================

function tf = has_converged(fval,de)

finite_values = ...
    fval(isfinite(fval));

if numel(finite_values) < 2
    tf = false;
    return;
end

lhs = ...
    std(finite_values);

rhs = ...
    de.absolute_tolerance + ...
    de.relative_tolerance* ...
    abs(mean(finite_values));

tf = ...
    lhs <= rhs;

end


function spread = convergence_measure(fval,de)

finite_values = ...
    fval(isfinite(fval));

if numel(finite_values) < 2
    spread = inf;
    return;
end

denominator = ...
    de.absolute_tolerance + ...
    abs(mean(finite_values));

if denominator <= eps
    spread = inf;
else
    spread = ...
        std(finite_values) / denominator;
end

end


%% =========================================================================
% Safe statistics
% =========================================================================

function value = safe_mean(x)

x = x(isfinite(x));

if isempty(x)
    value = inf;
else
    value = mean(x);
end

end


function value = safe_std(x)

x = x(isfinite(x));

if numel(x) <= 1
    value = inf;
else
    value = std(x);
end

end


%% =========================================================================
% Build checkpoint
% =========================================================================

function state = build_checkpoint_state( ...
        cfg, ...
        de, ...
        p1_star_all, ...
        p2_star_all, ...
        best_K_all, ...
        rel_l2_all, ...
        rel_linf_all, ...
        rms_error_all, ...
        max_error_all, ...
        setup_time_all, ...
        search_time_all, ...
        final_refit_time_all, ...
        test_time_all, ...
        algorithm_time_all, ...
        total_time_all, ...
        generation_all, ...
        function_evaluations_all, ...
        final_training_residual_all, ...
        success_all, ...
        completed_all, ...
        error_message, ...
        de_history_all)

state = struct();

state.cfg = cfg;
state.de = de;

state.seeds = cfg.seeds;

state.p1_star_all = p1_star_all;
state.p2_star_all = p2_star_all;

state.best_K_all = best_K_all;

state.rel_l2_all = rel_l2_all;
state.rel_linf_all = rel_linf_all;

state.rms_error_all = rms_error_all;
state.max_error_all = max_error_all;

state.setup_time_all = setup_time_all;
state.search_time_all = search_time_all;
state.final_refit_time_all = final_refit_time_all;
state.test_time_all = test_time_all;

state.algorithm_time_all = algorithm_time_all;
state.total_time_all = total_time_all;

state.generation_all = generation_all;
state.function_evaluations_all = function_evaluations_all;

state.final_training_residual_all = ...
    final_training_residual_all;

state.success_all = success_all;
state.completed_all = completed_all;

state.error_message = error_message;
state.de_history_all = de_history_all;

end


%% =========================================================================
% Restore checkpoint field
% =========================================================================

function output = ...
    restore_field( ...
        S, ...
        name, ...
        default_value)

if isfield(S,name)

    candidate = S.(name);

    if isequal(size(candidate),size(default_value))

        output = candidate;

    else

        warning( ...
            ['Checkpoint field %s has incompatible dimensions. ', ...
             'Using a fresh array.'], ...
            name);

        output = default_value;

    end

else

    output = default_value;

end

end


%% =========================================================================
% Summary plots
% =========================================================================

function make_summary_plots( ...
        cfg, ...
        p1_all, ...
        p2_all, ...
        l2_all, ...
        linf_all, ...
        search_time_all, ...
        total_time_all, ...
        generation_all, ...
        valid)

seed_values = ...
    cfg.seeds(valid);


%% ------------------------------------------------------------------------
% Optimized p*
% -------------------------------------------------------------------------

fig = figure( ...
    'Color','w', ...
    'Position',[150,150,760,520]);

plot( ...
    p1_all(valid), ...
    p2_all(valid), ...
    'o', ...
    'MarkerSize',6, ...
    'LineWidth',1.0);

box on;
grid off;

set(gca, ...
    'FontSize',12, ...
    'LineWidth',1.0);

xlabel( ...
    '$p_1^*$', ...
    'Interpreter','latex', ...
    'FontSize',14);

ylabel( ...
    '$p_2^*$', ...
    'Interpreter','latex', ...
    'FontSize',14);

title( ...
    'Optimized DE distribution parameters over 100 random seeds', ...
    'FontSize',12);

exportgraphics( ...
    fig, ...
    fullfile( ...
        cfg.output_dir, ...
        'de_pstar_100seeds.png'), ...
    'Resolution',220);


%% ------------------------------------------------------------------------
% Relative L2
% -------------------------------------------------------------------------

fig = figure( ...
    'Color','w', ...
    'Position',[150,150,800,500]);

semilogy( ...
    seed_values, ...
    l2_all(valid), ...
    'o-', ...
    'LineWidth',1.0, ...
    'MarkerSize',4);

box on;
grid off;

set(gca, ...
    'FontSize',12, ...
    'LineWidth',1.0);

xlabel( ...
    'Random seed', ...
    'FontSize',14);

ylabel( ...
    'Relative $\ell_2$ error', ...
    'Interpreter','latex', ...
    'FontSize',14);

title( ...
    'Anisotropic DE robustness over 100 random seeds', ...
    'FontSize',12);

exportgraphics( ...
    fig, ...
    fullfile( ...
        cfg.output_dir, ...
        'de_l2_100seeds.png'), ...
    'Resolution',220);


%% ------------------------------------------------------------------------
% Error boxchart
% -------------------------------------------------------------------------

fig = figure( ...
    'Color','w', ...
    'Position',[150,150,700,520]);

values = [ ...
    l2_all(valid); ...
    linf_all(valid)];

groups = [ ...
    ones(nnz(valid),1); ...
    2*ones(nnz(valid),1)];

boxchart( ...
    groups, ...
    values);

set(gca, ...
    'YScale','log', ...
    'FontSize',12, ...
    'LineWidth',1.0);

box on;
grid off;

xticks([1,2]);

xticklabels({ ...
    'Relative L2', ...
    'Relative Linf'});

ylabel( ...
    'Error', ...
    'FontSize',14);

title( ...
    'Anisotropic DE error distribution', ...
    'FontSize',12);

exportgraphics( ...
    fig, ...
    fullfile( ...
        cfg.output_dir, ...
        'de_error_boxchart_100seeds.png'), ...
    'Resolution',220);


%% ------------------------------------------------------------------------
% Timing
% -------------------------------------------------------------------------

fig = figure( ...
    'Color','w', ...
    'Position',[150,150,800,500]);

plot( ...
    seed_values, ...
    search_time_all(valid), ...
    'o-', ...
    'LineWidth',1.0, ...
    'MarkerSize',4);

hold on;

plot( ...
    seed_values, ...
    total_time_all(valid), ...
    's-', ...
    'LineWidth',1.0, ...
    'MarkerSize',4);

hold off;

box on;
grid off;

set(gca, ...
    'FontSize',12, ...
    'LineWidth',1.0);

xlabel( ...
    'Random seed', ...
    'FontSize',14);

ylabel( ...
    'Wall time / s', ...
    'FontSize',14);

legend( ...
    'DE search', ...
    'Total', ...
    'Location','best');

title( ...
    'Anisotropic DE computational cost', ...
    'FontSize',12);

exportgraphics( ...
    fig, ...
    fullfile( ...
        cfg.output_dir, ...
        'de_time_100seeds.png'), ...
    'Resolution',220);


%% ------------------------------------------------------------------------
% Generations
% -------------------------------------------------------------------------

fig = figure( ...
    'Color','w', ...
    'Position',[150,150,800,500]);

plot( ...
    seed_values, ...
    generation_all(valid), ...
    'o-', ...
    'LineWidth',1.0, ...
    'MarkerSize',4);

box on;
grid off;

set(gca, ...
    'FontSize',12, ...
    'LineWidth',1.0);

xlabel( ...
    'Random seed', ...
    'FontSize',14);

ylabel( ...
    'DE generations', ...
    'FontSize',14);

title( ...
    'Differential-evolution generations over 100 random seeds', ...
    'FontSize',12);

exportgraphics( ...
    fig, ...
    fullfile( ...
        cfg.output_dir, ...
        'de_generations_100seeds.png'), ...
    'Resolution',220);

end
