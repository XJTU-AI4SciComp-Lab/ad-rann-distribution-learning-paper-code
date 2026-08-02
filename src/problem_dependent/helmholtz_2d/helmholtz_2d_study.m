function results = helmholtz_2d_study(cfg,project_root)
%HELMHOLTZ_2D_STUDY Run one Fre-based or PDAD Helmholtz experiment.
%
% Reduced-feature PDAD:
%   1. create one full frozen random basis with m features;
%   2. use its first m_train columns to optimize p;
%   3. rebuild and solve the original PDE system with all m features.

    if nargin < 2 || isempty(project_root)
        error('helmholtz_2d_study requires cfg and project_root.');
    end

    check_required_common_functions();

    method = upper(strtrim(char(cfg.method)));

    if ~ismember(method,{'FRE','PDAD'})
        error('cfg.method must be FRE or PDAD.');
    end

    if cfg.linear_solver.use_gpu
        select_gpu(cfg.linear_solver.gpu_id);
    end

    if ~isfolder(cfg.output_root)
        mkdir(cfg.output_root);
    end

    total_timer = tic;

    %% Frequency initialization

    initialization_timer = tic;

    switch lower(strtrim(char(cfg.initialization.method)))

        case 'frequency'
            [p0,frequency_info] = ...
                frequency_initialization_helmholtz(cfg);

        case 'manual'
            p0 = cfg.initialization.manual_p(:);
            frequency_info = struct('used_frequency_search',false);

        otherwise
            error('Unknown initialization method: %s', ...
                cfg.initialization.method);
    end

    initialization_time = toc(initialization_timer);

    %% One nested random basis

    basis_full = build_random_weights_nd( ...
        cfg.num_features, ...
        cfg.domain, ...
        cfg.seed);

    if strcmp(method,'PDAD')

        if cfg.training_reduction.enabled

            m_train = cfg.training_reduction.num_features;

            if m_train > cfg.num_features
                error('m_train cannot exceed the final number of features.');
            end

            basis_train = subset_random_basis(basis_full,m_train);

        else

            m_train = cfg.num_features;
            basis_train = basis_full;
        end

    else

        % The Fre baseline has no Adam training stage.
        m_train = 0;
        basis_train = struct();
    end

    %% PDE collocation data

    problem = build_helmholtz_problem(cfg);

    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    %% Optimize p, or retain the frequency initialization

    optimization_time = 0;
    history = struct();

    if strcmp(method,'PDAD')

        if cfg.use_fast_evaluator

            cache = prepare_helmholtz_cache(problem,basis_train);

            objective_fun = @(p) ...
                evaluate_helmholtz_reduced_fast( ...
                    p,cache,cfg.lambda,ls_opts);

        else

            system_builder = @(p) ...
                build_helmholtz_system( ...
                    p,problem,basis_train,true, ...
                    cfg.assembly_chunk_rows);

            objective_fun = @(p) ...
                evaluate_pde_reduced_generic( ...
                    p,system_builder,cfg.lambda,ls_opts);
        end

        [objective0,gradient0,info0] = objective_fun(p0);

        if ~isfinite(objective0) || any(~isfinite(gradient0))
            error('Initial reduced objective or gradient is non-finite.');
        end

        if cfg.verbose
            fprintf('\n============================================================\n');
            fprintf('2-D HELMHOLTZ PDAD\n');
            fprintf('============================================================\n');
            fprintf('case (a1,a2)       = (%g,%g)\n',cfg.a1,cfg.a2);
            fprintf('interior grid       = %d x %d\n', ...
                cfg.interior_grid(1),cfg.interior_grid(2));
            fprintf('train/final features= %d / %d\n', ...
                m_train,cfg.num_features);
            fprintf('frequency p0        = [%.8f, %.8f]\n',p0(1),p0(2));
            fprintf('initial objective   = %.6e\n',objective0);
            fprintf('initial residual MSE= %.6e\n',info0.residual_mse);
            fprintf('============================================================\n\n');
        end

        optimization_timer = tic;

        [p_opt,history] = optimize_distribution_adam( ...
            p0,objective_fun,cfg.optimizer);

        optimization_time = toc(optimization_timer);

    else

        p_opt = p0;

        if cfg.verbose
            fprintf('\n============================================================\n');
            fprintf('2-D HELMHOLTZ FREQUENCY BASELINE\n');
            fprintf('============================================================\n');
            fprintf('case (a1,a2)       = (%g,%g)\n',cfg.a1,cfg.a2);
            fprintf('final features      = %d\n',cfg.num_features);
            fprintf('fixed p             = [%.8f, %.8f]\n',p0(1),p0(2));
            fprintf('============================================================\n\n');
        end
    end

    p_opt = p_opt(:);

    %% Final unregularized PDE solve with all features

    final_solve_timer = tic;

    [M_final,y_final] = build_helmholtz_system( ...
        p_opt,problem,basis_full,false,cfg.assembly_chunk_rows);

    [coef,final_ls_info] = solve_least_squares( ...
        M_final,y_final,cfg.linear_solver);

    if isa(coef,'gpuArray')
        coef = gather(coef);
    end

    coef = coef(:);

    final_solve_time = toc(final_solve_timer);

    final_residual = M_final*coef-y_final;
    final_residual_mse = real(final_residual'*final_residual)/numel(y_final);

    clear M_final y_final final_residual;

    %% Independent test grid

    Xtest = tensor_grid(cfg.domain,cfg.test_grid,0);

    u_exact = helmholtz_exact_solution(Xtest,problem);

    u_pred = evaluate_solution_chunked( ...
        Xtest,p_opt,basis_full,coef,cfg.activation, ...
        cfg.evaluation_chunk_rows);

    rel_l2_error = relative_l2(u_pred,u_exact);
    rel_linf_error = relative_linf(u_pred,u_exact);

    total_time = toc(total_timer);

    %% Training history table

    TrainingHistory = build_history_table(history,method);

    if strcmp(method,'PDAD')
        selected_checkpoint = history.best_iteration;
        selected_residual_mse = history.best_selection_value;
        stop_reason = string(history.stop_reason);
    else
        selected_checkpoint = 0;
        selected_residual_mse = NaN;
        stop_reason = "frequency baseline";
    end

    %% Compact summary

    Method = string(method);
    CaseName = string(cfg.case_name);
    a1 = cfg.a1;
    a2 = cfg.a2;
    TrainFeatures = m_train;
    FinalFeatures = cfg.num_features;
    InitialP1 = p0(1);
    InitialP2 = p0(2);
    OptimizedP1 = p_opt(1);
    OptimizedP2 = p_opt(2);
    SelectedCheckpoint = selected_checkpoint;
    RelativeL2 = rel_l2_error;
    RelativeLinf = rel_linf_error;
    FinalResidualMSE = final_residual_mse;
    InitializationTimeSec = initialization_time;
    OptimizationTimeSec = optimization_time;
    FinalSolveTimeSec = final_solve_time;
    TotalTimeSec = total_time;

    SummaryTable = table( ...
        Method,CaseName,a1,a2,TrainFeatures,FinalFeatures, ...
        InitialP1,InitialP2,OptimizedP1,OptimizedP2, ...
        SelectedCheckpoint,RelativeL2,RelativeLinf, ...
        FinalResidualMSE,InitializationTimeSec, ...
        OptimizationTimeSec,FinalSolveTimeSec,TotalTimeSec);

    if cfg.verbose
        fprintf('\n============================================================\n');
        fprintf('FINAL HELMHOLTZ RESULT\n');
        fprintf('============================================================\n');
        fprintf('method                 = %s\n',method);
        fprintf('case (a1,a2)           = (%g,%g)\n',cfg.a1,cfg.a2);
        fprintf('train/final features   = %d / %d\n', ...
            m_train,cfg.num_features);
        fprintf('p0                     = [%.8f, %.8f]\n',p0(1),p0(2));
        fprintf('p*                     = [%.8f, %.8f]\n',p_opt(1),p_opt(2));
        fprintf('relative L2            = %.6e\n',rel_l2_error);
        fprintf('relative Linf          = %.6e\n',rel_linf_error);
        fprintf('final residual MSE     = %.6e\n',final_residual_mse);
        fprintf('selected checkpoint    = %d\n',selected_checkpoint);
        fprintf('initialization time    = %.6f s\n',initialization_time);
        fprintf('optimization time      = %.6f s\n',optimization_time);
        fprintf('final PDE solve time   = %.6f s\n',final_solve_time);
        fprintf('total time             = %.6f s\n',total_time);
        fprintf('============================================================\n\n');
    end

    %% Results structure

    results = struct();

    results.cfg = cfg;
    results.problem_parameters = struct( ...
        'a1',cfg.a1,'a2',cfg.a2,'k',cfg.k);

    results.method = method;
    results.p0 = p0;
    results.p_opt = p_opt;
    results.frequency_info = frequency_info;
    results.history = history;
    results.training_history = TrainingHistory;

    results.basis_full = basis_full;
    results.num_train_features = m_train;
    results.coef = coef;
    results.final_ls_info = final_ls_info;

    results.test.X = Xtest;
    results.test.u_exact = u_exact;
    results.test.u_pred = u_pred;
    results.test.abs_error = abs(u_pred-u_exact);

    results.relative_l2 = rel_l2_error;
    results.relative_linf = rel_linf_error;
    results.final_residual_mse = final_residual_mse;
    results.selected_checkpoint = selected_checkpoint;
    results.selected_residual_mse = selected_residual_mse;
    results.stop_reason = stop_reason;

    results.timing.initialization = initialization_time;
    results.timing.optimization = optimization_time;
    results.timing.final_solve = final_solve_time;
    results.timing.total = total_time;

    results.summary = SummaryTable;

    %% Save

    tag = sprintf('helmholtz_2d_%s_case_%s',lower(method),cfg.case_name);

    mat_file = fullfile(cfg.output_root,[tag,'.mat']);
    summary_file = fullfile(cfg.output_root,[tag,'_summary.csv']);
    history_file = fullfile(cfg.output_root,[tag,'_training_history.csv']);

    results.files.mat = mat_file;
    results.files.summary_csv = summary_file;

    if ~isempty(TrainingHistory)
        results.files.history_csv = history_file;
    else
        results.files.history_csv = '';
    end

    save(mat_file,'results','-v7.3');
    writetable(SummaryTable,summary_file);

    if ~isempty(TrainingHistory)
        writetable(TrainingHistory,history_file);
    end

    fprintf('Saved MAT     : %s\n',mat_file);
    fprintf('Saved summary : %s\n',summary_file);
end


function pred = evaluate_solution_chunked( ...
        X,p,basis,coef,activation,chunk_rows)

    N = size(X,1);
    pred = zeros(N,1);

    if nargin < 6 || isempty(chunk_rows) || chunk_rows <= 0
        chunk_rows = N;
    end

    for first = 1:chunk_rows:N
        last = min(first+chunk_rows-1,N);
        rows = first:last;

        Phi = activation_features(X(rows,:),p,basis,activation);
        pred(rows) = Phi*coef;
    end
end


function T = build_history_table(history,method)

    if ~strcmp(method,'PDAD') || isempty(fieldnames(history))
        T = table();
        return;
    end

    Iteration = history.iteration(:);
    P1 = history.p(1,:).';
    P2 = history.p(2,:).';
    Objective = history.objective(:);
    ResidualMSE = history.residual_mse(:);
    GradientNorm = history.grad_norm(:);
    StepNorm = history.step_norm(:);

    if isfield(history,'eval_time')
        EvaluationTimeSec = history.eval_time(:);
    else
        EvaluationTimeSec = nan(size(Iteration));
    end

    SelectedCheckpoint = false(size(Iteration));

    if isfield(history,'best_index') && ...
            history.best_index >= 1 && ...
            history.best_index <= numel(Iteration)
        SelectedCheckpoint(history.best_index) = true;
    else
        idx = find(Iteration == history.best_iteration,1,'first');
        if ~isempty(idx)
            SelectedCheckpoint(idx) = true;
        end
    end

    T = table( ...
        Iteration,P1,P2,Objective,ResidualMSE,GradientNorm, ...
        StepNorm,EvaluationTimeSec,SelectedCheckpoint);
end


function check_required_common_functions()

    required = { ...
        'build_random_weights_nd', ...
        'subset_random_basis', ...
        'tensor_grid', ...
        'build_preactivation', ...
        'activation_derivatives', ...
        'activation_features', ...
        'optimize_distribution_adam', ...
        'evaluate_pde_reduced_generic', ...
        'solve_ridge', ...
        'solve_least_squares', ...
        'relative_l2', ...
        'relative_linf'};

    for k = 1:numel(required)
        if exist(required{k},'file') ~= 2
            error('Required common src function not found: %s.m',required{k});
        end
    end
end


function select_gpu(gpu_id)

    count = gpuDeviceCount;

    if gpu_id < 1 || gpu_id > count
        error('Requested GPU %d, but MATLAB sees %d GPU(s).',gpu_id,count);
    end

    device = gpuDevice(gpu_id);
    fprintf('Using GPU %d: %s\n',device.Index,device.Name);
end
