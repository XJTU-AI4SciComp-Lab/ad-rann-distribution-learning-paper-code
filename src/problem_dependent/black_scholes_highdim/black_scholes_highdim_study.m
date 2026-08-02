function results = black_scholes_highdim_study(cfg)
%BLACK_SCHOLES_HIGHDIM_STUDY PDAD-ST solver for high-dimensional BS.

    dataset = load_black_scholes_dataset(cfg);

    if cfg.verbose
        print_header(cfg,dataset);
    end

    total_timer = tic;

    %% Full frozen random basis and deterministic reduced subset

    t = tic;
    basis_full = build_random_weights_nd( ...
        cfg.num_features,cfg.feature_domain,cfg.seed);

    if cfg.training_reduction.enabled
        m_train = cfg.training_reduction.num_features;
    else
        m_train = cfg.num_features;
    end

    basis_train = subset_random_basis(basis_full,m_train);

    cache = prepare_black_scholes_reduced_cache( ...
        dataset.train,basis_train,cfg,cfg.training.constraint_penalty);

    timing.setup = toc(t);

    %% Ridge-reduced PDAD optimization

    ls_train = cfg.linear_solver;
    ls_train.compute_spectrum = false;

    objective_fun = @(p) evaluate_black_scholes_reduced_fast( ...
        p,cache,cfg.ridge_lambda,ls_train,cfg.activation);

    t = tic;
    [p_opt,history] = optimize_distribution_adam( ...
        cfg.initial_p,objective_fun,cfg.optimizer);
    timing.optimization = toc(t);

    %% Full unregularized final PDE refit

    t = tic;
    [M,y] = build_black_scholes_system( ...
        p_opt,dataset.solve,basis_full,cfg, ...
        cfg.final.constraint_penalty,cfg.final.assembly_chunk_rows);
    timing.final_assembly = toc(t);

    t = tic;
    [coef,final_ls_info] = solve_least_squares( ...
        M,y,cfg.linear_solver);
    timing.final_solve = toc(t);

    if isa(coef,'gpuArray')
        coef = gather(coef);
    end
    coef = coef(:);

    %% Independent fixed-reference test set

    t = tic;
    prediction = evaluate_black_scholes_network( ...
        dataset.test.xt,p_opt,basis_full,coef,cfg, ...
        cfg.final.test_chunk_rows);

    reference = double(dataset.test.reference_values(:));
    relative_l2 = norm(prediction-reference)/max(norm(reference),eps);
    timing.test = toc(t);

    timing.total = toc(total_timer);

    %% Result

    results = struct();
    results.cfg = cfg;
    results.p_opt = p_opt;
    results.history = history;
    results.coef = coef;
    results.relative_l2 = relative_l2;
    results.prediction = prediction;
    results.reference = reference;
    results.final_ls_info = final_ls_info;
    results.timing = timing;
    results.data_metadata = dataset.metadata;

    if cfg.verbose
        fprintf('\n');
        fprintf('====================================================================\n');
        fprintf('FINAL BLACK-SCHOLES RESULT\n');
        fprintf('====================================================================\n');
        fprintf('d                         = %d\n',cfg.dimension);
        fprintf('parameterization          = %s\n',cfg.optimizer.parameterization);
        fprintf('selected p=(r_s,r_t)      = [%.8f, %.8f]\n',p_opt(1),p_opt(2));
        fprintf('selected checkpoint       = %d\n',history.best_iteration);
        fprintf('selected residual MSE     = %.6e\n',history.best_selection_value);
        fprintf('relative L2               = %.6e\n',relative_l2);
        fprintf('setup                     = %.3f s\n',timing.setup);
        fprintf('optimization              = %.3f s\n',timing.optimization);
        fprintf('final assembly            = %.3f s\n',timing.final_assembly);
        fprintf('final solve               = %.3f s\n',timing.final_solve);
        fprintf('test                      = %.3f s\n',timing.test);
        fprintf('total solver time         = %.3f s\n',timing.total);
        fprintf('====================================================================\n');
    end

    if cfg.save_results
        if exist(cfg.output_dir,'dir') ~= 7
            mkdir(cfg.output_dir);
        end

        result_file = fullfile( ...
            cfg.output_dir,sprintf('black_scholes_d%d_results.mat',cfg.dimension));
        save(result_file,'results','-v7.3');

        T = black_scholes_training_history_table(history);
        history_file = fullfile( ...
            cfg.output_dir,sprintf('black_scholes_d%d_training_history.csv',cfg.dimension));
        writetable(T,history_file);

        fprintf('Result saved to:\n%s\n',result_file);
    end
end


function print_header(cfg,dataset)

    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('HIGH-DIMENSIONAL BLACK-SCHOLES | PDAD-ST\n');
    fprintf('====================================================================\n');
    fprintf('dimension                  = %d\n',cfg.dimension);
    fprintf('domain                     = [%.1f,%.1f]^d x [%.1f,%.1f]\n', ...
        cfg.x_lower,cfg.x_upper,cfg.t_domain(1),cfg.t_domain(2));
    fprintf('strike K                   = %.8g\n',cfg.payoff_strike);
    fprintf('activation                 = %s\n',cfg.activation);
    fprintf('full / train features      = %d / %d\n', ...
        cfg.num_features,resolve_train_features(cfg));
    fprintf('initial p                  = [%.6f, %.6f]\n', ...
        cfg.initial_p(1),cfg.initial_p(2));
    fprintf('optimizer parameterization= %s\n',cfg.optimizer.parameterization);
    fprintf('train rows (int/bnd/init)  = %d / %d / %d\n', ...
        size(dataset.train.interior_xt,1), ...
        size(dataset.train.boundary_xt,1), ...
        size(dataset.train.initial_xt,1));
    fprintf('solve rows (int/bnd/init)  = %d / %d / %d\n', ...
        size(dataset.solve.interior_xt,1), ...
        size(dataset.solve.boundary_xt,1), ...
        size(dataset.solve.initial_xt,1));
    fprintf('Monte-Carlo samples        = %d\n',dataset.metadata.num_mc_samples);
    fprintf('data file                  = %s\n',cfg.data_file);
    fprintf('====================================================================\n\n');
end


function m = resolve_train_features(cfg)
    if cfg.training_reduction.enabled
        m = cfg.training_reduction.num_features;
    else
        m = cfg.num_features;
    end
end
