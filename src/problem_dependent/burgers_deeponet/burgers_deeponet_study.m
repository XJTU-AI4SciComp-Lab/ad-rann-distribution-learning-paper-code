function results = burgers_deeponet_study(cfg,project_root)
%BURGERS_DEEPONET_STUDY Paper-aligned data-driven RaNN-DeepONet study.

    load_timer = tic;
    data = load_burgers_deeponet_data(project_root,cfg);
    data_loading_time = toc(load_timer);

    sampling_timer = tic;
    training = sample_burgers_deeponet_training(data,cfg);

    reduced_cfg = cfg;
    reduced_cfg.training.num_rows = cfg.reduced.num_rows;
    reduced_cfg.training.sample_seed = cfg.reduced.sample_seed;
    reduced = sample_burgers_deeponet_training(data,reduced_cfg);
    sampling_time = toc(sampling_timer);

    if cfg.verbose
        fprintf(['Training rows: %d = %d initial + %d boundary ', ...
            '+ %d interior\n'],training.num_rows, ...
            training.category_counts(1),training.category_counts(2), ...
            training.category_counts(3));
        fprintf('Model widths: branch=%d, trunk=%d (%d coefficients)\n', ...
            cfg.model.num_branch,cfg.model.num_trunk, ...
            cfg.model.num_branch*cfg.model.num_trunk);
        fprintf(['DDAD reduced rows: %d; widths: branch=%d, ', ...
            'trunk=%d\n'], ...
            reduced.num_rows,cfg.reduced.num_branch, ...
            cfg.reduced.num_trunk);
    end

    training_timer = tic;

    reduced_basis = build_burgers_deeponet_basis( ...
        cfg.model.num_sensors,cfg.reduced.num_branch, ...
        cfg.reduced.num_trunk,cfg.seed+1000);

    objective_fun = @(p) ...
        evaluate_burgers_deeponet_ddad_reduced( ...
            p,reduced,reduced_basis,cfg);

    ddad_timer = tic;
    [p,ddad_history] = optimize_distribution_adam( ...
        cfg.ddad.initial_p,objective_fun,cfg.ddad.optimizer);
    ddad_time = toc(ddad_timer);

    clear reduced reduced_basis

    basis = build_burgers_deeponet_basis( ...
        cfg.model.num_sensors,cfg.model.num_branch, ...
        cfg.model.num_trunk,cfg.seed);

    final_fit_timer = tic;
    [model,fit_info] = fit_burgers_deeponet( ...
        training,basis,p,cfg);
    final_fit_time = toc(final_fit_timer);
    training_time = toc(training_timer);

    test_timer = tic;
    test = evaluate_burgers_deeponet_test(data,model,cfg);
    test_time = toc(test_timer);

    output_dir = fullfile(project_root,cfg.output_root_name);

    if exist(output_dir,'dir') ~= 7
        mkdir(output_dir);
    end

    results = struct();
    results.cfg = cfg;
    results.model = model;
    results.fit = fit_info;
    results.ddad = ddad_history;
    results.test = test;
    results.timing = struct( ...
        'training',training_time, ...
        'ddad',ddad_time, ...
        'final_fit',final_fit_time, ...
        'data_loading',data_loading_time, ...
        'training_sampling',sampling_time, ...
        'testing',test_time);
    results.output_dir = output_dir;

    save(fullfile(output_dir,'burgers_deeponet_results.mat'), ...
        'results','-v7.3');

    fprintf('Optimized p=[rb,rx,rt] = %s\n',mat2str(p.',8));
    fprintf('DDAD time: %.3f s\n',ddad_time);
    fprintf('Final coefficient-fit time: %.3f s\n',final_fit_time);
    fprintf('Total training time: %.3f s\n',training_time);
    fprintf('Mean test relative L2: %.6e\n',test.mean_relative_l2);
    fprintf('Worst test relative L2: %.6e (test sample %d)\n', ...
        test.worst_relative_l2,test.worst_index);
    fprintf('Results saved to: %s\n',output_dir);
end
