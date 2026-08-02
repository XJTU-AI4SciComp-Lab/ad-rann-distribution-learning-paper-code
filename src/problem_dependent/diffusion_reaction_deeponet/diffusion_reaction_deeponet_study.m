function results = diffusion_reaction_deeponet_study(cfg,project_root)
%DIFFUSION_REACTION_DEEPONET_STUDY RaNN/PI-RaNN DeepONet with DDAD.
%
% Modes:
%   RANN_DDAD    - supervised coefficient fit and supervised DDAD.
%   PI_RANN_DDAD - PDE coefficient fit; DDAD targets the current physical
%                  solution and is capped by cfg.physics.max_ddad_updates.

    mode = upper(strtrim(char(cfg.mode)));

    if ~ismember(mode,{'RANN_DDAD','PI_RANN_DDAD'})
        error('Unsupported cfg.mode: %s',mode);
    end

    data = load_diffusion_reaction_deeponet_data(project_root,cfg);
    training_source = prepare_training_source(data,mode);

    full_basis = build_deeponet_basis( ...
        cfg.model.num_sensors, ...
        cfg.model.num_branch, ...
        cfg.model.num_trunk, ...
        cfg.seed);

    reduced_basis = build_deeponet_basis( ...
        cfg.model.num_sensors, ...
        cfg.reduced.num_branch, ...
        cfg.reduced.num_trunk, ...
        cfg.seed+1000);

    p = cfg.initial_p(:);
    max_history_count = max(1,cfg.physics.max_ddad_updates);
    ddad_histories = cell(max_history_count,1);
    update_log = struct( ...
        'update_id',{}, ...
        'p_before',{}, ...
        'p_after',{}, ...
        'target_source',{}, ...
        'best_iteration',{}, ...
        'best_selection_value',{}, ...
        'training_time',{});

    training_points_per_function = ...
        cfg.data.points_per_function;

    switch mode

        case 'RANN_DDAD'

            reduced = sample_diffusion_reaction_rows( ...
                training_source, ...
                cfg.reduced.num_training_rows, ...
                cfg.seed+10,true,training_points_per_function);

            full = sample_diffusion_reaction_rows( ...
                training_source, ...
                cfg.model.num_training_rows, ...
                cfg.seed+20,true,training_points_per_function);

            clear training_source

            objective_fun = @(pp) ...
                evaluate_deeponet_ddad_reduced( ...
                    pp,reduced.F,reduced.Y,reduced.u, ...
                    reduced_basis,cfg.ddad.lambda, ...
                    cfg.linear_solver,cfg.activation);

            training_timer = tic;
            train_timer = tic;
            [p,ddad_history] = optimize_distribution_adam( ...
                p,objective_fun,cfg.ddad.optimizer);
            ddad_time = toc(train_timer);

            ddad_histories{1,1} = ddad_history;
            update_log(1) = make_update_log( ...
                1,cfg.initial_p(:),p,'reference solution', ...
                ddad_history,ddad_time);

            clear reduced

            [model,fit_info] = fit_deeponet_data( ...
                full.F,full.Y,full.u,p,full_basis,cfg);

            physics_history = {};

        case 'PI_RANN_DDAD'

            full = sample_diffusion_reaction_rows( ...
                training_source, ...
                cfg.model.num_training_rows, ...
                cfg.seed+20,false,training_points_per_function);

            reduced_batches = cell( ...
                cfg.physics.max_ddad_updates,1);

            for update_id = 1:cfg.physics.max_ddad_updates
                reduced_batches{update_id} = ...
                    sample_diffusion_reaction_rows( ...
                        training_source, ...
                        cfg.reduced.num_training_rows, ...
                        cfg.seed+100+update_id,false, ...
                        training_points_per_function);
            end

            clear training_source

            rhs = evaluate_deeponet_forcing( ...
                full.F,full.Y(:,1));

            training_timer = tic;

            [model,initial_physics_info] = ...
                fit_deeponet_physics_picard( ...
                    full.F,full.Y,rhs,p,full_basis,cfg,[]);

            physics_history = cell( ...
                cfg.physics.max_ddad_updates+1,1);
            physics_history{1} = initial_physics_info;
            physics_count = 1;
            fit_info = initial_physics_info;

            for update_id = 1:cfg.physics.max_ddad_updates

                reduced = reduced_batches{update_id};

                target = predict_deeponet_rows( ...
                    reduced.F,reduced.Y,model,cfg.activation);

                objective_fun = @(pp) ...
                    evaluate_deeponet_ddad_reduced( ...
                        pp,reduced.F,reduced.Y,target, ...
                        reduced_basis,cfg.ddad.lambda, ...
                        cfg.linear_solver,cfg.activation);

                p_before = p;
                train_timer = tic;

                [p_new,ddad_history] = ...
                    optimize_distribution_adam( ...
                        p,objective_fun,cfg.ddad.optimizer);

                ddad_time = toc(train_timer);
                p = p_new(:);

                ddad_histories{update_id,1} = ddad_history;
                update_log(update_id) = make_update_log( ...
                    update_id,p_before,p, ...
                    'current Picard physical solution', ...
                    ddad_history,ddad_time);

                if cfg.verbose
                    fprintf('\n');
                    fprintf('PI-RaNN DDAD update %d/%d\n', ...
                        update_id,cfg.physics.max_ddad_updates);
                    fprintf('p [rb rx rt]: %s -> %s\n', ...
                        mat2str(p_before.',7),mat2str(p.',7));
                    fprintf('\n');
                end

                [model,fit_info] = ...
                    fit_deeponet_physics_picard( ...
                        full.F,full.Y,rhs,p,full_basis,cfg,model.W);

                physics_count = physics_count+1;
                physics_history{physics_count} = fit_info;

                relative_p_change = ...
                    norm(p-p_before)/max(norm(p),eps);

                clear reduced target

                if relative_p_change <= ...
                        cfg.physics.ddad_update_tolerance
                    break;
                end
            end

            physics_history = physics_history(1:physics_count);
            clear reduced_batches
    end

    ddad_histories = ddad_histories(1:numel(update_log));

    training_time = toc(training_timer);

    test_metrics = evaluate_diffusion_reaction_deeponet_test( ...
        model,data,cfg);

    results = struct();
    results.cfg = cfg;
    results.mode = mode;
    results.data = data;
    results.model = model;
    results.final_p = model.p;
    results.fit_info = fit_info;
    results.ddad_histories = ddad_histories;
    results.update_log = update_log;
    results.physics_history = physics_history;
    results.num_ddad_updates = numel(update_log);
    results.training_time = training_time;
    results.test_metrics = test_metrics;

    output_dir = fullfile( ...
        project_root,cfg.output_root_name,mode);

    if exist(output_dir,'dir') ~= 7
        mkdir(output_dir);
    end

    save(fullfile(output_dir,[mode '_results.mat']), ...
        'results','-v7.3');

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('%s FINAL REPORT\n',strrep(mode,'_','-'));
    fprintf('============================================================\n');
    fprintf('DDAD updates            = %d\n', ...
        results.num_ddad_updates);
    fprintf('final p [rb rx rt]      = %s\n', ...
        mat2str(results.final_p.',8));
    fprintf('training time           = %.3f s\n',training_time);
    fprintf('overall relative L2     = %.8e\n', ...
        test_metrics.overall_relative_l2);
    fprintf('mean per-function L2    = %.8e\n', ...
        test_metrics.mean_relative_l2);
    fprintf('results directory       = %s\n',output_dir);
    fprintf('============================================================\n');
end


function source = prepare_training_source(data,mode)

    if data.train_supports_partial
        source = data.train_file;
        return;
    end

    warning([ ...
        'The training MAT file is not v7.3. Loading required variables ', ...
        'once in their stored precision.']);

    if strcmp(mode,'RANN_DDAD')
        source = load(data.train_file,'f','y','u');
    else
        source = load(data.train_file,'f','y');
    end
end


function entry = make_update_log( ...
    update_id,p_before,p_after,target_source,history,training_time)

    entry = struct();
    entry.update_id = update_id;
    entry.p_before = p_before(:);
    entry.p_after = p_after(:);
    entry.target_source = target_source;
    entry.best_iteration = history.best_iteration;
    entry.best_selection_value = history.best_selection_value;
    entry.training_time = training_time;
end
