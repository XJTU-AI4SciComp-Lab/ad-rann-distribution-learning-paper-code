function growth = burgers2d_growth_stage( ...
    p,basis,problem,base_coef,base_residual,previous_growth,cfg)
%BURGERS2D_GROWTH_STAGE Build or refresh one local DDAD growth block.
%
% Exact solution data are never used in rho training.  The centers are the
% m2 largest absolute current interior PDE residuals.  The DDAD target is
% the current pre-growth numerical base solution.

    if nargin < 6
        previous_growth = [];
    end

    t_total = tic;

    [centers,center_indices,center_scores] = ...
        select_growth_centers( ...
            problem.Xi, ...
            base_residual, ...
            cfg.m2, ...
            cfg.growth.center_policy);

    directions = build_growth_directions( ...
        2,cfg.m2,cfg.seed);

    base_i = evaluate_burgers2d_base_state( ...
        problem.Xi,p,basis,base_coef,2);

    center_state = evaluate_burgers2d_base_state( ...
        centers,p,basis,base_coef,0);

    center_values = center_state.u;
    y_target = base_i.u;

    if cfg.growth.target_noise ~= 0

        old_rng = rng;
        cleanup = onCleanup(@() rng(old_rng)); %#ok<NASGU>

        rng(cfg.seed,'twister');

        y_target = y_target + ...
            cfg.growth.target_noise*randn(size(y_target));
    end

    if cfg.growth.use_previous_rho && ...
       isstruct(previous_growth) && ...
       isfield(previous_growth,'rho') && ...
       isfinite(previous_growth.rho)

        rho0 = previous_growth.rho;

    else

        rho0 = cfg.growth.rho0;
    end

    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    t_train = tic;

    [model,rho,rho_history] = ...
        fit_growth_block_ddad( ...
            problem.Xi, ...
            y_target, ...
            base_i.phi, ...
            base_i.u, ...
            centers, ...
            directions, ...
            center_values, ...
            rho0, ...
            cfg.growth.lambda, ...
            ls_opts, ...
            cfg.growth.optimizer);

    training_time = toc(t_train);

    model.rho = rho;

    G_i = evaluate_growth_features( ...
        problem.Xi,model,rho,base_i,2);

    base_b = evaluate_burgers2d_base_state( ...
        problem.Xb,p,basis,base_coef,0);

    G_b = evaluate_growth_features( ...
        problem.Xb,model,rho,base_b,0);

    growth = struct();

    growth.active = true;

    growth.model = model;
    growth.rho = rho;
    growth.rho_history = rho_history;

    growth.centers = centers;
    growth.center_indices = center_indices;
    growth.center_scores = center_scores;
    growth.directions = directions;

    growth.frozen_p = p(:);
    growth.frozen_base_coef = base_coef(:);

    growth.Phi_i = G_i.phi;
    growth.Dsum_i = G_i.d1{1}+G_i.d1{2};
    growth.Lap_i = G_i.d2{1}+G_i.d2{2};
    growth.Phi_b = G_b.phi;

    growth.training_time = training_time;
    growth.total_build_time = toc(t_total);

    idx_best = selected_history_index(rho_history);

    fprintf('\n');
    fprintf('>>> DDAD layer-growth refresh\n');
    fprintf('    centers                    = %d\n',size(centers,1));
    fprintf('    largest |base residual|    = %.6e\n',center_scores(1));
    fprintf('    rho start / selected       = %.8f / %.8f\n',rho0,rho);
    fprintf('    selected Adam checkpoint   = %d\n', ...
        rho_history.best_iteration);

    if isfield(rho_history,'selection_metric')
        fprintf('    selection metric           = %s\n', ...
            rho_history.selection_metric);
    end

    fprintf('    selected residual MSE      = %.6e\n', ...
        rho_history.residual_mse(idx_best));
    fprintf('    selected ridge objective   = %.6e\n', ...
        rho_history.objective(idx_best));

    if isfield(rho_history,'stop_reason')
        fprintf('    Adam stop reason           = %s\n', ...
            rho_history.stop_reason);
    end

    fprintf('    evaluated checkpoints      = %d\n', ...
        numel(rho_history.iteration));
    fprintf('    growth training time       = %.3f s\n',training_time);
    fprintf('    total growth build time    = %.3f s\n\n', ...
        growth.total_build_time);
end


function idx = selected_history_index(history)

    if isfield(history,'best_index') && ...
       isfinite(history.best_index) && ...
       history.best_index >= 1 && ...
       history.best_index <= numel(history.iteration)

        idx = history.best_index;
        return;
    end

    idx = find( ...
        history.iteration == history.best_iteration, ...
        1,'first');

    if isempty(idx)
        idx = 1;
    end
end
