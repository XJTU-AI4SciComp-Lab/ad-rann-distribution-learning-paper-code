function growth = sharp_layer_ddad_growth_stage( ...
    p,problem,basis,base,cfg)
%SHARP_LAYER_DDAD_GROWTH_STAGE
%
% Layer growth is ALWAYS trained by DDAD.
%
% IMPORTANT TRAINING-SET RULE:
%   By default, rho is trained on ALL configured PDE interior collocation
%   points problem.Xi and with ALL m1 base features + ALL m2 growth features.
%
%   Only when
%
%       cfg.growth_training_reduction.enabled = true
%
%   are the growth-training point count and growth-training feature counts
%   reduced.  The final augmented Newton solve always uses the full
%   cfg.m1 + cfg.m2 trial space.
%
% Workflow:
%   1. select the full m2 residual centers from ALL interior PDE points;
%   2. use the converged full-m1 NUMERICAL base solution as the DDAD target;
%   3. train rho using either all configured points/features or the
%      independently controlled reduced growth-training set;
%   4. append the full m2 localized features;
%   5. perform line-searched Newton corrections in the full augmented space;
%   6. return the historical-best augmented Newton checkpoint.
%
% The exact manufactured solution is never used to train rho.

    % =====================================================================
    % 1. Full residual-center selection
    % =====================================================================
    [centers,center_indices,center_scores] = ...
        select_growth_centers( ...
            problem.Xi, ...
            base.pde_residual, ...
            cfg.m2, ...
            cfg.growth.center_policy);

    directions = ...
        build_growth_directions( ...
            size(problem.Xi,2), ...
            cfg.m2, ...
            cfg.seed);

    center_state = ...
        evaluate_base_solution( ...
            centers,p,basis,base.coef,0,cfg.chunk_rows);

    center_values = center_state.u;

    % =====================================================================
    % 2. Resolve the SECOND, independent training-reduction switch
    % =====================================================================
    growth_train = resolve_growth_training(cfg,problem);

    basis_growth_train = ...
        subset_random_basis( ...
            basis, ...
            growth_train.m1);

    Xdata = ...
        problem.Xi(growth_train.indices,:);

    % The target is always evaluated from the FULL m1 base numerical
    % solution, even when a reduced m1 block is used to train rho.
    base_data_state = ...
        evaluate_base_solution( ...
            Xdata,p,basis,base.coef,0,cfg.chunk_rows);

    base_solution_data = base_data_state.u;

    noise_unit = make_fixed_noise(size(Xdata,1),cfg.seed);

    y_target = ...
        base_solution_data + ...
        cfg.data.noise_delta*noise_unit;

    Phi_data_train = ...
        base_features( ...
            Xdata,p,basis_growth_train);

    % The residual centers are ordered from largest to smaller absolute
    % residual.  A reduced growth block therefore uses the strongest
    % residual centers first.
    centers_train = centers(1:growth_train.m2,:);
    directions_train = directions(:,1:growth_train.m2);
    center_values_train = center_values(1:growth_train.m2);

    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    t_train = tic;

    [~,rho,growth_history] = ...
        fit_growth_block_ddad( ...
            Xdata, ...
            y_target, ...
            Phi_data_train, ...
            base_solution_data, ...
            centers_train, ...
            directions_train, ...
            center_values_train, ...
            cfg.growth.rho0, ...
            cfg.growth.lambda, ...
            ls_opts, ...
            cfg.growth.optimizer);

    training_time = toc(t_train);

    clear Phi_data_train y_target base_data_state base_solution_data;

    % Full growth model used by the PDE solve.
    growth_model = ...
        prepare_growth_model( ...
            centers, ...
            directions, ...
            center_values);

    growth_model.rho = rho;

    fprintf('\n');
    fprintf('>>> DDAD layer-growth training\n');
    fprintf('    target                   = converged numerical solution\n');
    fprintf('    exact target used        = 0\n');
    fprintf('    target noise             = %.3e\n',cfg.data.noise_delta);
    fprintf('    growth reduction enabled = %d\n',growth_train.enabled);
    fprintf('    growth train points      = %d / all Xi=%d\n', ...
        growth_train.num_points,size(problem.Xi,1));
    fprintf('    base train features      = %d / solve m1=%d\n', ...
        growth_train.m1,cfg.m1);
    fprintf('    growth train features    = %d / solve m2=%d\n', ...
        growth_train.m2,cfg.m2);
    fprintf('    largest |residual|       = %.6e\n',center_scores(1));
    fprintf('    rho*                     = %.8f\n',rho);
    fprintf('    selected train ckpt      = %d\n',growth_history.best_iteration);
    fprintf('    best train objective     = %.6e\n',growth_history.best_selection_value);
    fprintf('    training time            = %.3f s\n',training_time);

    % =====================================================================
    % 3. Full augmented Newton solve
    % =====================================================================
    t_newton = tic;

    augmented = ...
        solve_augmented_newton( ...
            p,problem,basis,base.coef,growth_model,rho,cfg);

    newton_time = toc(t_newton);

    growth = augmented;

    growth.rho = rho;
    growth.model = growth_model;
    growth.rho_history = growth_history;

    growth.center_indices = center_indices;
    growth.center_scores = center_scores;

    growth.training_reduction_enabled = growth_train.enabled;
    growth.training_num_points = growth_train.num_points;
    growth.training_m1 = growth_train.m1;
    growth.training_m2 = growth_train.m2;

    growth.training_time = training_time;
    growth.newton_time = newton_time;
end


function growth_train = resolve_growth_training(cfg,problem)

    if ~isfield(cfg,'growth_training_reduction')
        error('cfg.growth_training_reduction is missing.');
    end

    s = cfg.growth_training_reduction;

    required = {'enabled','m1_train','m2_train','num_points'};

    for k = 1:numel(required)
        if ~isfield(s,required{k})
            error('cfg.growth_training_reduction.%s is missing.',required{k});
        end
    end

    growth_train = struct();
    growth_train.enabled = logical(s.enabled);

    Nfull = size(problem.Xi,1);

    if growth_train.enabled
        growth_train.m1 = s.m1_train;
        growth_train.m2 = s.m2_train;
        growth_train.num_points = s.num_points;
    else
        growth_train.m1 = cfg.m1;
        growth_train.m2 = cfg.m2;
        growth_train.num_points = Nfull;
    end

    validate_count(growth_train.m1,cfg.m1,'growth m1_train','m1');
    validate_count(growth_train.m2,cfg.m2,'growth m2_train','m2');
    validate_count(growth_train.num_points,Nfull,'growth num_points','all interior points');

    growth_train.indices = ...
        select_training_subset_indices( ...
            Nfull,growth_train.num_points,cfg.seed);
end


function validate_count(value,full_value,name,full_name)

    if ~isscalar(value) || ~isfinite(value) || ...
            value < 1 || value ~= floor(value)
        error('%s must be a positive integer.',name);
    end

    if value > full_value
        error('%s=%d cannot exceed %s=%d.', ...
            name,value,full_name,full_value);
    end
end


function noise = make_fixed_noise(N,seed)

    old_rng = rng;
    cleanup = onCleanup(@() rng(old_rng)); %#ok<NASGU>

    rng(seed,'twister');
    noise = rand(N,1);
end


function out = solve_augmented_newton( ...
    p,problem,basis,base_coef,growth_model,rho,cfg)
%SOLVE_AUGMENTED_NEWTON
%
% Augmented Newton correction with residual-based backtracking line search.
% The historical state with the smallest nonlinear residual MSE is retained.

    Ni = size(problem.Xi,1);
    eta = problem.boundary_penalty;

    base_op = prepare_base_operator( ...
        p,problem,basis,cfg.chunk_rows);

    frozen_i = ...
        evaluate_base_solution( ...
            problem.Xi,p,basis,base_coef,2,cfg.chunk_rows);

    frozen_b = ...
        evaluate_base_solution( ...
            problem.Xb,p,basis,base_coef,0,cfg.chunk_rows);

    G_i = ...
        evaluate_growth_features( ...
            problem.Xi,growth_model,rho,frozen_i,2);

    G_b = ...
        evaluate_growth_features( ...
            problem.Xb,growth_model,rho,frozen_b,0);

    minus_lap_growth = zeros(Ni,cfg.m2);

    for k = 1:growth_model.dim
        minus_lap_growth = ...
            minus_lap_growth-G_i.d2{k};
    end

    Phi_i = [base_op.Phi_i,G_i.phi];
    Phi_b = [base_op.Phi_b,G_b.phi];

    minus_lap = [base_op.minus_lap_i,minus_lap_growth];

    coef = [base_coef;zeros(cfg.m2,1)];

    history = repmat(struct( ...
        'iteration',[], ...
        'residual_l2',[], ...
        'residual_mse',[], ...
        'step_relative_l2',[], ...
        'accepted_alpha',[]), ...
        cfg.growth.newton_maxit,1);

    best_coef = coef;
    best_res_l2 = Inf;
    best_res_mse = Inf;
    best_iteration = 0;

    num_newton_updates = 0;

    for it = 1:cfg.growth.newton_maxit

        [u_i,~,F_i,F_b,res_l2,res_mse] = ...
            nonlinear_residual( ...
                coef,Phi_i,Phi_b,minus_lap,problem);

        if res_mse < best_res_mse
            best_res_mse = res_mse;
            best_res_l2 = res_l2;
            best_coef = coef;
            best_iteration = it;
        end

        history(it).iteration = it;
        history(it).residual_l2 = res_l2;
        history(it).residual_mse = res_mse;
        history(it).step_relative_l2 = 0;
        history(it).accepted_alpha = NaN;

        if cfg.growth.verbose
            fprintf(['  Growth Newton %02d/%02d | ', ...
                     'resL2=%.3e | resMSE=%.3e\n'], ...
                it,cfg.growth.newton_maxit,res_l2,res_mse);
        end

        if res_l2 <= cfg.newton.residual_tol
            break;
        end

        J_i = minus_lap + 2*u_i.*Phi_i;
        J_b = eta*Phi_b;

        rhs = [-F_i;-eta*F_b];

        [delta_coef,~] = ...
            solve_least_squares( ...
                [J_i;J_b],rhs,cfg.linear_solver);

        delta_u = Phi_i*delta_coef;

        if cfg.growth.line_search

            alpha = 1.0;
            accepted = false;
            trial_res_l2 = Inf;
            trial_res_mse = Inf;

            for ls_it = 1:cfg.growth.line_search_maxit

                coef_trial = coef + alpha*delta_coef;

                [~,~,~,~,trial_res_l2,trial_res_mse] = ...
                    nonlinear_residual( ...
                        coef_trial,Phi_i,Phi_b,minus_lap,problem);

                if trial_res_mse < res_mse
                    accepted = true;
                    break;
                end

                alpha = cfg.growth.line_search_beta*alpha;

                if alpha < cfg.growth.line_search_min_alpha
                    break;
                end
            end

            if ~accepted
                if cfg.growth.verbose
                    fprintf(['    Growth Newton line search failed: ', ...
                             'no residual-decreasing step found.\n']);
                end
                break;
            end

            coef = coef_trial;

        else

            alpha = cfg.growth.newton_damping;
            coef = coef + alpha*delta_coef;

            [~,~,~,~,trial_res_l2,trial_res_mse] = ...
                nonlinear_residual( ...
                    coef,Phi_i,Phi_b,minus_lap,problem);
        end

        num_newton_updates = num_newton_updates+1;

        step_rel = ...
            alpha*norm(delta_u) / ...
            max(norm(u_i),eps);

        history(it).step_relative_l2 = step_rel;
        history(it).accepted_alpha = alpha;

        if cfg.growth.verbose
            fprintf(['    accepted alpha=%.3e | ', ...
                     'next resL2=%.3e | next resMSE=%.3e\n'], ...
                alpha,trial_res_l2,trial_res_mse);
        end

        if trial_res_mse < best_res_mse
            best_res_mse = trial_res_mse;
            best_res_l2 = trial_res_l2;
            best_coef = coef;
            best_iteration = it+1;
        end

        if step_rel <= cfg.newton.step_tol
            break;
        end
    end

    history = history(1:it);

    coef = best_coef;

    [u_i,~,F_i,~,res_l2,res_mse] = ...
        nonlinear_residual( ...
            coef,Phi_i,Phi_b,minus_lap,problem);

    if cfg.growth.verbose
        fprintf('\n');
        fprintf('  Selected growth Newton checkpoint = %d\n',best_iteration);
        fprintf('  Best growth residual L2           = %.6e\n',best_res_l2);
        fprintf('  Best growth residual MSE          = %.6e\n',best_res_mse);
    end

    m1 = numel(base_coef);

    coef_base_final = coef(1:m1);
    coef_growth = coef(m1+1:end);

    base_final_test = ...
        evaluate_base_solution( ...
            problem.Xtest,p,basis,coef_base_final,1,cfg.chunk_rows);

    frozen_test = ...
        evaluate_base_solution( ...
            problem.Xtest,p,basis,base_coef,1,cfg.chunk_rows);

    G_test = ...
        evaluate_growth_features( ...
            problem.Xtest,growth_model,rho,frozen_test,1);

    pred_test = ...
        base_final_test.u + ...
        G_test.phi*coef_growth;

    grad_pred = zeros(size(problem.Xtest));

    for k = 1:size(problem.Xtest,2)
        grad_pred(:,k) = ...
            base_final_test.d1{k} + ...
            G_test.d1{k}*coef_growth;
    end

    [err_l2,err_h1,err_linf] = ...
        sharp_layer_error_metrics( ...
            pred_test,grad_pred, ...
            problem.utest,problem.grad_utest);

    out = struct();

    out.coef = coef;
    out.u_interior = u_i;
    out.pde_residual = F_i;

    out.pred_test = pred_test;
    out.grad_pred_test = grad_pred;

    out.relative_l2 = err_l2;
    out.relative_h1 = err_h1;
    out.relative_linf = err_linf;

    out.final_residual_l2 = res_l2;
    out.final_residual_mse = res_mse;

    out.newton_iterations = it;
    out.newton_updates = num_newton_updates;
    out.newton_history = history;

    out.selected_newton_checkpoint = best_iteration;
    out.best_newton_residual_l2 = best_res_l2;
    out.best_newton_residual_mse = best_res_mse;
end


function op = prepare_base_operator(p,problem,basis,chunk_rows)

    p = p(:);

    Xi = problem.Xi;
    Xb = problem.Xb;

    Ni = size(Xi,1);
    Nb = size(Xb,1);

    m = size(basis.Z,2);
    d = numel(p);

    Phi_i = zeros(Ni,m);
    minus_lap_i = zeros(Ni,m);

    for first = 1:chunk_rows:Ni

        rows = first:min(first+chunk_rows-1,Ni);
        X = Xi(rows,:);

        [z,phi] = preactivation_and_phi(X,p,basis);

        common2 = (4*z.^2-2).*phi;

        lap = zeros(numel(rows),m);

        for k = 1:d
            wk2 = (p(k)*basis.Z(k,:)).^2;
            lap = lap + common2.*wk2;
        end

        Phi_i(rows,:) = phi;
        minus_lap_i(rows,:) = -lap;
    end

    Phi_b = zeros(Nb,m);

    for first = 1:chunk_rows:Nb

        rows = first:min(first+chunk_rows-1,Nb);

        [~,phi] = preactivation_and_phi( ...
            Xb(rows,:),p,basis);

        Phi_b(rows,:) = phi;
    end

    op = struct();
    op.Phi_i = Phi_i;
    op.Phi_b = Phi_b;
    op.minus_lap_i = minus_lap_i;
end


function out = evaluate_base_solution( ...
    X,p,basis,coef,max_order,chunk_rows)

    p = p(:);
    coef = coef(:);

    N = size(X,1);
    d = numel(p);
    m = size(basis.Z,2);

    out = struct();

    out.u = zeros(N,1);
    out.d1 = cell(d,1);
    out.d2 = cell(d,1);

    if max_order >= 1
        for k = 1:d
            out.d1{k} = zeros(N,1);
        end
    end

    if max_order >= 2
        for k = 1:d
            out.d2{k} = zeros(N,1);
        end
    end

    for first = 1:chunk_rows:N

        rows = first:min(first+chunk_rows-1,N);
        Xc = X(rows,:);

        [z,phi] = preactivation_and_phi(Xc,p,basis);

        out.u(rows) = phi*coef;

        if max_order >= 1
            common1 = -2*z.*phi;

            for k = 1:d
                wk = p(k)*basis.Z(k,:);
                out.d1{k}(rows) = (common1.*wk)*coef;
            end
        end

        if max_order >= 2
            common2 = (4*z.^2-2).*phi;

            for k = 1:d
                wk2 = (p(k)*basis.Z(k,:)).^2;
                out.d2{k}(rows) = (common2.*wk2)*coef;
            end
        end
    end
end


function Phi = base_features(X,p,basis)

    [~,Phi] = preactivation_and_phi(X,p,basis);
end


function [z,phi] = preactivation_and_phi(X,p,basis)

    N = size(X,1);
    m = size(basis.Z,2);

    z = zeros(N,m);

    for k = 1:numel(p)
        z = z + ...
            (X(:,k)-basis.C(k,:)).* ...
            (p(k)*basis.Z(k,:));
    end

    phi = exp(-(z.^2));
end


function [u_i,u_b,F_i,F_b,res_l2,res_mse] = ...
    nonlinear_residual(coef,Phi_i,Phi_b,minus_lap,problem)

    u_i = Phi_i*coef;
    u_b = Phi_b*coef;

    F_i = ...
        minus_lap*coef + ...
        u_i.^2 - ...
        problem.fi;

    F_b = ...
        u_b - ...
        problem.gb;

    eta = problem.boundary_penalty;

    weighted = [F_i;eta*F_b];

    scale = norm([problem.fi;eta*problem.gb]);

    res_l2 = norm(weighted)/max(scale,eps);
    res_mse = mean(abs(weighted).^2);
end
