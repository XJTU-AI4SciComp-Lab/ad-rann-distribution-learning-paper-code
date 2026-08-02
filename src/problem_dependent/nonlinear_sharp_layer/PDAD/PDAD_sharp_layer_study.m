function results = PDAD_sharp_layer_study(cfg)
%PDAD_SHARP_LAYER_STUDY
%
% Global distribution:
%   PDAD.
%
% Growth:
%   DDAD-trained layer growth.
%
% Two independent training-reduction switches are resolved separately.

    out_dir = fullfile(cfg.project_root,cfg.output_dir_name);

    if exist(out_dir,'dir') ~= 7
        mkdir(out_dir);
    end

    problem = sharp_layer_problem(cfg);

    basis = ...
        build_random_weights( ...
            cfg.m1,cfg.domain,cfg.seed);

    ad_train = ...
        resolve_pdad_training(cfg,problem);

    basis_ad_train = ...
        subset_random_basis( ...
            basis,ad_train.m1);

    fprintf('\n');
    fprintf('============================================================================\n');
    fprintf('PDAD + DDAD LAYER GROWTH | NEWTON SOLVER\n');
    fprintf('============================================================================\n');
    fprintf('seed                            = %d\n',cfg.seed);
    fprintf('full m1 / m2                    = %d / %d\n',cfg.m1,cfg.m2);
    fprintf('all interior / boundary points  = %d / %d\n', ...
        size(problem.Xi,1),size(problem.Xb,1));
    fprintf('\n');
    fprintf('SWITCH 1: global PDAD reduction = %d\n',ad_train.enabled);
    fprintf('  PDAD train m1                  = %d / %d\n',ad_train.m1,cfg.m1);
    fprintf('  PDAD interior points           = %d / %d\n', ...
        ad_train.num_interior_points,size(problem.Xi,1));
    fprintf('  PDAD boundary points           = %d / %d\n', ...
        ad_train.num_boundary_points,size(problem.Xb,1));
    fprintf('\n');
    fprintf('SWITCH 2: growth reduction      = %d\n', ...
        cfg.growth_training_reduction.enabled);
    fprintf('  false => growth uses ALL Xi, full m1, full m2\n');
    fprintf('Newton maxit                    = %d\n',cfg.newton.maxit);
    fprintf('maximum GLOBAL AD updates       = %d\n',cfg.ad.max_updates);
    fprintf('Adam maxit / AD update          = %d\n',cfg.ad.optimizer.maxit);
    fprintf('growth Adam maxit               = %d\n',cfg.growth.optimizer.maxit);
    fprintf('growth Newton maxit             = %d\n',cfg.growth.newton_maxit);
    fprintf('============================================================================\n\n');

    base_timer = tic;

    base = ...
        solve_pdad_newton( ...
            problem,basis,basis_ad_train,ad_train,cfg);

    base_time = toc(base_timer);

    fprintf('\nBase PDAD finished:\n');
    fprintf('  Newton iterations = %d\n',base.newton_iterations);
    fprintf('  AD updates used   = %d/%d\n', ...
        base.ad_updates_used,cfg.ad.max_updates);
    fprintf('  p*                = [%.8f, %.8f]\n',base.p(1),base.p(2));
    fprintf('  rel L2            = %.6e\n',base.relative_l2);
    fprintf('  rel H1            = %.6e\n',base.relative_h1);
    fprintf('  rel Linf          = %.6e\n',base.relative_linf);
    fprintf('  method time       = %.3f s\n',base_time);

    growth_timer = tic;

    growth = ...
        sharp_layer_ddad_growth_stage( ...
            base.p,problem,basis,base,cfg);

    growth_stage_time = toc(growth_timer);
    total_time = base_time+growth_stage_time;

    fprintf('\n');
    fprintf('================================================================================\n');
    fprintf('FINAL REPORT: PDAD VS. PDAD + LAYER GROWTH\n');
    fprintf('================================================================================\n');
    fprintf('PDAD only:\n');
    fprintf('  time              = %.3f s\n',base_time);
    fprintf('  rel L2            = %.6e\n',base.relative_l2);
    fprintf('  rel H1            = %.6e\n',base.relative_h1);
    fprintf('  rel Linf          = %.6e\n',base.relative_linf);

    fprintf('\nPDAD + layer growth:\n');
    fprintf('  total time        = %.3f s\n',total_time);
    fprintf('  growth extra time = %.3f s\n',growth_stage_time);
    fprintf('    DDAD train time = %.3f s\n',growth.training_time);
    fprintf('    Newton time     = %.3f s\n',growth.newton_time);
    fprintf('  rho*              = %.8f\n',growth.rho);
    fprintf('  growth train pts  = %d\n',growth.training_num_points);
    fprintf('  growth train m1/m2= %d / %d\n', ...
        growth.training_m1,growth.training_m2);
    fprintf('  growth train ckpt = %d\n',growth.rho_history.best_iteration);
    fprintf('  growth Newton ckpt= %d\n',growth.selected_newton_checkpoint);
    fprintf('  rel L2            = %.6e\n',growth.relative_l2);
    fprintf('  rel H1            = %.6e\n',growth.relative_h1);
    fprintf('  rel Linf          = %.6e\n',growth.relative_linf);
    fprintf('================================================================================\n\n');

    Stage = {'PDAD only';'PDAD+growth'};

    m1 = [cfg.m1;cfg.m1];
    m2 = [0;cfg.m2];

    ADReduction = [ad_train.enabled;ad_train.enabled];
    ADTrainM1 = [ad_train.m1;ad_train.m1];
    ADTrainInteriorPoints = [ ...
        ad_train.num_interior_points; ...
        ad_train.num_interior_points];
    ADTrainBoundaryPoints = [ ...
        ad_train.num_boundary_points; ...
        ad_train.num_boundary_points];

    GrowthReduction = [false;growth.training_reduction_enabled];
    GrowthTrainPoints = [NaN;growth.training_num_points];
    GrowthTrainM1 = [NaN;growth.training_m1];
    GrowthTrainM2 = [NaN;growth.training_m2];

    ADUpdates = [base.ad_updates_used;base.ad_updates_used];

    BaseNewtonIters = [base.newton_iterations;base.newton_iterations];
    GrowthNewtonIters = [0;growth.newton_iterations];

    GrowthTrainCheckpoint = [NaN;growth.rho_history.best_iteration];
    GrowthNewtonCheckpoint = [NaN;growth.selected_newton_checkpoint];

    p1 = [base.p(1);base.p(1)];
    p2 = [base.p(2);base.p(2)];
    rho = [NaN;growth.rho];

    TimeSec = [base_time;total_time];
    GrowthExtraTimeSec = [0;growth_stage_time];
    GrowthDDADTrainTimeSec = [0;growth.training_time];
    GrowthNewtonTimeSec = [0;growth.newton_time];

    RelL2 = [base.relative_l2;growth.relative_l2];
    RelH1 = [base.relative_h1;growth.relative_h1];
    RelLinf = [base.relative_linf;growth.relative_linf];

    summary = table( ...
        Stage,m1,m2, ...
        ADReduction,ADTrainM1, ...
        ADTrainInteriorPoints,ADTrainBoundaryPoints, ...
        GrowthReduction,GrowthTrainPoints,GrowthTrainM1,GrowthTrainM2, ...
        ADUpdates,BaseNewtonIters,GrowthNewtonIters, ...
        GrowthTrainCheckpoint,GrowthNewtonCheckpoint, ...
        p1,p2,rho, ...
        TimeSec,GrowthExtraTimeSec, ...
        GrowthDDADTrainTimeSec,GrowthNewtonTimeSec, ...
        RelL2,RelH1,RelLinf);

    writetable( ...
        summary, ...
        fullfile(out_dir,'PDAD_sharp_layer_summary.csv'));

    results = struct();

    results.cfg = cfg;
    results.problem = problem;
    results.base = base;
    results.growth = growth;

    results.ad_training = ad_train;

    results.base_time = base_time;
    results.growth_stage_time = growth_stage_time;
    results.total_time = total_time;

    results.summary = summary;

    save( ...
        fullfile(out_dir,'PDAD_sharp_layer_newton_results.mat'), ...
        'results','-v7.3');
end


function ad_train = resolve_pdad_training(cfg,problem)

    if ~isfield(cfg,'ad_training_reduction')
        error('cfg.ad_training_reduction is missing.');
    end

    s = cfg.ad_training_reduction;

    required = { ...
        'enabled','m1_train', ...
        'num_interior_points','num_boundary_points'};

    for k = 1:numel(required)
        if ~isfield(s,required{k})
            error('cfg.ad_training_reduction.%s is missing.',required{k});
        end
    end

    ad_train = struct();
    ad_train.enabled = logical(s.enabled);

    Ni = size(problem.Xi,1);
    Nb = size(problem.Xb,1);

    if ad_train.enabled
        ad_train.m1 = s.m1_train;
        ad_train.num_interior_points = s.num_interior_points;
        ad_train.num_boundary_points = s.num_boundary_points;
    else
        ad_train.m1 = cfg.m1;
        ad_train.num_interior_points = Ni;
        ad_train.num_boundary_points = Nb;
    end

    validate_count(ad_train.m1,cfg.m1,'PDAD m1_train','m1');
    validate_count( ...
        ad_train.num_interior_points,Ni, ...
        'PDAD num_interior_points','all interior points');
    validate_count( ...
        ad_train.num_boundary_points,Nb, ...
        'PDAD num_boundary_points','all boundary points');

    ad_train.interior_indices = ...
        select_training_subset_indices( ...
            Ni,ad_train.num_interior_points,cfg.seed);

    ad_train.boundary_indices = ...
        select_training_subset_indices( ...
            Nb,ad_train.num_boundary_points,cfg.seed);
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


function base = solve_pdad_newton( ...
    problem,basis,basis_ad_train,ad_train,cfg)

    p = cfg.initial_p(:);

    op = prepare_base_operator( ...
        p,problem,basis,cfg.chunk_rows);

    coef = initialize_coefficients( ...
        op,problem,cfg);

    train_problem = ...
        make_pdad_training_problem( ...
            problem,ad_train);

    previous_residual_mse = Inf;
    ad_updates = 0;

    history = repmat(struct( ...
        'iteration',[], ...
        'residual_l2',[], ...
        'residual_mse',[], ...
        'relative_residual_change',[], ...
        'step_relative_l2',[], ...
        'accepted_alpha',[], ...
        'p',[], ...
        'ad_updates_used',[]), ...
        cfg.newton.maxit,1);

    ad_log = repmat(struct( ...
        'update_id',[], ...
        'newton_iteration',[], ...
        'p_before',[], ...
        'p_after',[], ...
        'selected_checkpoint',[], ...
        'training_time',[], ...
        'residual_mse_before',[]), ...
        cfg.ad.max_updates,1);

    for it = 1:cfg.newton.maxit

        [u_i,~,F_i,F_b,res_l2,res_mse] = ...
            nonlinear_residual(coef,op,problem);

        if isfinite(previous_residual_mse)
            relative_change = ...
                abs(previous_residual_mse-res_mse) / ...
                max(previous_residual_mse,eps);
        else
            relative_change = Inf;
        end

        history(it).iteration = it;
        history(it).residual_l2 = res_l2;
        history(it).residual_mse = res_mse;
        history(it).relative_residual_change = relative_change;
        history(it).step_relative_l2 = NaN;
        history(it).accepted_alpha = NaN;
        history(it).p = p;
        history(it).ad_updates_used = ad_updates;

        if cfg.newton.verbose
            fprintf(['Newton %02d/%02d | p=[%.4f,%.4f] | ', ...
                     'resL2=%.3e | resMSE=%.3e | dRes=%.3e | AD=%d/%d\n'], ...
                it,cfg.newton.maxit,p(1),p(2), ...
                res_l2,res_mse,relative_change, ...
                ad_updates,cfg.ad.max_updates);
        end

        if res_l2 <= cfg.newton.residual_tol
            history(it).step_relative_l2 = 0;
            break;
        end

        should_adapt = ...
            it < cfg.newton.maxit && ...
            it >= cfg.ad.min_newton_iteration && ...
            ad_updates < cfg.ad.max_updates && ...
            relative_change < cfg.ad.trigger_relative_change;

        if should_adapt

            update_id = ad_updates+1;

            fprintf('\n');
            fprintf('>>> COMPLETE PDAD UPDATE %d/%d at Newton iteration %d\n', ...
                update_id,cfg.ad.max_updates,it);
            fprintf('    p before            = [%.8f, %.8f]\n',p(1),p(2));
            fprintf('    AD train features   = %d / solve m1=%d\n', ...
                ad_train.m1,cfg.m1);
            fprintf('    AD interior points  = %d / all=%d\n', ...
                ad_train.num_interior_points,size(problem.Xi,1));
            fprintf('    AD boundary points  = %d / all=%d\n', ...
                ad_train.num_boundary_points,size(problem.Xb,1));
            fprintf('    inner Adam maxit    = %d\n',cfg.ad.optimizer.maxit);

            p_before = p;

            ls_opts = cfg.linear_solver;
            ls_opts.compute_spectrum = false;

            u_lin_train = ...
                u_i(ad_train.interior_indices);

            objective_fun = @(pp) ...
                evaluate_pdad_newton_reduced_fast( ...
                    pp,train_problem,basis_ad_train, ...
                    u_lin_train,cfg.ad.lambda, ...
                    ls_opts,cfg.chunk_rows);

            t_ad = tic;

            [p_new,ad_history] = ...
                optimize_distribution_adam( ...
                    p,objective_fun,cfg.ad.optimizer);

            ad_time = toc(t_ad);

            p = p_new(:);
            ad_updates = update_id;

            ad_log(update_id).update_id = update_id;
            ad_log(update_id).newton_iteration = it;
            ad_log(update_id).p_before = p_before;
            ad_log(update_id).p_after = p;
            ad_log(update_id).selected_checkpoint = ...
                ad_history.best_iteration;
            ad_log(update_id).training_time = ad_time;
            ad_log(update_id).residual_mse_before = res_mse;

            fprintf('    p after             = [%.8f, %.8f]\n',p(1),p(2));
            fprintf('    selected ckpt       = %d\n',ad_history.best_iteration);
            fprintf('    AD event time       = %.3f s\n\n',ad_time);

            op = prepare_base_operator( ...
                p,problem,basis,cfg.chunk_rows);

            coef = newton_next_state_in_new_space( ...
                u_i,op,problem,cfg);

            previous_residual_mse = Inf;
            continue;
        end

        J_i = op.minus_lap_i + 2*u_i.*op.Phi_i;
        J_b = problem.boundary_penalty*op.Phi_b;

        rhs = [-F_i;-problem.boundary_penalty*F_b];

        [delta_coef,~] = ...
            solve_least_squares( ...
                [J_i;J_b],rhs,cfg.linear_solver);

        delta_u = op.Phi_i*delta_coef;

        [coef,alpha,trial_res_l2,trial_res_mse,accepted] = ...
            take_newton_step( ...
                coef,delta_coef,op,problem, ...
                res_mse,cfg.newton);

        if ~accepted
            fprintf(['  Newton line search failed at iteration %d: ', ...
                     'no residual-decreasing step found.\n'],it);
            break;
        end

        step_rel = ...
            alpha*norm(delta_u)/max(norm(u_i),eps);

        history(it).step_relative_l2 = step_rel;
        history(it).accepted_alpha = alpha;

        if cfg.newton.verbose
            fprintf(['    accepted alpha=%.3e | ', ...
                     'next resL2=%.3e | next resMSE=%.3e\n'], ...
                alpha,trial_res_l2,trial_res_mse);
        end

        previous_residual_mse = res_mse;

        if step_rel <= cfg.newton.step_tol
            break;
        end
    end

    history = history(1:it);
    ad_log = ad_log(1:ad_updates);

    [u_i,~,F_i,~,res_l2,res_mse] = ...
        nonlinear_residual(coef,op,problem);

    test_state = ...
        evaluate_gaussian_state( ...
            problem.Xtest,p,basis,coef,1,cfg.chunk_rows);

    pred_test = test_state.u;

    grad_pred_test = zeros(size(problem.Xtest));

    for k = 1:size(problem.Xtest,2)
        grad_pred_test(:,k) = test_state.d1{k};
    end

    [err_l2,err_h1,err_linf] = ...
        sharp_layer_error_metrics( ...
            pred_test,grad_pred_test, ...
            problem.utest,problem.grad_utest);

    base = struct();

    base.p = p;
    base.coef = coef;
    base.u_interior = u_i;
    base.pde_residual = F_i;

    base.pred_test = pred_test;
    base.grad_pred_test = grad_pred_test;

    base.relative_l2 = err_l2;
    base.relative_h1 = err_h1;
    base.relative_linf = err_linf;

    base.final_residual_l2 = res_l2;
    base.final_residual_mse = res_mse;

    base.newton_iterations = it;
    base.newton_history = history;

    base.ad_updates_used = ad_updates;
    base.ad_log = ad_log;
end


function train_problem = make_pdad_training_problem(problem,ad_train)

    ii = ad_train.interior_indices;
    ib = ad_train.boundary_indices;

    train_problem = struct();

    train_problem.Xi = problem.Xi(ii,:);
    train_problem.fi = problem.fi(ii);

    train_problem.Xb = problem.Xb(ib,:);
    train_problem.gb = problem.gb(ib);

    train_problem.boundary_penalty = problem.boundary_penalty;
end


function [coef_new,alpha,res_l2,res_mse,accepted] = ...
    take_newton_step(coef,delta_coef,op,problem,current_res_mse,newton_cfg)

    if newton_cfg.line_search

        alpha = 1.0;
        accepted = false;
        res_l2 = Inf;
        res_mse = Inf;

        for ls_it = 1:newton_cfg.line_search_maxit

            coef_trial = coef + alpha*delta_coef;

            [~,~,~,~,res_l2,res_mse] = ...
                nonlinear_residual( ...
                    coef_trial,op,problem);

            if res_mse < current_res_mse
                accepted = true;
                coef_new = coef_trial;
                return;
            end

            alpha = newton_cfg.line_search_beta*alpha;

            if alpha < newton_cfg.line_search_min_alpha
                break;
            end
        end

        coef_new = coef;

    else

        alpha = newton_cfg.damping;
        accepted = true;

        coef_new = coef + alpha*delta_coef;

        [~,~,~,~,res_l2,res_mse] = ...
            nonlinear_residual(coef_new,op,problem);
    end
end


function [objective,grad,info] = ...
    evaluate_pdad_newton_reduced_fast( ...
        p,problem,basis,u_lin,lambda,ls_opts,chunk_rows)

    p = p(:);

    d = numel(p);

    Xi = problem.Xi;
    Xb = problem.Xb;

    Ni = size(Xi,1);
    Nb = size(Xb,1);

    m = size(basis.Z,2);

    eta = problem.boundary_penalty;

    y = [problem.fi+u_lin.^2;eta*problem.gb];

    M = zeros(Ni+Nb,m);

    for first = 1:chunk_rows:Ni

        rows = first:min(first+chunk_rows-1,Ni);
        X = Xi(rows,:);

        [z,phi] = preactivation_and_phi(X,p,basis);

        common2 = (4*z.^2-2).*phi;

        lap = zeros(numel(rows),m);

        for j = 1:d
            wj2 = (p(j)*basis.Z(j,:)).^2;
            lap = lap + common2.*wj2;
        end

        M(rows,:) = -lap + 2*u_lin(rows).*phi;
    end

    for first = 1:chunk_rows:Nb

        rows = first:min(first+chunk_rows-1,Nb);

        [~,phi] = preactivation_and_phi( ...
            Xb(rows,:),p,basis);

        M(Ni+rows,:) = eta*phi;
    end

    [w,ridge_info] = solve_ridge(M,y,lambda,ls_opts);

    if isa(w,'gpuArray')
        w = gather(w);
    end

    w = w(:);

    res = M*w-y;

    residual_sq = real(res'*res);
    coefficient_sq = real(w'*w);

    objective = ...
        (residual_sq+lambda*coefficient_sq)/(2*(Ni+Nb));

    grad = zeros(d,1);

    res_i = res(1:Ni);
    res_b = res(Ni+1:end);

    for k = 1:d

        accum = 0;

        for first = 1:chunk_rows:Ni

            rows = first:min(first+chunk_rows-1,Ni);
            X = Xi(rows,:);

            [z,phi] = preactivation_and_phi(X,p,basis);

            qk = ...
                (X(:,k)-basis.C(k,:)).*basis.Z(k,:);

            dphi = -2*z.*phi.*qk;

            F = (4*z.^2-2).*phi;
            Fprime = (12*z-8*z.^3).*phi;

            dlap = zeros(numel(rows),m);

            for j = 1:d

                Zj2 = basis.Z(j,:).^2;
                pj = p(j);

                term = ...
                    (pj^2)*Zj2.*(Fprime.*qk);

                if j == k
                    term = term + 2*pj*Zj2.*F;
                end

                dlap = dlap+term;
            end

            dA = -dlap + 2*u_lin(rows).*dphi;

            accum = accum + real(res_i(rows)'*(dA*w));
        end

        for first = 1:chunk_rows:Nb

            rows = first:min(first+chunk_rows-1,Nb);
            X = Xb(rows,:);

            [z,phi] = preactivation_and_phi(X,p,basis);

            qk = ...
                (X(:,k)-basis.C(k,:)).*basis.Z(k,:);

            dphi = -2*z.*phi.*qk;

            accum = ...
                accum + ...
                real(res_b(rows)'*((eta*dphi)*w));
        end

        grad(k) = accum/(Ni+Nb);
    end

    info = struct();
    info.residual_mse = residual_sq/(Ni+Nb);
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;
end


function coef = initialize_coefficients(op,problem,cfg)

    switch lower(cfg.newton.initialization)

        case 'linear_poisson'

            A = [ ...
                op.minus_lap_i; ...
                problem.boundary_penalty*op.Phi_b];

            b = [ ...
                problem.fi; ...
                problem.boundary_penalty*problem.gb];

            [coef,~] = ...
                solve_least_squares(A,b,cfg.linear_solver);

        otherwise
            error('Unknown Newton initialization: %s', ...
                cfg.newton.initialization);
    end
end


function coef = newton_next_state_in_new_space( ...
    u_current,op,problem,cfg)

    A = [ ...
        op.minus_lap_i + 2*u_current.*op.Phi_i; ...
        problem.boundary_penalty*op.Phi_b];

    b = [ ...
        problem.fi+u_current.^2; ...
        problem.boundary_penalty*problem.gb];

    [coef,~] = solve_least_squares(A,b,cfg.linear_solver);
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


function [S,Phi] = preactivation_and_phi(X,p,basis)

    S = build_preactivation(X,p,basis);
    Phi = evaluate_activation(S,'gaussian');

end


function [u_i,u_b,F_i,F_b,res_l2,res_mse] = ...
    nonlinear_residual(coef,op,problem)

    u_i = op.Phi_i*coef;
    u_b = op.Phi_b*coef;

    F_i = ...
        op.minus_lap_i*coef + ...
        u_i.^2 - ...
        problem.fi;

    F_b = u_b-problem.gb;

    eta = problem.boundary_penalty;

    weighted = [F_i;eta*F_b];

    scale = norm([problem.fi;eta*problem.gb]);

    res_l2 = norm(weighted)/max(scale,eps);
    res_mse = mean(abs(weighted).^2);
end
