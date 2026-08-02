function results = burgers_2d_dt_study(cfg,project_root)
%BURGERS_2D_DT_STUDY
% 2-D Burgers PDAD-DT / DDAD-DT solver with optional layer growth.
%
% The exact solution is used only for time-dependent Dirichlet data and
% post-processing errors.  It is never used as a DDAD update target or a
% layer-growth training target.

    method = upper(strtrim(char(cfg.method)));

    if ~ismember(method,{'DDAD','PDAD'})
        error('cfg.method must be DDAD or PDAD.');
    end

    validate_configuration(cfg);

    ref = build_burgers2d_reference(cfg);
    problem = build_burgers2d_problem(cfg);

    t0 = cfg.t_domain(1);
    tf = cfg.t_domain(2);

    Nt = cfg.num_time_steps;
    dt = (tf-t0)/Nt;

    snapshot_steps_real = (ref.tt-t0)/dt;
    snapshot_steps = round(snapshot_steps_real);

    if max(abs(snapshot_steps_real-snapshot_steps)) > 1e-10
        error('Exact snapshot times do not align with Nt=%d.',Nt);
    end

    % =====================================================================
    % Frozen random realization
    % =====================================================================
    basis = build_random_weights_nd( ...
        cfg.m1,cfg.domain,cfg.seed);

    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('%s-DT 2-D BURGERS\n',method);
    fprintf('============================================================\n');
    fprintf('epsilon                    = %.6e\n',cfg.epsilon_burgers);
    fprintf('Nt / dt                    = %d / %.6e\n',Nt,dt);
    fprintf('interior points            = %d x %d = %d\n', ...
        cfg.num_collocation_x,cfg.num_collocation_y, ...
        problem.num_interior);
    fprintf('boundary points            = %d (%d per side)\n', ...
        problem.num_boundary,cfg.num_boundary_per_side);
    fprintf('m1 / m2                    = %d / %d\n',cfg.m1,cfg.m2);
    fprintf('layer growth enabled       = %d\n',cfg.growth.enabled);
    fprintf('growth refresh policy      = %s\n', ...
        cfg.growth.refresh_policy);
    fprintf('saved exact snapshots      = %d\n',cfg.num_saved_snapshots);
    fprintf('seed                       = %d\n',cfg.seed);
    fprintf('============================================================\n\n');

    % =====================================================================
    % Initial distribution
    % =====================================================================
    initial_history = [];
    initial_training_time = 0;

    switch lower(strtrim(char(cfg.initialization.mode)))

        case 'ddad'

            cache0 = prepare_data_cache( ...
                problem.Xi,problem.u0,basis);

            objective0 = @(pp) ...
                evaluate_data_reduced_fast( ...
                    pp,cache0, ...
                    cfg.initialization.lambda, ...
                    ls_opts,cfg.activation);

            t_init = tic;

            [p,initial_history] = ...
                optimize_distribution_adam( ...
                    cfg.initialization.p0, ...
                    objective0, ...
                    cfg.initialization.optimizer);

            initial_training_time = toc(t_init);

        case 'fixed'

            p = cfg.initialization.fixed_p;

        otherwise

            error('initialization.mode must be ddad or fixed.');
    end

    p = p(:);

    if numel(p) ~= 2
        error('The initial p must contain two values.');
    end

    total_timer = tic;

    % =====================================================================
    % Storage
    % =====================================================================
    ne = cfg.evaluation_grid_size;
    ns = cfg.num_saved_snapshots;

    pred_snapshots = nan(ne,ne,ns);

    p_history = nan(Nt+1,2);
    rho_history = nan(Nt+1,1);

    base_residual_history = nan(Nt+1,1);
    final_residual_history = nan(Nt+1,1);
    feature_count_history = nan(Nt+1,1);

    p_history(1,:) = p.';
    feature_count_history(1) = cfg.m1;

    update_log = empty_update_log();
    growth_refresh_log = empty_growth_log();

    update_histories = {};
    growth_histories = {};

    num_update_attempts = 0;
    num_effective_updates = 0;
    num_growth_refreshes = 0;

    growth = [];

    base_op = prepare_burgers2d_base_operator( ...
        p,problem,basis);

    eta = cfg.boundary_penalty;

    % =====================================================================
    % Step 1: first-order frozen-coefficient IMEX step
    %
    %   u^1 + dt*u^0*(u_x^1+u_y^1) - dt*eps*Delta u^1 = u^0.
    % =====================================================================
    k = 1;
    t_now = t0+dt;

    a0 = 1;
    convection_weight = dt;
    diffusion_weight = dt*cfg.epsilon_burgers;

    qhat = problem.u0;
    rhs_i = problem.u0;

    boundary_values = burgers2d_exact( ...
        problem.Xb,t_now,cfg.epsilon_burgers);
    rhs_b = eta*boundary_values;

    [base_coef,base_residual,Ebase] = ...
        solve_base_candidate( ...
            base_op,qhat,rhs_i,rhs_b, ...
            a0,convection_weight,diffusion_weight,eta,cfg);

    if cfg.growth.enabled && cfg.growth.build_at_first_step

        growth = burgers2d_growth_stage( ...
            p,basis,problem,base_coef,base_residual,[],cfg);

        num_growth_refreshes = 1;

        growth_refresh_log(1) = make_growth_record( ...
            1,k,t_now,growth);

        growth_histories{1,1} = growth.rho_history;
    end

    [coef_base_current,coef_growth_current,u_new,Efinal] = ...
        solve_accepted_candidate( ...
            base_op,growth,qhat,rhs_i,rhs_b, ...
            a0,convection_weight,diffusion_weight,eta,cfg);

    u_ll = problem.u0;
    u_l = u_new;

    El = Ebase;
    last_base_residual = base_residual;

    p_history(2,:) = p.';
    base_residual_history(2) = Ebase;
    final_residual_history(2) = Efinal;

    if growth_is_active(growth)
        rho_history(2) = growth.rho;
        feature_count_history(2) = cfg.m1+cfg.m2;
    else
        feature_count_history(2) = cfg.m1;
    end

    store_snapshot( ...
        1,p,coef_base_current,growth,coef_growth_current);

    fprintf([ ...
        'step=%4d/%4d | t=%.6f | baseE=%.6e | ', ...
        'finalE=%.6e | p=[%.6f %.6f] | rho=%s\n'], ...
        1,Nt,t_now,Ebase,Efinal,p(1),p(2),rho_text(growth));

    % =====================================================================
    % BDF2 time stepping
    %
    %   3u^k + 2dt*(2u^{k-1}-u^{k-2})(u_x^k+u_y^k)
    %        - 2dt*eps*Delta u^k = 4u^{k-1}-u^{k-2}.
    % =====================================================================
    k = 2;
    update_attempted_this_step = false;
    p_changed_this_step = false;

    while k <= Nt

        t_now = t0+k*dt;

        a0 = 3;
        convection_weight = 2*dt;
        diffusion_weight = 2*dt*cfg.epsilon_burgers;

        qhat = 2*u_l-u_ll;
        rhs_i = 4*u_l-u_ll;

        boundary_values = burgers2d_exact( ...
            problem.Xb,t_now,cfg.epsilon_burgers);
        rhs_b = eta*boundary_values;

        [base_coef,base_residual,Ebase] = ...
            solve_base_candidate( ...
                base_op,qhat,rhs_i,rhs_b, ...
                a0,convection_weight,diffusion_weight,eta,cfg);

        Delta_plus = ...
            max(Ebase-El,0) / ...
            max(abs(El),cfg.adaptation.residual_epsilon);

        Delta_l = ...
            abs(Ebase-El) / ...
            max( ...
                min(abs(El),abs(Ebase)), ...
                cfg.adaptation.residual_epsilon);

        accept_global = ...
            Delta_plus < cfg.adaptation.tau_k || ...
            update_attempted_this_step;

        if ~accept_global

            p_before = p;

            if strcmp(method,'DDAD')

                cache = prepare_data_cache( ...
                    problem.Xi,u_l,basis);

                objective_fun = @(pp) ...
                    evaluate_data_reduced_fast( ...
                        pp,cache,cfg.ddad.lambda, ...
                        ls_opts,cfg.activation);

                opt_cfg = cfg.ddad.optimizer;

            else

                objective_fun = @(pp) ...
                    evaluate_burgers2d_pdad_reduced_fast( ...
                        pp, ...
                        problem.Xi, ...
                        problem.Xb, ...
                        basis, ...
                        qhat, ...
                        rhs_i, ...
                        boundary_values, ...
                        a0, ...
                        convection_weight, ...
                        diffusion_weight, ...
                        eta, ...
                        cfg.pdad.lambda, ...
                        ls_opts);

                opt_cfg = cfg.pdad.optimizer;
            end

            t_update = tic;

            [p_candidate,ad_history] = ...
                optimize_distribution_adam( ...
                    p,objective_fun,opt_cfg);

            update_time = toc(t_update);

            p_candidate = p_candidate(:);

            relative_p_change = ...
                norm(p_candidate-p_before) / ...
                max(1,norm(p_before));

            update_effective = ...
                ad_history.best_iteration > 0 && ...
                relative_p_change > ...
                    cfg.adaptation.min_relative_parameter_change;

            num_update_attempts = num_update_attempts+1;

            if update_effective

                p = p_candidate;
                num_effective_updates = num_effective_updates+1;

                base_op = prepare_burgers2d_base_operator( ...
                    p,problem,basis);

                p_changed_this_step = true;

            else

                p = p_before;
                p_changed_this_step = false;
            end

            update_log(num_update_attempts) = ...
                make_update_record( ...
                    num_update_attempts,k,t_now,method, ...
                    p_before,p_candidate,p, ...
                    Ebase,El,Delta_plus, ...
                    ad_history,update_time, ...
                    relative_p_change,update_effective);

            update_histories{num_update_attempts,1} = ad_history;

            fprintf('\n');
            fprintf('>>> %s global update attempt %d at step %d\n', ...
                method,num_update_attempts,k);
            fprintf('    p candidate: [%.8f %.8f] -> [%.8f %.8f]\n', ...
                p_before(1),p_before(2), ...
                p_candidate(1),p_candidate(2));
            fprintf('    selected checkpoint      = %d\n', ...
                ad_history.best_iteration);
            fprintf('    best selection value     = %.6e\n', ...
                ad_history.best_selection_value);
            fprintf('    relative p change        = %.6e\n', ...
                relative_p_change);
            fprintf('    effective update         = %d\n', ...
                update_effective);
            fprintf('    update time              = %.3f s\n\n', ...
                update_time);

            update_attempted_this_step = true;

            if Delta_l > cfg.adaptation.tau_l
                El = Ebase;
            end

            % Re-solve this same physical step.  If p did not change, the
            % second pass is accepted but no basis or growth refresh occurs.
            continue;
        end

        do_refresh = should_refresh_growth( ...
            cfg,growth,p_changed_this_step);

        if do_refresh

            previous_growth = growth;

            growth = burgers2d_growth_stage( ...
                p,basis,problem,base_coef, ...
                base_residual,previous_growth,cfg);

            num_growth_refreshes = num_growth_refreshes+1;

            growth_refresh_log(num_growth_refreshes) = ...
                make_growth_record( ...
                    num_growth_refreshes,k,t_now,growth);

            growth_histories{num_growth_refreshes,1} = ...
                growth.rho_history;
        end

        [coef_base_current,coef_growth_current,u_new,Efinal] = ...
            solve_accepted_candidate( ...
                base_op,growth,qhat,rhs_i,rhs_b, ...
                a0,convection_weight,diffusion_weight,eta,cfg);

        u_ll = u_l;
        u_l = u_new;

        last_base_residual = base_residual;

        p_history(k+1,:) = p.';
        base_residual_history(k+1) = Ebase;
        final_residual_history(k+1) = Efinal;

        if growth_is_active(growth)

            rho_history(k+1) = growth.rho;
            feature_count_history(k+1) = cfg.m1+cfg.m2;

        else

            feature_count_history(k+1) = cfg.m1;
        end

        store_snapshot( ...
            k,p,coef_base_current,growth,coef_growth_current);

        if cfg.verbose && ...
           (cfg.print_every <= 1 || ...
            mod(k,cfg.print_every) == 0 || ...
            k == Nt)

            fprintf([ ...
                'step=%4d/%4d | t=%.6f | baseE=%.6e | ', ...
                'finalE=%.6e | Delta+=%.3e | p=[%.6f %.6f] | ', ...
                'rho=%s | attempted=%d | pChanged=%d\n'], ...
                k,Nt,t_now,Ebase,Efinal,Delta_plus, ...
                p(1),p(2),rho_text(growth), ...
                update_attempted_this_step,p_changed_this_step);
        end

        if Delta_l > cfg.adaptation.tau_l
            El = Ebase;
        end

        k = k+1;
        update_attempted_this_step = false;
        p_changed_this_step = false;
    end

    total_time = toc(total_timer)+initial_training_time;

    % =====================================================================
    % Exact errors and final residual points
    % =====================================================================
    if any(~isfinite(pred_snapshots(:)))
        error('One or more exact-time snapshots were not stored.');
    end

    snapshot_rel_l2 = relative_l2( ...
        pred_snapshots(:),ref.UU(:));

    final_rel_l2 = relative_l2( ...
        pred_snapshots(:,:,end),ref.U);

    [final_selected_centers, ...
        final_selected_indices, ...
        final_selected_scores] = ...
        select_growth_centers( ...
            problem.Xi,last_base_residual, ...
            cfg.m2,cfg.growth.center_policy);

    % =====================================================================
    % Save
    % =====================================================================
    if cfg.growth.enabled
        growth_tag = 'LG';
    else
        growth_tag = 'NoLG';
    end

    run_name = sprintf( ...
        '%s_DT_%s_Nt%d', ...
        method,growth_tag,Nt);

    out_dir = fullfile( ...
        project_root,cfg.output_root_name,run_name);

    if exist(out_dir,'dir') ~= 7
        mkdir(out_dir);
    end

    results = struct();

    results.cfg = cfg;
    results.method = method;
    results.growth_tag = growth_tag;

    results.reference = ref;
    results.problem = problem;

    results.snapshot_steps = snapshot_steps;
    results.snapshot_times = ref.tt;
    results.pred_snapshots = pred_snapshots;

    results.p_history = p_history;
    results.rho_history = rho_history;

    results.base_residual_history = base_residual_history;
    results.final_residual_history = final_residual_history;
    results.feature_count_history = feature_count_history;

    results.initial_history = initial_history;
    results.initial_training_time = initial_training_time;

    results.num_update_attempts = num_update_attempts;
    results.num_effective_updates = num_effective_updates;
    results.num_updates = num_effective_updates;
    results.update_log = update_log;
    results.update_histories = update_histories;

    results.num_growth_refreshes = num_growth_refreshes;
    results.growth_refresh_log = growth_refresh_log;
    results.growth_histories = growth_histories;

    results.final_growth = growth;

    % These are the top residual points at exactly t=1, regardless of the
    % time at which the last growth block itself was refreshed.
    results.final_selected_centers = final_selected_centers;
    results.final_selected_indices = final_selected_indices;
    results.final_selected_scores = final_selected_scores;
    results.final_base_residual_vector = last_base_residual;

    results.snapshot_relative_l2 = snapshot_rel_l2;
    results.final_time_relative_l2 = final_rel_l2;

    results.final_p = p;
    results.final_rho = NaN;

    if growth_is_active(growth)
        results.final_rho = growth.rho;
    end

    results.total_time = total_time;
    results.output_dir = out_dir;

    save( ...
        fullfile(out_dir,[run_name '_results.mat']), ...
        'results','-v7.3');

    save_summary_tables(results,out_dir,run_name);

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('FINAL %s REPORT\n',run_name);
    fprintf('============================================================\n');
    fprintf('%d-snapshot rel L2        = %.8e\n', ...
        cfg.num_saved_snapshots,snapshot_rel_l2);
    fprintf('final-time rel L2         = %.8e\n',final_rel_l2);
    fprintf('global update attempts    = %d\n',num_update_attempts);
    fprintf('effective p updates       = %d\n',num_effective_updates);
    fprintf('growth refreshes          = %d\n',num_growth_refreshes);
    fprintf('final p                   = [%.10f %.10f]\n',p(1),p(2));

    if growth_is_active(growth)
        fprintf('final rho                 = %.10f\n',growth.rho);
    end

    fprintf('total method time         = %.3f s\n',total_time);
    fprintf('output directory          = %s\n',out_dir);
    fprintf('============================================================\n');

    % =====================================================================
    % Nested snapshot helper
    % =====================================================================
    function store_snapshot( ...
        step,p_now,coef_base_now,growth_now,coef_growth_now)

        j = find(snapshot_steps == step,1);

        if isempty(j)
            return;
        end

        pred_vec = evaluate_burgers2d_trial( ...
            ref.points, ...
            p_now, ...
            basis, ...
            coef_base_now, ...
            growth_now, ...
            coef_growth_now);

        pred_snapshots(:,:,j) = reshape( ...
            pred_vec,ne,ne);

        fprintf('Saved snapshot %2d/%d at step %4d, t=%.6f\n', ...
            j,ns,step,ref.tt(j));
    end
end


function [coef,residual,E] = solve_base_candidate( ...
    base_op,qhat,rhs_i,rhs_b, ...
    a0,convection_weight,diffusion_weight,eta,cfg)

    A_i = assemble_interior_operator( ...
        base_op.Phi_i, ...
        base_op.Dsum_i, ...
        base_op.Lap_i, ...
        qhat,a0,convection_weight,diffusion_weight);

    B = eta*base_op.Phi_b;
    M = [A_i;B];

    coef = solve_least_squares( ...
        M,[rhs_i;rhs_b],cfg.linear_solver);

    residual = A_i*coef-rhs_i;
    E = sqrt(mean(residual.^2));
end


function [coef_base,coef_growth,u,E] = ...
    solve_accepted_candidate( ...
        base_op,growth,qhat,rhs_i,rhs_b, ...
        a0,convection_weight,diffusion_weight,eta,cfg)

    if growth_is_active(growth)

        Phi = [base_op.Phi_i,growth.Phi_i];
        Dsum = [base_op.Dsum_i,growth.Dsum_i];
        Lap = [base_op.Lap_i,growth.Lap_i];
        Phi_b = [base_op.Phi_b,growth.Phi_b];

        A_i = assemble_interior_operator( ...
            Phi,Dsum,Lap,qhat, ...
            a0,convection_weight,diffusion_weight);

        M = [A_i;eta*Phi_b];

        coef = solve_least_squares( ...
            M,[rhs_i;rhs_b],cfg.linear_solver);

        coef_base = coef(1:cfg.m1);
        coef_growth = coef(cfg.m1+1:end);

        u = ...
            base_op.Phi_i*coef_base + ...
            growth.Phi_i*coef_growth;

    else

        A_i = assemble_interior_operator( ...
            base_op.Phi_i, ...
            base_op.Dsum_i, ...
            base_op.Lap_i, ...
            qhat,a0,convection_weight,diffusion_weight);

        M = [A_i;eta*base_op.Phi_b];

        coef_base = solve_least_squares( ...
            M,[rhs_i;rhs_b],cfg.linear_solver);

        coef_growth = zeros(0,1);
        u = base_op.Phi_i*coef_base;
    end

    residual = A_i*[coef_base;coef_growth]-rhs_i;
    E = sqrt(mean(residual.^2));
end


function A_i = assemble_interior_operator( ...
    Phi,Dsum,Lap,qhat,a0,convection_weight,diffusion_weight)

    qhat = qhat(:);

    A_i = ...
        a0*Phi + ...
        convection_weight*(qhat.*Dsum) - ...
        diffusion_weight*Lap;
end


function tf = growth_is_active(growth)

    tf = ...
        isstruct(growth) && ...
        isfield(growth,'active') && ...
        logical(growth.active);
end


function tf = should_refresh_growth(cfg,growth,p_changed_this_step)

    if ~cfg.growth.enabled
        tf = false;
        return;
    end

    if ~growth_is_active(growth)
        tf = logical(p_changed_this_step);
        return;
    end

    switch lower(strtrim(char(cfg.growth.refresh_policy)))

        case 'on_global_update'
            tf = logical(p_changed_this_step);

        case 'every_step'
            tf = true;

        case 'never_after_first'
            tf = false;

        otherwise
            error('Unknown growth refresh policy.');
    end
end


function validate_configuration(cfg)

    policies = { ...
        'on_global_update', ...
        'every_step', ...
        'never_after_first'};

    if ~ismember( ...
            lower(strtrim(char(cfg.growth.refresh_policy))), ...
            policies)

        error('Unknown cfg.growth.refresh_policy.');
    end

    if cfg.m1 < 1 || cfg.m1 ~= floor(cfg.m1)
        error('cfg.m1 must be a positive integer.');
    end

    if cfg.m2 < 1 || cfg.m2 ~= floor(cfg.m2)
        error('cfg.m2 must be a positive integer.');
    end

    if mod(cfg.num_time_steps,cfg.num_saved_snapshots) ~= 0
        error('Nt must be divisible by num_saved_snapshots.');
    end
end


function txt = rho_text(growth)

    if growth_is_active(growth)
        txt = sprintf('%.6f',growth.rho);
    else
        txt = '--';
    end
end


function log = empty_update_log()

    log = struct( ...
        'attempt_id',{}, ...
        'time_step',{}, ...
        'time',{}, ...
        'method',{}, ...
        'p_before',{}, ...
        'p_candidate',{}, ...
        'p_after',{}, ...
        'trigger_Ek',{}, ...
        'trigger_El',{}, ...
        'delta_plus',{}, ...
        'selected_checkpoint',{}, ...
        'best_selection_value',{}, ...
        'relative_p_change',{}, ...
        'effective_update',{}, ...
        'training_time',{});
end


function record = make_update_record( ...
    id,step,time,method,p_before,p_candidate,p_after, ...
    Ek,El,delta_plus,history,training_time, ...
    relative_p_change,effective_update)

    record = struct();

    record.attempt_id = id;
    record.time_step = step;
    record.time = time;
    record.method = method;

    record.p_before = p_before(:);
    record.p_candidate = p_candidate(:);
    record.p_after = p_after(:);

    record.trigger_Ek = Ek;
    record.trigger_El = El;
    record.delta_plus = delta_plus;

    record.selected_checkpoint = history.best_iteration;
    record.best_selection_value = history.best_selection_value;

    record.relative_p_change = relative_p_change;
    record.effective_update = logical(effective_update);
    record.training_time = training_time;
end


function log = empty_growth_log()

    log = struct( ...
        'refresh_id',{}, ...
        'time_step',{}, ...
        'time',{}, ...
        'rho',{}, ...
        'largest_center_score',{}, ...
        'selected_checkpoint',{}, ...
        'best_selection_value',{}, ...
        'stop_reason',{}, ...
        'training_time',{}, ...
        'total_build_time',{});
end


function record = make_growth_record(id,step,time,growth)

    record = struct();

    record.refresh_id = id;
    record.time_step = step;
    record.time = time;

    record.rho = growth.rho;
    record.largest_center_score = growth.center_scores(1);

    record.selected_checkpoint = ...
        growth.rho_history.best_iteration;

    record.best_selection_value = ...
        growth.rho_history.best_selection_value;

    if isfield(growth.rho_history,'stop_reason')
        record.stop_reason = growth.rho_history.stop_reason;
    else
        record.stop_reason = '';
    end

    record.training_time = growth.training_time;
    record.total_build_time = growth.total_build_time;
end


function save_summary_tables(results,out_dir,run_name)

    cfg = results.cfg;
    Nt = cfg.num_time_steps;
    dt = diff(cfg.t_domain)/Nt;

    Method = {results.method};
    Growth = {results.growth_tag};

    Nt_col = Nt;
    Dt = dt;

    M1 = cfg.m1;
    M2 = cfg.m2;

    NumUpdateAttempts = results.num_update_attempts;
    NumEffectiveUpdates = results.num_effective_updates;
    NumGrowthRefreshes = results.num_growth_refreshes;

    FinalPx = results.final_p(1);
    FinalPy = results.final_p(2);
    FinalRho = results.final_rho;

    SnapshotRelL2 = results.snapshot_relative_l2;
    FinalTimeRelL2 = results.final_time_relative_l2;
    TotalTimeSec = results.total_time;

    summary = table( ...
        Method,Growth,Nt_col,Dt,M1,M2, ...
        NumUpdateAttempts,NumEffectiveUpdates,NumGrowthRefreshes, ...
        FinalPx,FinalPy,FinalRho, ...
        SnapshotRelL2,FinalTimeRelL2,TotalTimeSec, ...
        'VariableNames',{ ...
            'Method','Growth','Nt','dt','m1','m2', ...
            'NumUpdateAttempts','NumEffectiveUpdates', ...
            'NumGrowthRefreshes','FinalPx','FinalPy','FinalRho', ...
            'SnapshotRelL2','FinalTimeRelL2','TotalTimeSec'});

    writetable( ...
        summary, ...
        fullfile(out_dir,[run_name '_summary.csv']));

    TimeStep = (0:Nt).';
    Time = cfg.t_domain(1)+TimeStep*dt;

    Px = results.p_history(:,1);
    Py = results.p_history(:,2);
    Rho = results.rho_history;

    BaseResidualRMSE = results.base_residual_history;
    FinalResidualRMSE = results.final_residual_history;
    NumFeatures = results.feature_count_history;

    history_table = table( ...
        TimeStep,Time,Px,Py,Rho, ...
        BaseResidualRMSE,FinalResidualRMSE,NumFeatures);

    writetable( ...
        history_table, ...
        fullfile(out_dir,[run_name '_parameter_history.csv']));

    if ~isempty(results.update_log)

        n = numel(results.update_log);

        AttemptID = (1:n).';
        TimeStep = reshape([results.update_log.time_step],[],1);
        Time = reshape([results.update_log.time],[],1);

        PBeforeX = arrayfun( ...
            @(s) s.p_before(1),results.update_log).';
        PBeforeY = arrayfun( ...
            @(s) s.p_before(2),results.update_log).';

        PCandidateX = arrayfun( ...
            @(s) s.p_candidate(1),results.update_log).';
        PCandidateY = arrayfun( ...
            @(s) s.p_candidate(2),results.update_log).';

        PAfterX = arrayfun( ...
            @(s) s.p_after(1),results.update_log).';
        PAfterY = arrayfun( ...
            @(s) s.p_after(2),results.update_log).';

        TriggerEk = reshape([results.update_log.trigger_Ek],[],1);
        TriggerEl = reshape([results.update_log.trigger_El],[],1);
        DeltaPlus = reshape([results.update_log.delta_plus],[],1);

        SelectedCheckpoint = reshape( ...
            [results.update_log.selected_checkpoint],[],1);

        BestSelectionValue = reshape( ...
            [results.update_log.best_selection_value],[],1);

        RelativePChange = reshape( ...
            [results.update_log.relative_p_change],[],1);

        EffectiveUpdate = reshape( ...
            [results.update_log.effective_update],[],1);

        TrainingTimeSec = reshape( ...
            [results.update_log.training_time],[],1);

        update_table = table( ...
            AttemptID,TimeStep,Time, ...
            PBeforeX,PBeforeY,PCandidateX,PCandidateY, ...
            PAfterX,PAfterY,TriggerEk,TriggerEl,DeltaPlus, ...
            SelectedCheckpoint,BestSelectionValue, ...
            RelativePChange,EffectiveUpdate,TrainingTimeSec);

        writetable( ...
            update_table, ...
            fullfile(out_dir,[run_name '_global_updates.csv']));
    end

    if ~isempty(results.growth_refresh_log)

        n = numel(results.growth_refresh_log);

        RefreshID = (1:n).';
        TimeStep = reshape( ...
            [results.growth_refresh_log.time_step],[],1);
        Time = reshape( ...
            [results.growth_refresh_log.time],[],1);
        Rho = reshape( ...
            [results.growth_refresh_log.rho],[],1);
        LargestCenterScore = reshape( ...
            [results.growth_refresh_log.largest_center_score],[],1);
        SelectedCheckpoint = reshape( ...
            [results.growth_refresh_log.selected_checkpoint],[],1);
        BestSelectionValue = reshape( ...
            [results.growth_refresh_log.best_selection_value],[],1);
        StopReason = {results.growth_refresh_log.stop_reason}.';
        TrainingTimeSec = reshape( ...
            [results.growth_refresh_log.training_time],[],1);
        TotalBuildTimeSec = reshape( ...
            [results.growth_refresh_log.total_build_time],[],1);

        growth_table = table( ...
            RefreshID,TimeStep,Time,Rho,LargestCenterScore, ...
            SelectedCheckpoint,BestSelectionValue,StopReason, ...
            TrainingTimeSec,TotalBuildTimeSec);

        writetable( ...
            growth_table, ...
            fullfile(out_dir,[run_name '_growth_refreshes.csv']));
    end
end
