function results = ac_2d_dt_study(cfg,project_root)
%AC_2D_DT_STUDY
% Paper-aligned 2-D Allen--Cahn PDAD-DT / DDAD-DT solver with an optional
% DDAD-trained local layer-growth block.

    method = upper(strtrim(char(cfg.method)));

    if ~ismember(method,{'DDAD','PDAD'})
        error('cfg.method must be DDAD or PDAD.');
    end

    validate_growth_policy(cfg);

    ref = load_ac2d_reference(project_root);
    problem = build_ac2d_problem(cfg);

    t0 = cfg.t_domain(1);
    tf = cfg.t_domain(2);

    Nt = cfg.num_time_steps;
    dt = (tf-t0)/Nt;

    if mod(Nt,cfg.num_saved_snapshots) ~= 0
        error('Nt=%d must be divisible by 10.',Nt);
    end

    expected_times = ...
        t0+(1:cfg.num_saved_snapshots).' ...
        *(tf-t0)/cfg.num_saved_snapshots;

    if max(abs(ref.tt-expected_times)) > 1e-12
        error('Reference times do not match the requested ten snapshots.');
    end

    snapshot_steps_real = (ref.tt-t0)/dt;
    snapshot_steps = round(snapshot_steps_real);

    if max(abs(snapshot_steps_real-snapshot_steps)) > 1e-10
        error('Reference times do not align with Nt=%d.',Nt);
    end

    % =====================================================================
    % One frozen first-layer random basis
    % =====================================================================
    basis = ...
        build_random_weights_nd( ...
            cfg.m1,cfg.domain,cfg.seed);

    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('%s-DT 2-D ALLEN--CAHN\n',method);
    fprintf('============================================================\n');
    fprintf('Nt / dt                    = %d / %.6e\n',Nt,dt);
    fprintf('interior points            = %d x %d = %d\n', ...
        cfg.num_collocation_x,cfg.num_collocation_y, ...
        problem.num_interior);
    fprintf('boundary points            = %d\n',problem.num_boundary);
    fprintf('m1 / m2                    = %d / %d\n',cfg.m1,cfg.m2);
    fprintf('layer growth enabled       = %d\n',cfg.growth.enabled);
    fprintf('growth refresh policy      = %s\n', ...
        cfg.growth.refresh_policy);
    fprintf('same DDAD/solve base basis = 1\n');
    fprintf('seed                       = %d\n',cfg.seed);
    fprintf('============================================================\n\n');

    % =====================================================================
    % Initial distribution
    % =====================================================================
    initial_history = [];
    initial_training_time = 0;

    switch lower(strtrim(char(cfg.initialization.mode)))

        case 'ddad'

            cache0 = ...
                prepare_data_cache( ...
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

    % Method time includes initial DDAD, all p/rho optimization, and all
    % physical time-step solves.  Reference loading/problem generation and
    % plotting are excluded.
    total_timer = tic;

    % =====================================================================
    % Storage
    % =====================================================================
    pred_snapshots = nan(50,50,cfg.num_saved_snapshots);

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

    num_updates = 0;
    num_growth_refreshes = 0;

    growth = [];

    % =====================================================================
    % First-layer matrices at the selected initial p
    % =====================================================================
    base_op = ...
        prepare_ac2d_base_operator( ...
            p,problem,basis);

    eta = cfg.boundary_penalty;
    rhs_b = eta*problem.boundary_values;

    % =====================================================================
    % Step 1: first-order IMEX startup
    % =====================================================================
    a0 = 1;
    diffusion_weight = dt*cfg.epsilon_ac;

    rhs_i = ...
        problem.u0 ...
        -dt*cfg.alpha*(problem.u0.^3-problem.u0);

    [base_coef,base_residual,Ebase] = ...
        solve_base_candidate( ...
            base_op,rhs_i,rhs_b,a0,diffusion_weight,eta,cfg);

    if cfg.growth.enabled && cfg.growth.build_at_first_step

        growth = ...
            ac2d_growth_stage( ...
                p,basis,problem,base_coef,base_residual,[],cfg);

        num_growth_refreshes = 1;

        growth_refresh_log(1) = ...
            make_growth_record( ...
                num_growth_refreshes,1,t0+dt,growth);

        growth_histories{1,1} = growth.rho_history;
    end

    [coef_base_current,coef_growth_current, ...
        u_new,Efinal] = ...
        solve_accepted_candidate( ...
            base_op,growth,rhs_i,rhs_b, ...
            a0,diffusion_weight,eta,cfg);

    u_ll = problem.u0;
    u_l = u_new;

    El = Ebase;

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

    fprintf(['step=%4d/%4d | t=%.6f | baseE=%.6e | ', ...
             'finalE=%.6e | p=[%.6f %.6f] | rho=%s\n'], ...
        1,Nt,t0+dt,Ebase,Efinal,p(1),p(2),rho_text(growth));

    % =====================================================================
    % BDF2 time stepping with Algorithm 3
    % =====================================================================
    base_factor = [];
    augmented_factor = [];

    k = 2;
    updated_this_step = false;

    while k <= Nt

        f_l = u_l.^3-u_l;
        f_ll = u_ll.^3-u_ll;

        a0 = 3;
        diffusion_weight = 2*dt*cfg.epsilon_ac;

        rhs_i = ...
            4*u_l-u_ll ...
            -2*dt*cfg.alpha*(2*f_l-f_ll);

        % -------------------------------------------------------------
        % Base m1 candidate used by Algorithm 3.
        % -------------------------------------------------------------
        A_base_i = ...
            a0*base_op.Phi_i ...
            -diffusion_weight*base_op.Lap_i;

        if isempty(base_factor)

            B_base = eta*base_op.Phi_b;

            if size(A_base_i,2) ~= size(B_base,2)

                error([ ...
                    'Base operator column mismatch: interior has %d ', ...
                    'columns, boundary has %d columns.'], ...
                    size(A_base_i,2),size(B_base,2));
            end

            % Compute the interior arithmetic expression before vertical
            % concatenation.  Inside MATLAB square brackets, an expression
            % such as [A -B; C] may be parsed as two horizontal blocks.
            Mbase = [A_base_i;B_base];

            base_factor = ...
                ac2d_factorize_ls(Mbase,cfg.linear_solver);
        end

        base_coef = ...
            ac2d_solve_factored( ...
                base_factor,[rhs_i;rhs_b]);

        base_residual = ...
            A_base_i*base_coef-rhs_i;

        Ebase = sqrt(mean(base_residual.^2));

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
            updated_this_step;

        if ~accept_global

            p_before = p;

            if strcmp(method,'DDAD')

                cache = ...
                    prepare_data_cache( ...
                        problem.Xi,u_l,basis);

                objective_fun = @(pp) ...
                    evaluate_data_reduced_fast( ...
                        pp,cache,cfg.ddad.lambda, ...
                        ls_opts,cfg.activation);

                opt_cfg = cfg.ddad.optimizer;

            else

                objective_fun = @(pp) ...
                    evaluate_ac2d_pdad_reduced_fast( ...
                        pp, ...
                        problem.Xi, ...
                        problem.Xb, ...
                        basis, ...
                        rhs_i, ...
                        a0, ...
                        diffusion_weight, ...
                        eta, ...
                        cfg.pdad.lambda, ...
                        ls_opts);

                opt_cfg = cfg.pdad.optimizer;
            end

            t_update = tic;

            [p_new,ad_history] = ...
                optimize_distribution_adam( ...
                    p,objective_fun,opt_cfg);

            update_time = toc(t_update);

            p = p_new(:);

            num_updates = num_updates+1;

            update_log(num_updates) = ...
                make_update_record( ...
                    num_updates,k,t0+k*dt,method, ...
                    p_before,p,Ebase,El,Delta_plus, ...
                    ad_history,update_time);

            update_histories{num_updates,1} = ad_history;

            fprintf('\n');
            fprintf('>>> %s global update %d at step %d\n', ...
                method,num_updates,k);
            fprintf('    p: [%.8f %.8f] -> [%.8f %.8f]\n', ...
                p_before(1),p_before(2),p(1),p(2));
            fprintf('    selected checkpoint = %d\n', ...
                ad_history.best_iteration);
            fprintf('    best selection MSE   = %.6e\n', ...
                ad_history.best_selection_value);
            fprintf('    update time          = %.3f s\n\n', ...
                update_time);

            base_op = ...
                prepare_ac2d_base_operator( ...
                    p,problem,basis);

            base_factor = [];
            augmented_factor = [];

            updated_this_step = true;

            if Delta_l > cfg.adaptation.tau_l
                El = Ebase;
            end

            % Re-solve the same physical time step once with the new p.
            continue;
        end

        % -------------------------------------------------------------
        % Refresh the local block according to the selected policy.
        % -------------------------------------------------------------
        do_refresh = ...
            should_refresh_growth( ...
                cfg,growth,updated_this_step);

        if do_refresh

            previous_growth = growth;

            growth = ...
                ac2d_growth_stage( ...
                    p,basis,problem,base_coef, ...
                    base_residual,previous_growth,cfg);

            num_growth_refreshes = num_growth_refreshes+1;

            growth_refresh_log(num_growth_refreshes) = ...
                make_growth_record( ...
                    num_growth_refreshes,k,t0+k*dt,growth);

            growth_histories{num_growth_refreshes,1} = ...
                growth.rho_history;

            augmented_factor = [];
        end

        % -------------------------------------------------------------
        % Accepted current solution: base only or base+growth.
        % -------------------------------------------------------------
        if growth_is_active(growth)

            Phi_aug_i = ...
                [base_op.Phi_i,growth.Phi_i];

            Lap_aug_i = ...
                [base_op.Lap_i,growth.Lap_i];

            A_aug_i = ...
                a0*Phi_aug_i ...
                -diffusion_weight*Lap_aug_i;

            if isempty(augmented_factor)

                B_aug = ...
                    eta*[base_op.Phi_b,growth.Phi_b];

                if size(A_aug_i,2) ~= size(B_aug,2)

                    error([ ...
                        'Augmented operator column mismatch: interior ', ...
                        'has %d columns, boundary has %d columns.'], ...
                        size(A_aug_i,2),size(B_aug,2));
                end

                Maug = [A_aug_i;B_aug];

                augmented_factor = ...
                    ac2d_factorize_ls( ...
                        Maug,cfg.linear_solver);
            end

            coef_aug = ...
                ac2d_solve_factored( ...
                    augmented_factor,[rhs_i;rhs_b]);

            coef_base_current = coef_aug(1:cfg.m1);
            coef_growth_current = coef_aug(cfg.m1+1:end);

            final_residual = ...
                A_aug_i*coef_aug-rhs_i;

            Efinal = sqrt(mean(final_residual.^2));

            u_new = ...
                base_op.Phi_i*coef_base_current + ...
                growth.Phi_i*coef_growth_current;

        else

            coef_base_current = base_coef;
            coef_growth_current = zeros(0,1);

            Efinal = Ebase;
            u_new = base_op.Phi_i*base_coef;
        end

        u_ll = u_l;
        u_l = u_new;

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
                'rho=%s | globalUpdated=%d\n'], ...
                k,Nt,t0+k*dt,Ebase,Efinal,Delta_plus, ...
                p(1),p(2),rho_text(growth),updated_this_step);
        end

        if Delta_l > cfg.adaptation.tau_l
            El = Ebase;
        end

        k = k+1;
        updated_this_step = false;
    end

    total_time = toc(total_timer)+initial_training_time;

    % =====================================================================
    % Validate and compute errors
    % =====================================================================
    if any(~isfinite(pred_snapshots(:)))
        error('One or more reference-time snapshots were not stored.');
    end

    snapshot_rel_l2 = ...
        relative_l2( ...
            pred_snapshots(:),ref.UU(:));

    final_rel_l2 = ...
        relative_l2( ...
            pred_snapshots(:,:,end),ref.U);

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

    results.num_updates = num_updates;
    results.update_log = update_log;
    results.update_histories = update_histories;

    results.num_growth_refreshes = num_growth_refreshes;
    results.growth_refresh_log = growth_refresh_log;
    results.growth_histories = growth_histories;

    results.final_growth = growth;

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
    fprintf('10-snapshot rel L2       = %.8e\n',snapshot_rel_l2);
    fprintf('final-time rel L2        = %.8e\n',final_rel_l2);
    fprintf('global p updates         = %d\n',num_updates);
    fprintf('growth refreshes         = %d\n',num_growth_refreshes);
    fprintf('final p                  = [%.10f %.10f]\n',p(1),p(2));

    if growth_is_active(growth)
        fprintf('final rho                = %.10f\n',growth.rho);
    end

    fprintf('total method time        = %.3f s\n',total_time);
    fprintf('output directory         = %s\n',out_dir);
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

        pred_vec = ...
            evaluate_ac2d_trial( ...
                ref.points, ...
                p_now, ...
                basis, ...
                coef_base_now, ...
                growth_now, ...
                coef_growth_now);

        pred_snapshots(:,:,j) = ...
            reshape(pred_vec,size(ref.U));

        fprintf('Saved snapshot %2d/10 at step %4d, t=%.6f\n', ...
            j,step,ref.tt(j));
    end
end


function [coef,residual,E] = ...
    solve_base_candidate( ...
        base_op,rhs_i,rhs_b,a0,diffusion_weight,eta,cfg)

    A_i = ...
        a0*base_op.Phi_i ...
        -diffusion_weight*base_op.Lap_i;

    M = [A_i;eta*base_op.Phi_b];

    factor = ac2d_factorize_ls(M,cfg.linear_solver);

    coef = ac2d_solve_factored(factor,[rhs_i;rhs_b]);

    residual = A_i*coef-rhs_i;
    E = sqrt(mean(residual.^2));
end


function [coef_base,coef_growth,u,E] = ...
    solve_accepted_candidate( ...
        base_op,growth,rhs_i,rhs_b, ...
        a0,diffusion_weight,eta,cfg)

    if growth_is_active(growth)

        Phi = [base_op.Phi_i,growth.Phi_i];
        Lap = [base_op.Lap_i,growth.Lap_i];
        Phi_b = [base_op.Phi_b,growth.Phi_b];

        A_i = a0*Phi-diffusion_weight*Lap;

        factor = ...
            ac2d_factorize_ls( ...
                [A_i;eta*Phi_b],cfg.linear_solver);

        coef = ...
            ac2d_solve_factored( ...
                factor,[rhs_i;rhs_b]);

        coef_base = coef(1:cfg.m1);
        coef_growth = coef(cfg.m1+1:end);

        u = ...
            base_op.Phi_i*coef_base + ...
            growth.Phi_i*coef_growth;

    else

        A_i = ...
            a0*base_op.Phi_i ...
            -diffusion_weight*base_op.Lap_i;

        factor = ...
            ac2d_factorize_ls( ...
                [A_i;eta*base_op.Phi_b], ...
                cfg.linear_solver);

        coef_base = ...
            ac2d_solve_factored( ...
                factor,[rhs_i;rhs_b]);

        coef_growth = zeros(0,1);
        u = base_op.Phi_i*coef_base;
    end

    residual = ...
        A_i*[coef_base;coef_growth]-rhs_i;

    E = sqrt(mean(residual.^2));
end


function tf = growth_is_active(growth)

    tf = ...
        isstruct(growth) && ...
        isfield(growth,'active') && ...
        logical(growth.active);
end


function tf = should_refresh_growth(cfg,growth,updated_this_step)

    if ~cfg.growth.enabled
        tf = false;
        return;
    end

    if ~growth_is_active(growth)
        tf = cfg.growth.build_at_first_step;
        return;
    end

    switch lower(strtrim(char(cfg.growth.refresh_policy)))

        case 'on_global_update'
            tf = logical(updated_this_step);

        case 'every_step'
            tf = true;

        case 'never_after_first'
            tf = false;

        otherwise
            error('Unknown growth refresh policy.');
    end
end


function validate_growth_policy(cfg)

    policies = { ...
        'on_global_update', ...
        'every_step', ...
        'never_after_first'};

    if ~ismember( ...
            lower(strtrim(char(cfg.growth.refresh_policy))), ...
            policies)

        error('Unknown cfg.growth.refresh_policy.');
    end

    if cfg.m2 < 1 || cfg.m2 ~= floor(cfg.m2)
        error('cfg.m2 must be a positive integer.');
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
        'update_id',{}, ...
        'time_step',{}, ...
        'time',{}, ...
        'method',{}, ...
        'p_before',{}, ...
        'p_after',{}, ...
        'trigger_Ek',{}, ...
        'trigger_El',{}, ...
        'delta_plus',{}, ...
        'selected_checkpoint',{}, ...
        'best_selection_value',{}, ...
        'training_time',{});
end


function record = make_update_record( ...
    id,step,time,method,p_before,p_after, ...
    Ek,El,delta_plus,history,training_time)

    record = struct();

    record.update_id = id;
    record.time_step = step;
    record.time = time;
    record.method = method;

    record.p_before = p_before(:);
    record.p_after = p_after(:);

    record.trigger_Ek = Ek;
    record.trigger_El = El;
    record.delta_plus = delta_plus;

    record.selected_checkpoint = history.best_iteration;
    record.best_selection_value = history.best_selection_value;
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

    NumGlobalUpdates = results.num_updates;
    NumGrowthRefreshes = results.num_growth_refreshes;

    FinalPx = results.final_p(1);
    FinalPy = results.final_p(2);
    FinalRho = results.final_rho;

    SnapshotRelL2 = results.snapshot_relative_l2;
    FinalTimeRelL2 = results.final_time_relative_l2;
    TotalTimeSec = results.total_time;

    summary = table( ...
        Method,Growth,Nt_col,Dt,M1,M2, ...
        NumGlobalUpdates,NumGrowthRefreshes, ...
        FinalPx,FinalPy,FinalRho, ...
        SnapshotRelL2,FinalTimeRelL2,TotalTimeSec, ...
        'VariableNames',{ ...
            'Method','Growth','Nt','dt','m1','m2', ...
            'NumGlobalUpdates','NumGrowthRefreshes', ...
            'FinalPx','FinalPy','FinalRho', ...
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

        UpdateID = (1:n).';
        TimeStep = reshape([results.update_log.time_step],[],1);
        Time = reshape([results.update_log.time],[],1);

        PBeforeX = arrayfun( ...
            @(s) s.p_before(1),results.update_log).';

        PBeforeY = arrayfun( ...
            @(s) s.p_before(2),results.update_log).';

        PAfterX = arrayfun( ...
            @(s) s.p_after(1),results.update_log).';

        PAfterY = arrayfun( ...
            @(s) s.p_after(2),results.update_log).';

        TriggerEk = reshape([results.update_log.trigger_Ek],[],1);
        TriggerEl = reshape([results.update_log.trigger_El],[],1);
        DeltaPlus = reshape([results.update_log.delta_plus],[],1);

        SelectedCheckpoint = ...
            reshape([results.update_log.selected_checkpoint],[],1);

        BestSelectionValue = ...
            reshape([results.update_log.best_selection_value],[],1);

        TrainingTimeSec = ...
            reshape([results.update_log.training_time],[],1);

        update_table = table( ...
            UpdateID,TimeStep,Time, ...
            PBeforeX,PBeforeY,PAfterX,PAfterY, ...
            TriggerEk,TriggerEl,DeltaPlus, ...
            SelectedCheckpoint,BestSelectionValue,TrainingTimeSec);

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

        TrainingTimeSec = reshape( ...
            [results.growth_refresh_log.training_time],[],1);

        TotalBuildTimeSec = reshape( ...
            [results.growth_refresh_log.total_build_time],[],1);

        growth_table = table( ...
            RefreshID,TimeStep,Time,Rho,LargestCenterScore, ...
            SelectedCheckpoint,BestSelectionValue, ...
            TrainingTimeSec,TotalBuildTimeSec);

        writetable( ...
            growth_table, ...
            fullfile(out_dir,[run_name '_growth_refreshes.csv']));
    end
end
