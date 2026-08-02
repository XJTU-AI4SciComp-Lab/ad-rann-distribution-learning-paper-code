function results = ac_1d_dt_study(cfg,project_root)
%AC_1D_DT_STUDY Allen--Cahn PDAD-DT / DDAD-DT driver.
%
% The code saves exactly 25 numerical snapshots at
%
%   t_j = j/25,  j=1,...,25.
%
% The initial state t=0 is not included.  Therefore Nt must be divisible
% by 25.  No temporal rounding or interpolation is used.
%
% DDAD training and physical PDE solving use the same frozen random basis.

    method = upper(strtrim(char(cfg.method)));

    if ~ismember(method,{'DDAD','PDAD'})
        error('cfg.method must be DDAD or PDAD.');
    end

    ref = load_ac_reference(project_root);

    xa = cfg.x_domain(1);
    xb = cfg.x_domain(2);

    t0 = cfg.t_domain(1);
    tf = cfg.t_domain(2);

    Nt = cfg.num_time_steps;
    dt = (tf-t0)/Nt;

    nsnap = cfg.num_saved_snapshots;

    if nsnap ~= 25
        error('This example requires cfg.num_saved_snapshots=25.');
    end

    if mod(Nt,nsnap) ~= 0

        error([ ...
            'Nt=%d is not divisible by 25. ', ...
            'Use Nt=125,250,500,1000,2000, or another multiple of 25.'], ...
            Nt);
    end

    N = cfg.num_collocation_points;

    Xi = (xa+1e-6 : (xb-xa)/N : xb-1e-6).';

    if numel(Xi) ~= N
        error('Generated %d collocation points; expected %d.', ...
            numel(Xi),N);
    end

    Xb = [xa;xb];
    u0 = ac_initial_condition(Xi);

    % =====================================================================
    % Same frozen basis for DDAD training and PDE solving
    % =====================================================================
    basis = build_random_weights_nd( ...
        cfg.num_features, ...
        cfg.x_domain(:).', ...
        cfg.seed);

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('Random-feature realization\n');
    fprintf('============================================================\n');
    fprintf('seed                       = %d\n',cfg.seed);
    fprintf('features                   = %d\n',cfg.num_features);
    fprintf('DDAD train basis == solve  = 1\n');
    fprintf('============================================================\n\n');

    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    % =====================================================================
    % Initial distribution parameter
    % =====================================================================
    initial_history = [];
    initial_training_time = 0;

    init_mode = lower(strtrim(char(cfg.initialization.mode)));

    switch init_mode

        case 'fixed'

            p = cfg.initialization.fixed_p;

        case 'ddad'

            p0 = cfg.initialization.ddad_p0;

            cache0 = prepare_data_cache(Xi,u0,basis);

            objective0 = @(pp) ...
                evaluate_data_reduced_fast( ...
                    pp,cache0,cfg.initialization.lambda, ...
                    ls_opts,cfg.activation);

            t_init = tic;

            [p,initial_history] = ...
                optimize_distribution_adam( ...
                    p0,objective0,cfg.initialization.optimizer);

            initial_training_time = toc(t_init);

        otherwise

            error('cfg.initialization.mode must be fixed or ddad.');
    end

    p = p(:);

    % =====================================================================
    % Exact 25-snapshot schedule
    % =====================================================================
    snapshot_stride = Nt/nsnap;
    snapshot_steps = (snapshot_stride:snapshot_stride:Nt).';
    snapshot_times = t0 + snapshot_steps*dt;

    requested_times = ...
        t0 + (1:nsnap).'*(tf-t0)/nsnap;

    if max(abs(snapshot_times-requested_times)) > ...
            100*eps(max(1,tf))

        error('Snapshot schedule is not exactly aligned.');
    end

    pred_snapshots = nan(numel(ref.xx),nsnap);

    % =====================================================================
    % Histories
    % =====================================================================
    p_history = nan(Nt+1,1);
    E_history = nan(Nt+1,1);

    p_history(1) = p;

    update_log = struct( ...
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

    update_histories = {};
    num_updates = 0;

    st_i = evaluate_ac_state(Xi,p,basis);

    B = build_ac_periodic_boundary( ...
        Xb,p,basis,cfg.boundary_penalty);

    rhs_b = zeros(2,1);

    % =====================================================================
    % First-order IMEX startup
    % =====================================================================
    total_timer = tic;

    A = ...
        st_i.phi ...
        -dt*cfg.epsilon_ac*st_i.uxx;

    rhs_i = ...
        u0 ...
        -dt*cfg.alpha*(u0.^3-u0);

    coef = solve_least_squares( ...
        [A;B], ...
        [rhs_i;rhs_b], ...
        cfg.linear_solver);

    coef = coef(:);

    E1 = sqrt(mean((A*coef-rhs_i).^2));

    u_ll = u0;
    u_l = st_i.phi*coef;

    El = E1;

    E_history(2) = E1;
    p_history(2) = p;

    store_snapshot(1,coef,p);

    fprintf('============================================================\n');
    fprintf('%s-DT Allen--Cahn\n',method);
    fprintf('============================================================\n');
    fprintf('Nt / dt               = %d / %.6e\n',Nt,dt);
    fprintf('saved snapshots       = %d\n',nsnap);
    fprintf('snapshot stride       = %d steps\n',snapshot_stride);
    fprintf('snapshot times        = %.4f, %.4f, ..., %.4f\n', ...
        snapshot_times(1),snapshot_times(2),snapshot_times(end));
    fprintf('collocation / m       = %d / %d\n',N,cfg.num_features);
    fprintf('epsilon               = %.6e\n',cfg.epsilon_ac);
    fprintf('alpha                 = %.6f\n',cfg.alpha);
    fprintf('first-step E1         = %.6e\n',E1);
    fprintf('initial p             = %.8f\n',p);
    fprintf('============================================================\n');

    % =====================================================================
    % BDF2 time loop
    % =====================================================================
    k = 2;
    updated_this_step = false;

    while k <= Nt

        f_l = u_l.^3-u_l;
        f_ll = u_ll.^3-u_ll;

        A = ...
            3*st_i.phi ...
            -2*dt*cfg.epsilon_ac*st_i.uxx;

        rhs_i = ...
            4*u_l-u_ll ...
            -2*dt*cfg.alpha*(2*f_l-f_ll);

        coef = solve_least_squares( ...
            [A;B], ...
            [rhs_i;rhs_b], ...
            cfg.linear_solver);

        coef = coef(:);

        Ek = sqrt(mean((A*coef-rhs_i).^2));

        Delta_plus = ...
            max(Ek-El,0) / ...
            max(abs(El),cfg.adaptation.residual_epsilon);

        Delta_l = ...
            abs(Ek-El) / ...
            max( ...
                min(abs(El),abs(Ek)), ...
                cfg.adaptation.residual_epsilon);

        do_accept = ...
            Delta_plus < cfg.adaptation.tau_k || ...
            updated_this_step;

        if cfg.verbose && ...
           (cfg.print_every <= 1 || mod(k,cfg.print_every) == 0)

            fprintf([ ...
                'step=%4d/%4d | t=%.6f | Ek=%.6e | El=%.6e | ', ...
                'Delta+=%.3e | Delta_l=%.3e | p=%.6f | updated=%d\n'], ...
                k,Nt,t0+k*dt,Ek,El, ...
                Delta_plus,Delta_l,p,updated_this_step);
        end

        if do_accept

            u_new = st_i.phi*coef;

            u_ll = u_l;
            u_l = u_new;

            E_history(k+1) = Ek;
            p_history(k+1) = p;

            store_snapshot(k,coef,p);

            k = k+1;
            updated_this_step = false;

        else

            p_before = p;

            if strcmp(method,'DDAD')

                cache = prepare_data_cache( ...
                    Xi,u_l,basis);

                objective_fun = @(pp) ...
                    evaluate_data_reduced_fast( ...
                        pp,cache,cfg.ddad.lambda, ...
                        ls_opts,cfg.activation);

                opt_cfg = cfg.ddad.optimizer;

            else

                objective_fun = @(pp) ...
                    evaluate_ac_pdad_reduced_fast( ...
                        pp,Xi,Xb,basis,u_l,u_ll, ...
                        dt,cfg.epsilon_ac,cfg.alpha, ...
                        cfg.boundary_penalty, ...
                        cfg.pdad.lambda,ls_opts);

                opt_cfg = cfg.pdad.optimizer;
            end

            t_train = tic;

            [p_new,ad_history] = ...
                optimize_distribution_adam( ...
                    p,objective_fun,opt_cfg);

            train_time = toc(t_train);

            p = p_new(:);
            num_updates = num_updates+1;

            update_log(num_updates).update_id = num_updates;
            update_log(num_updates).time_step = k;
            update_log(num_updates).time = t0+k*dt;
            update_log(num_updates).method = method;
            update_log(num_updates).p_before = p_before;
            update_log(num_updates).p_after = p;
            update_log(num_updates).trigger_Ek = Ek;
            update_log(num_updates).trigger_El = El;
            update_log(num_updates).delta_plus = Delta_plus;
            update_log(num_updates).selected_checkpoint = ...
                ad_history.best_iteration;
            update_log(num_updates).best_selection_value = ...
                ad_history.best_selection_value;
            update_log(num_updates).training_time = train_time;

            update_histories{num_updates,1} = ad_history; %#ok<AGROW>

            fprintf('\n');
            fprintf('>>> %s-DT update %d at step %d, t=%.6f\n', ...
                method,num_updates,k,t0+k*dt);
            fprintf('    p: %.10f -> %.10f\n',p_before,p);
            fprintf('    selected checkpoint  = %d\n', ...
                ad_history.best_iteration);
            fprintf('    best selection value = %.6e\n', ...
                ad_history.best_selection_value);
            fprintf('    training time        = %.3f s\n\n', ...
                train_time);

            % Same frozen random basis; only p changes.
            st_i = evaluate_ac_state(Xi,p,basis);

            B = build_ac_periodic_boundary( ...
                Xb,p,basis,cfg.boundary_penalty);

            updated_this_step = true;
        end

        if Delta_l > cfg.adaptation.tau_l
            El = Ek;
        end
    end

    total_time = toc(total_timer) + initial_training_time;

    % =====================================================================
    % Validate saved snapshots
    % =====================================================================
    if any(~isfinite(pred_snapshots(:)))

        missing_columns = find(any(~isfinite(pred_snapshots),1));

        error('Missing snapshot columns: %s', ...
            mat2str(missing_columns));
    end

    if ~isequal(size(pred_snapshots),[numel(ref.xx),25])

        error('Expected uu size %d x 25, obtained %d x %d.', ...
            numel(ref.xx), ...
            size(pred_snapshots,1),size(pred_snapshots,2));
    end

    % =====================================================================
    % Error calculation
    % =====================================================================
    final_rel_l2 = NaN;
    snapshot_rel_l2 = NaN;
    reference_times_match = false;

    if abs(ref.tt(end)-tf) <= 100*eps(max(1,tf))

        final_rel_l2 = relative_l2( ...
            pred_snapshots(:,end), ...
            ref.uu(:,end));
    end

    if size(ref.uu,2) == nsnap

        reference_times_match = ...
            max(abs(ref.tt(:)-snapshot_times(:))) <= ...
            100*eps(max(1,tf));

        if reference_times_match

            snapshot_rel_l2 = relative_l2( ...
                pred_snapshots(:), ...
                ref.uu(:));

        else

            fprintf([ ...
                'NOTE: the reference has 25 columns, but its tt does ', ...
                'not match j/25.  Full snapshot error is not reported.\n']);
        end

    else

        fprintf([ ...
            'NOTE: generated uu has 25 columns, while the loaded ', ...
            'reference has %d columns.  Full snapshot error is not ', ...
            'reported.\n'], ...
            size(ref.uu,2));
    end

    % =====================================================================
    % Output
    % =====================================================================
    out_dir = fullfile( ...
        project_root, ...
        cfg.output_root_name, ...
        [method '_DT']);

    if exist(out_dir,'dir') ~= 7
        mkdir(out_dir);
    end

    results = struct();

    results.cfg = cfg;
    results.method = method;
    results.reference = ref;

    results.snapshot_stride = snapshot_stride;
    results.snapshot_steps = snapshot_steps;
    results.snapshot_times = snapshot_times;
    results.pred_snapshots = pred_snapshots;

    results.p_history = p_history;
    results.E_history = E_history;

    results.initial_history = initial_history;
    results.initial_training_time = initial_training_time;

    results.num_updates = num_updates;
    results.update_log = update_log;
    results.update_histories = update_histories;

    results.reference_times_match = reference_times_match;
    results.final_time_relative_l2 = final_rel_l2;
    results.snapshot_relative_l2 = snapshot_rel_l2;
    results.final_p = p;
    results.total_time = total_time;

    save( ...
        fullfile(out_dir,[method '_DT_AC_results.mat']), ...
        'results','-v7.3');

    % =====================================================================
    % Compact MAT file: same main names as AC_new.mat
    % =====================================================================
    xx = ref.xx;
    uu = pred_snapshots;
    tt = snapshot_times;

    compact_file = fullfile( ...
        out_dir, ...
        sprintf('AC_%s_Nt%d_25snap.mat',method,Nt));

    save( ...
        compact_file, ...
        'xx','uu','tt','snapshot_steps','Nt','dt','-v7.3');

    % =====================================================================
    % Summary CSV
    % =====================================================================
    Method = {method};
    Nt_col = Nt;
    Dt = dt;
    NumSavedSnapshots = nsnap;
    SnapshotStride = snapshot_stride;
    NumCollocation = N;
    NumFeatures = cfg.num_features;
    NumDistributionUpdates = num_updates;
    FinalP = p;
    FinalTimeRelL2 = final_rel_l2;
    SnapshotRelL2 = snapshot_rel_l2;
    TotalTimeSec = total_time;

    summary = table( ...
        Method,Nt_col,Dt, ...
        NumSavedSnapshots,SnapshotStride, ...
        NumCollocation,NumFeatures, ...
        NumDistributionUpdates,FinalP, ...
        FinalTimeRelL2,SnapshotRelL2,TotalTimeSec, ...
        'VariableNames',{ ...
            'Method','Nt','dt', ...
            'NumSavedSnapshots','SnapshotStride', ...
            'NumCollocation','NumFeatures', ...
            'NumDistributionUpdates','FinalP', ...
            'FinalTimeRelL2','SnapshotRelL2','TotalTimeSec'});

    writetable( ...
        summary, ...
        fullfile(out_dir,[method '_DT_AC_summary.csv']));

    % =====================================================================
    % Complete p(t) history
    % =====================================================================
    TimeStep = (0:Nt).';
    Time = t0 + TimeStep*dt;
    P = p_history;
    ResidualRMSE = E_history;

    parameter_history = table( ...
        TimeStep,Time,P,ResidualRMSE);

    writetable( ...
        parameter_history, ...
        fullfile(out_dir,[method '_DT_parameter_history.csv']));

    % =====================================================================
    % Update log
    % =====================================================================
    if num_updates > 0

        UpdateID = (1:num_updates).';
        TimeStep = reshape([update_log.time_step],[],1);
        Time = reshape([update_log.time],[],1);
        PBefore = reshape([update_log.p_before],[],1);
        PAfter = reshape([update_log.p_after],[],1);
        TriggerEk = reshape([update_log.trigger_Ek],[],1);
        TriggerEl = reshape([update_log.trigger_El],[],1);
        DeltaPlus = reshape([update_log.delta_plus],[],1);
        SelectedCheckpoint = ...
            reshape([update_log.selected_checkpoint],[],1);
        BestSelectionValue = ...
            reshape([update_log.best_selection_value],[],1);
        TrainingTimeSec = ...
            reshape([update_log.training_time],[],1);

        updates = table( ...
            UpdateID,TimeStep,Time,PBefore,PAfter, ...
            TriggerEk,TriggerEl,DeltaPlus, ...
            SelectedCheckpoint,BestSelectionValue,TrainingTimeSec);

        writetable( ...
            updates, ...
            fullfile(out_dir,[method '_DT_distribution_updates.csv']));
    end

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('FINAL %s-DT ALLEN--CAHN REPORT\n',method);
    fprintf('============================================================\n');

    if isfinite(final_rel_l2)
        fprintf('final-time rel L2     = %.8e\n',final_rel_l2);
    end

    if isfinite(snapshot_rel_l2)
        fprintf('25-snapshot rel L2    = %.8e\n',snapshot_rel_l2);
    end

    fprintf('distribution updates  = %d\n',num_updates);
    fprintf('final p               = %.10f\n',p);
    fprintf('total time            = %.3f s\n',total_time);
    fprintf('compact MAT file      = %s\n',compact_file);
    fprintf('size(xx)              = %d x %d\n', ...
        size(xx,1),size(xx,2));
    fprintf('size(uu)              = %d x %d\n', ...
        size(uu,1),size(uu,2));
    fprintf('size(tt)              = %d x %d\n', ...
        size(tt,1),size(tt,2));
    fprintf('first/last time       = %.8f / %.8f\n', ...
        tt(1),tt(end));
    fprintf('============================================================\n');

    % =====================================================================
    % Nested snapshot helper
    % =====================================================================
    function store_snapshot(step,coef_now,p_now)

        j = find(snapshot_steps == step,1);

        if isempty(j)
            return;
        end

        st_ref = evaluate_ac_state( ...
            ref.xx,p_now,basis,coef_now);

        pred_snapshots(:,j) = st_ref.u;

        fprintf([ ...
            'Saved snapshot %2d/%2d at step %4d, ', ...
            't=%.8f\n'], ...
            j,nsnap,step,snapshot_times(j));
    end
end
