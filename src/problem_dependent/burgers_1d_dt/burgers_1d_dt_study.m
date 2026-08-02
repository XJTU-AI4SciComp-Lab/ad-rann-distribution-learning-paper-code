function results = burgers_1d_dt_study(cfg,project_root)
%BURGERS_1D_DT_STUDY Shared driver for DDAD-DT and PDAD-DT.
%
% PDE:
%
%   u_t + u u_x - nu u_xx = 0,
%   x in (-1,1), t in (0,1].
%
% Time discretization:
%   first step : first-order semi-implicit startup;
%   k >= 2     : BDF2 with extrapolated convection coefficient.
%
% Adaptive rule:
%   E_k is the PDE-row RMSE only; the boundary penalty is not part of E_k.
%   Each physical time step performs at most one distribution update.
%
% DDAD-DT update:
%   target = previously accepted numerical solution u_h^{k-1}.
%
% PDAD-DT update:
%   p is trained directly against the current BDF2 discrete PDE system.
%
% In both cases, after p changes the SAME physical time step is rebuilt and
% solved again.  Final PDE coefficients always come from the project
% unregularized solve_least_squares routine.

    method = upper(strtrim(char(cfg.method)));

    if ~ismember(method,{'DDAD','PDAD'})
        error('cfg.method must be DDAD or PDAD.');
    end

    % =====================================================================
    % Reference data
    % =====================================================================
    ref = load_burgers_reference(project_root);

    % =====================================================================
    % Problem / grid
    % =====================================================================
    xa = cfg.x_domain(1);
    xb = cfg.x_domain(2);

    t0 = cfg.t_domain(1);
    tf = cfg.t_domain(2);

    Nt = cfg.num_time_steps;
    dt = (tf-t0)/Nt;

    Xi = linspace( ...
        xa+1e-6, ...
        xb-1e-6, ...
        cfg.num_collocation_points).';

    Xb = [xa;xb];

    u0 = burgers_initial_condition(Xi);

    % =====================================================================
    % Frozen random basis -- common src implementation
    % =====================================================================
    basis = build_random_weights_nd( ...
        cfg.num_features, ...
        cfg.x_domain(:).', ...
        cfg.seed);

    % =====================================================================
    % Linear/ridge options
    % =====================================================================
    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    % =====================================================================
    % Optional common DDAD initialization from u^0
    % =====================================================================
    p = cfg.initial_p(:);

    initial_training_time = 0;
    initial_history = [];

    if cfg.initialization.use_ddad

        cache0 = prepare_data_cache(Xi,u0,basis);

        objective0 = @(pp) ...
            evaluate_data_reduced_fast( ...
                pp,cache0,cfg.initialization.lambda, ...
                ls_opts,cfg.activation);

        t_init = tic;

        [p,initial_history] = ...
            optimize_distribution_adam( ...
                p,objective0,cfg.initialization.optimizer);

        initial_training_time = toc(t_init);

        fprintf('\n');
        fprintf('============================================================\n');
        fprintf('Initial DDAD distribution initialization\n');
        fprintf('============================================================\n');
        fprintf('p*                  = %.10f\n',p);
        fprintf('selected checkpoint = %d\n',initial_history.best_iteration);
        fprintf('training time       = %.3f s\n',initial_training_time);
        fprintf('============================================================\n\n');
    end

    % =====================================================================
    % Reference snapshot map
    % =====================================================================
    nsnap = ref.num_snapshots;

    snapshot_steps = round(linspace(0,Nt,nsnap));

    if numel(unique(snapshot_steps)) ~= nsnap
        error('Nt is too small for the number of stored reference snapshots.');
    end

    pred_snapshots = nan(numel(ref.xx),nsnap);
    pred_snapshots(:,1) = burgers_initial_condition(ref.xx);

    snapshot_times = t0 + snapshot_steps(:)*dt;

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

    % =====================================================================
    % Helper: rebuild feature matrices whenever p changes
    % =====================================================================
    state_i = evaluate_burgers_state(Xi,p,basis);
    state_b = evaluate_burgers_state(Xb,p,basis);

    B = cfg.boundary_penalty*state_b.phi;
    rhs_b = zeros(numel(Xb),1);

    % =====================================================================
    % First time step
    %
    %   u^1 + dt u^0 u_x^1 - dt nu u_xx^1 = u^0
    % =====================================================================
    total_timer = tic;

    A = ...
        state_i.phi + ...
        dt*u0.*state_i.ux - ...
        dt*cfg.nu*state_i.uxx;

    rhs_i = u0;

    coef = solve_least_squares( ...
        [A;B], ...
        [rhs_i;cfg.boundary_penalty*rhs_b], ...
        cfg.linear_solver);

    if iscell(coef)
        error('Unexpected solve_least_squares return type.');
    end

    % Some project versions return [coef,info], some callers request one
    % output.  The one-output form above intentionally keeps only coef.
    coef = coef(:);

    E1 = sqrt(mean((A*coef-rhs_i).^2));

    u_ll = u0;
    u_l = state_i.phi*coef;

    El = E1;

    E_history(2) = E1;
    p_history(2) = p;

    store_snapshot(1,coef,p);

    if cfg.verbose
        fprintf('============================================================\n');
        fprintf('%s-DT Burgers\n',method);
        fprintf('============================================================\n');
        fprintf('Nt / dt             = %d / %.6e\n',Nt,dt);
        fprintf('collocation / m     = %d / %d\n', ...
            cfg.num_collocation_points,cfg.num_features);
        fprintf('nu                   = %.12e\n',cfg.nu);
        fprintf('first-step E1        = %.6e\n',E1);
        fprintf('initial p            = %.8f\n',p);
        fprintf('============================================================\n');
    end

    % =====================================================================
    % BDF2 time loop
    % =====================================================================
    k = 2;
    updated_this_step = false;

    while k <= Nt

        adv = 2*u_l-u_ll;

        A = ...
            3*state_i.phi + ...
            2*dt*( ...
                adv.*state_i.ux - ...
                cfg.nu*state_i.uxx);

        rhs_i = 4*u_l-u_ll;

        coef = solve_least_squares( ...
            [A;B], ...
            [rhs_i;cfg.boundary_penalty*rhs_b], ...
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

            u_new = state_i.phi*coef;

            u_ll = u_l;
            u_l = u_new;

            E_history(k+1) = Ek;
            p_history(k+1) = p;

            store_snapshot(k,coef,p);

            k = k+1;
            updated_this_step = false;

        else

            % =============================================================
            % One adaptive distribution update at this physical time step.
            % =============================================================
            p_before = p;

            if strcmp(method,'DDAD')

                % Target = previously accepted full numerical solution.
                cache = prepare_data_cache(Xi,u_l,basis);

                objective_fun = @(pp) ...
                    evaluate_data_reduced_fast( ...
                        pp,cache,cfg.ddad.lambda, ...
                        ls_opts,cfg.activation);

                opt_cfg = cfg.ddad.optimizer;

            else

                % Train directly from the SAME BDF2 operator that is used
                % in the physical time-step solve.
                objective_fun = @(pp) ...
                    evaluate_burgers_pdad_reduced_fast( ...
                        pp,Xi,Xb,basis,u_l,u_ll, ...
                        dt,cfg.nu,cfg.boundary_penalty, ...
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
            fprintf('>>> %s-DT distribution update %d at step %d, t=%.6f\n', ...
                method,num_updates,k,t0+k*dt);
            fprintf('    p: %.10f -> %.10f\n',p_before,p);
            fprintf('    selected checkpoint = %d\n',ad_history.best_iteration);
            fprintf('    best selection value= %.6e\n', ...
                ad_history.best_selection_value);
            fprintf('    training time       = %.3f s\n\n',train_time);

            % Rebuild the full trial space and re-solve the SAME k.
            state_i = evaluate_burgers_state(Xi,p,basis);
            state_b = evaluate_burgers_state(Xb,p,basis);

            B = cfg.boundary_penalty*state_b.phi;

            updated_this_step = true;
        end

        % Algorithm-3 reference residual update.
        if Delta_l > cfg.adaptation.tau_l
            El = Ek;
        end
    end

    total_time = toc(total_timer) + initial_training_time;

    % =====================================================================
    % Final errors against stored reference snapshots
    % =====================================================================
    if any(~isfinite(pred_snapshots(:)))
        missing = find(any(~isfinite(pred_snapshots),1));
        error('Missing predicted reference snapshots at columns: %s', ...
            mat2str(missing));
    end

    overall_rel_l2 = relative_l2( ...
        pred_snapshots(:), ...
        ref.exact(:));

    final_rel_l2 = relative_l2( ...
        pred_snapshots(:,end), ...
        ref.exact(:,end));

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
    results.snapshot_steps = snapshot_steps(:);
    results.snapshot_times = snapshot_times;

    results.pred_snapshots = pred_snapshots;

    results.p_history = p_history;
    results.E_history = E_history;

    results.initial_p_selected = p_history(1);
    results.initial_training_time = initial_training_time;
    results.initial_history = initial_history;

    results.num_updates = num_updates;
    results.update_log = update_log;
    results.update_histories = update_histories;

    results.overall_relative_l2 = overall_rel_l2;
    results.final_time_relative_l2 = final_rel_l2;
    results.final_p = p;
    results.total_time = total_time;

    save( ...
        fullfile(out_dir,[method '_DT_burgers_results.mat']), ...
        'results','-v7.3');

    Method = {method};
    Nt_col = Nt;
    Dt = dt;
    NumCollocation = cfg.num_collocation_points;
    NumFeatures = cfg.num_features;
    NumDistributionUpdates = num_updates;
    FinalP = p;
    OverallRelL2 = overall_rel_l2;
    FinalTimeRelL2 = final_rel_l2;
    TotalTimeSec = total_time;

    summary = table( ...
        Method,Nt_col,Dt,NumCollocation,NumFeatures, ...
        NumDistributionUpdates,FinalP, ...
        OverallRelL2,FinalTimeRelL2,TotalTimeSec, ...
        'VariableNames',{ ...
            'Method','Nt','dt','NumCollocation','NumFeatures', ...
            'NumDistributionUpdates','FinalP', ...
            'OverallRelL2','FinalTimeRelL2','TotalTimeSec'});

    writetable( ...
        summary, ...
        fullfile(out_dir,[method '_DT_burgers_summary.csv']));

    if num_updates > 0

        UpdateID = (1:num_updates).';
        TimeStep = reshape([update_log.time_step],[],1);
        Time = reshape([update_log.time],[],1);
        PBefore = reshape([update_log.p_before],[],1);
        PAfter = reshape([update_log.p_after],[],1);
        TriggerEk = reshape([update_log.trigger_Ek],[],1);
        TriggerEl = reshape([update_log.trigger_El],[],1);
        DeltaPlus = reshape([update_log.delta_plus],[],1);
        SelectedCheckpoint = reshape([update_log.selected_checkpoint],[],1);
        BestSelectionValue = reshape([update_log.best_selection_value],[],1);
        TrainingTimeSec = reshape([update_log.training_time],[],1);

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
    fprintf('FINAL %s-DT BURGERS REPORT\n',method);
    fprintf('============================================================\n');
    fprintf('overall relative L2 = %.8e\n',overall_rel_l2);
    fprintf('final-time rel L2    = %.8e\n',final_rel_l2);
    fprintf('distribution updates = %d\n',num_updates);
    fprintf('final p              = %.10f\n',p);
    fprintf('total time            = %.3f s\n',total_time);
    fprintf('results directory     = %s\n',out_dir);
    fprintf('============================================================\n');

    % =====================================================================
    % Nested snapshot helper
    % =====================================================================
    function store_snapshot(step,coef_now,p_now)

        j = find(snapshot_steps == step,1);

        if isempty(j)
            return;
        end

        st_ref = evaluate_burgers_state( ...
            ref.xx,p_now,basis,coef_now);

        pred_snapshots(:,j) = st_ref.phi*coef_now;
    end
end
