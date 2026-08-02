function [p_best, history] = optimize_distribution_adam(p0, objective_fun, opts)
%OPTIMIZE_DISTRIBUTION_ADAM Projected Adam for AD-RaNN parameters.
%
% When a component is clipped by the box projection, the corresponding
% Adam first/second moment components are reset to zero.
%
% If opts.store_moments=true, history stores:
%   m_eval, v_eval
%   m_after_update, v_after_update
%   hit_bound
%   final_m, final_v

    if nargin < 3 || isempty(opts)
        opts = struct();
    end

    p0 = p0(:);
    np = numel(p0);

    maxit = get_opt(opts,'maxit',20);

    lb = expand_bound( ...
        get_opt(opts,'lower_bound',-Inf),np);

    ub = expand_bound( ...
        get_opt(opts,'upper_bound',Inf),np);

    parameterization = lower(char( ...
        get_opt(opts,'parameterization','direct')));

    selection_metric = lower(char( ...
        get_opt(opts,'selection_metric','objective')));

    grad_tol = get_opt(opts,'grad_tol',0);
    step_tol = get_opt(opts,'step_tol',0);
    rel_obj_tol = get_opt(opts,'relative_obj_tol',0);
    patience = get_opt(opts,'patience',Inf);
    min_delta = get_opt(opts,'min_delta',0);
    verbose = logical(get_opt(opts,'verbose',true));

    store_moments = logical(get_opt(opts,'store_moments',true));
    store_full_info = logical(get_opt(opts,'store_full_info',false));

    if any(lb > ub)
        error('Lower bounds must not exceed upper bounds.');
    end

    p = min(max(p0,lb),ub);

    switch parameterization
        case 'direct'
            theta = p;

        case 'log'
            if any(p <= 0) || any(lb <= 0)
                error('Log parameterization requires positive bounds.');
            end
            theta = log(p);

        otherwise
            error('parameterization must be ''direct'' or ''log''.');
    end

    adam_opts.learning_rate = get_opt(opts,'learning_rate',0.5);
    adam_opts.beta1 = get_opt(opts,'beta1',0.9);
    adam_opts.beta2 = get_opt(opts,'beta2',0.999);
    adam_opts.epsilon = get_opt(opts,'epsilon',1e-8);

    m = zeros(size(theta));
    v = zeros(size(theta));

    nmax = maxit+1;

    history.iteration = nan(nmax,1);
    history.objective = nan(nmax,1);
    history.residual_mse = nan(nmax,1);
    history.selection_value = nan(nmax,1);
    history.grad_norm = nan(nmax,1);
    history.step_norm = nan(nmax,1);
    history.eval_time = nan(nmax,1);
    history.p = nan(np,nmax);

    if store_moments
        history.m_eval = nan(np,nmax);
        history.v_eval = nan(np,nmax);
        history.m_after_update = nan(np,nmax);
        history.v_after_update = nan(np,nmax);
        history.hit_bound = false(np,nmax);
    else
        history.m_eval = [];
        history.v_eval = [];
        history.m_after_update = [];
        history.v_after_update = [];
        history.hit_bound = [];
    end

    if store_full_info
        history.info = cell(nmax,1);
    else
        history.info = {};
    end

    best_value = Inf;
    p_best = p;
    best_index = 1;

    no_improve = 0;
    stop_reason = 'maxit';
    previous_objective = NaN;

    total_timer = tic;

    if verbose
        fprintf('==== AD-RaNN Adam optimization ====\n');
    end

    for k = 0:maxit

        switch parameterization
            case 'direct'
                p = theta;
            case 'log'
                p = exp(theta);
        end

        p = min(max(p,lb),ub);

        idx = k+1;

        if store_moments
            history.m_eval(:,idx) = m;
            history.v_eval(:,idx) = v;
        end

        t_eval = tic;
        [obj,grad_p,info] = objective_fun(p);
        eval_time = toc(t_eval);

        grad_p = grad_p(:);

        if ~isscalar(obj) || ~isfinite(obj) || any(~isfinite(grad_p))
            error('objective_fun returned a non-finite objective or gradient.');
        end

        residual_mse = NaN;

        if isstruct(info) && isfield(info,'residual_mse')
            residual_mse = info.residual_mse;
        end

        switch selection_metric
            case 'objective'
                select_value = obj;

            case 'residual_mse'
                if ~isfinite(residual_mse)
                    error(['selection_metric=''residual_mse'' requires ', ...
                           'info.residual_mse.']);
                end
                select_value = residual_mse;

            otherwise
                error('Unknown selection_metric.');
        end

        history.iteration(idx) = k;
        history.objective(idx) = obj;
        history.residual_mse(idx) = residual_mse;
        history.selection_value(idx) = select_value;
        history.grad_norm(idx) = norm(grad_p);
        history.eval_time(idx) = eval_time;
        history.p(:,idx) = p;

        if store_full_info
            history.info{idx} = info;
        end

        if select_value < best_value-min_delta
            best_value = select_value;
            p_best = p;
            best_index = idx;
            no_improve = 0;
        else
            no_improve = no_improve+1;
        end

        if verbose
            fprintf( ...
                'it=%02d | p=%s | obj=%.6e | MSE=%.6e | |g|=%.3e\n', ...
                k,mat2str(p.',6),obj,residual_mse,norm(grad_p));
        end

        if k == maxit
            break
        end

        if grad_tol > 0 && norm(grad_p) <= grad_tol
            stop_reason = 'grad_tol';
            break
        end

        if isfinite(patience) && no_improve >= patience
            stop_reason = 'patience';
            break
        end

        if rel_obj_tol > 0 && isfinite(previous_objective)
            rel_change = ...
                abs(obj-previous_objective) / ...
                max(abs(previous_objective),eps);

            if rel_change <= rel_obj_tol
                stop_reason = 'relative_obj_tol';
                break
            end
        end

        previous_objective = obj;

        if strcmp(parameterization,'log')
            grad_theta = grad_p.*p;
        else
            grad_theta = grad_p;
        end

        [theta_raw,m_new,v_new,~] = ...
            adam_step(theta,grad_theta,m,v,k+1,adam_opts);

        if strcmp(parameterization,'log')

            p_raw = exp(theta_raw);

            hit_bound = ...
                (p_raw < lb) | ...
                (p_raw > ub);

            p_candidate = ...
                min(max(p_raw,lb),ub);

            m_new(hit_bound) = 0;
            v_new(hit_bound) = 0;

            theta_candidate = log(p_candidate);

        else

            p_raw = theta_raw;

            hit_bound = ...
                (p_raw < lb) | ...
                (p_raw > ub);

            p_candidate = ...
                min(max(p_raw,lb),ub);

            m_new(hit_bound) = 0;
            v_new(hit_bound) = 0;

            theta_candidate = p_candidate;
        end

        history.step_norm(idx) = norm(p_candidate-p);

        if store_moments
            history.m_after_update(:,idx) = m_new;
            history.v_after_update(:,idx) = v_new;
            history.hit_bound(:,idx) = hit_bound;
        end

        theta = theta_candidate;
        m = m_new;
        v = v_new;

        if step_tol > 0 && history.step_norm(idx) <= step_tol
            stop_reason = 'step_tol';
            break
        end
    end

    last = find(isfinite(history.iteration),1,'last');

    scalar_fields = { ...
        'iteration','objective','residual_mse', ...
        'selection_value','grad_norm','step_norm','eval_time'};

    for j = 1:numel(scalar_fields)
        name = scalar_fields{j};
        history.(name) = history.(name)(1:last,:);
    end

    history.p = history.p(:,1:last);

    if store_moments
        history.m_eval = history.m_eval(:,1:last);
        history.v_eval = history.v_eval(:,1:last);
        history.m_after_update = history.m_after_update(:,1:last);
        history.v_after_update = history.v_after_update(:,1:last);
        history.hit_bound = history.hit_bound(:,1:last);
    end

    if store_full_info
        history.info = history.info(1:last);
    end

    history.best_index = best_index;
    history.best_iteration = history.iteration(best_index);
    history.best_selection_value = best_value;
    history.p_best = p_best;
    history.selection_metric = selection_metric;
    history.stop_reason = stop_reason;
    history.total_time = toc(total_timer);

    history.final_m = m;
    history.final_v = v;

    if verbose
        fprintf('best p = %s\n',mat2str(p_best.',8));
        fprintf('best %s = %.6e\n',selection_metric,best_value);
        fprintf('stop reason = %s\n',stop_reason);
    end
end


function value = get_opt(opts,name,default_value)

    if isfield(opts,name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default_value;
    end
end


function b = expand_bound(b,n)

    if isscalar(b)
        b = repmat(b,n,1);
    else
        b = b(:);
    end

    if numel(b) ~= n
        error('Bound size must be scalar or match p.');
    end
end
