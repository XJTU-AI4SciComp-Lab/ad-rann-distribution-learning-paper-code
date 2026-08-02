function [r0,info] = frequency_initialization_activation(cfg,activation)
%FREQUENCY_INITIALIZATION_ACTIVATION Frequency initialization for tanh/sin.
%
% This keeps the same candidate set, feasibility test, batching, and exact
% early-stop logic as frequency_initialization.m. Only the activation's
% second derivative in the frequency indicator is changed.
%
% The legacy Gaussian run continues to call frequency_initialization.m
% directly and is therefore numerically unchanged.

    activation = normalize_activation_name(activation);

    domain = cfg.domain;

    a1 = domain(1,1);
    a2 = domain(1,2);
    a3 = domain(2,1);
    a4 = domain(2,2);

    max_r = cfg.initialization.frequency.max_r;
    n_cand_1d = cfg.initialization.frequency.num_candidates;
    Fs = cfg.initialization.frequency.Fs;

    batch_size = get_field( ...
        cfg.initialization.frequency,'batch_size',64);

    exact_early_stop = get_field( ...
        cfg.initialization.frequency,'exact_early_stop',true);

    T = 1/Fs;

    %% Frequency grid
    x1 = a1:T:a2;
    x2 = a3:T:a4;

    [X,Y] = meshgrid(x1,x2);

    L1 = numel(x1);
    L2 = numel(x2);

    n1 = floor(L2/2)+1;
    n2 = floor(L1/2)+1;

    x_fre = [X(:),Y(:)];

    %% RHS spectrum
    rhs_value = rhs(x_fre);

    P_rhs = abs(fft2(reshape(rhs_value,L2,L1)));
    P_rhs = P_rhs(1:n1,1:n2);

    [~,idx_rhs] = max(P_rhs,[],'all','linear');
    [row_rhs,col_rhs] = ind2sub([n1,n2],idx_rhs);

    %% Original 2-D candidate grid
    W = linspace(0,max_r,n_cand_1d+1);
    W = W(2:end);

    [W1,W2] = meshgrid(W,W);

    W_candidate = [W1(:)';W2(:)'];
    num_candidate = size(W_candidate,2);

    candidate_norm = sum(W_candidate.^2,1);
    original_index = 1:num_candidate;

    if exact_early_stop
        order_table = [candidate_norm(:),original_index(:)];
        [~,order] = sortrows(order_table,[1 2]);
    else
        order = original_index(:);
    end

    best_norm = Inf;
    best_original_index = Inf;
    best_candidate = [];

    num_fft_candidates = 0;
    num_batches = 0;

    pos = 1;

    while pos <= num_candidate

        if exact_early_stop && isfinite(best_norm)
            next_norm = candidate_norm(order(pos));
            tol_norm = 64*eps(max(1,best_norm));

            if next_norm > best_norm+tol_norm
                break
            end
        end

        last = min(pos+batch_size-1,num_candidate);
        ids = order(pos:last).';

        if exact_early_stop && isfinite(best_norm)
            tol_norm = 64*eps(max(1,best_norm));
            ids = ids(candidate_norm(ids) <= best_norm+tol_norm);

            if isempty(ids)
                break
            end
        end

        num_batches = num_batches+1;
        num_fft_candidates = num_fft_candidates+numel(ids);

        nb = numel(ids);

        w1 = reshape(W_candidate(1,ids),1,1,nb);
        w2 = reshape(W_candidate(2,ids),1,1,nb);

        S = X.*w1 + Y.*w2;

        sigma_second = activation_second_derivative(S,activation);

        % Same indicator as the original routine:
        % response = -d^2 phi/dx^2 = -(w1.^2)*sigma''(S).
        response = -(w1.^2).*sigma_second;

        P = abs(fft2(response));
        P = P(1:n1,1:n2,:);

        P2 = reshape(P,n1*n2,nb);
        [~,idx] = max(P2,[],1);

        peak_row = mod(idx-1,n1)+1;
        peak_col = floor((idx-1)/n1)+1;

        feasible = ...
            (peak_row > row_rhs) & ...
            (peak_col > col_rhs);

        feasible_ids = ids(feasible);

        for j = 1:numel(feasible_ids)

            id = feasible_ids(j);
            nrm = candidate_norm(id);

            if isfinite(best_norm)
                tol_best = 64*eps(max([1,nrm,best_norm]));
            else
                tol_best = 0;
            end

            if ~isfinite(best_norm) || ...
                    nrm < best_norm-tol_best || ...
                    (abs(nrm-best_norm) <= tol_best && ...
                     id < best_original_index)

                best_norm = nrm;
                best_original_index = id;
                best_candidate = W_candidate(:,id);
            end
        end

        pos = last+1;
    end

    if isempty(best_candidate)

        warning( ...
            ['Frequency initialization failed to find an admissible ', ...
             'candidate. Using manual initialization.']);

        r0 = cfg.initialization.manual_p(:);

    else
        r0 = best_candidate(:);
    end

    info.rhs_peak_index = [col_rhs-1;row_rhs-1];
    info.num_fft_candidates = num_fft_candidates;
    info.num_batches = num_batches;
    info.total_candidates = num_candidate;
    info.exact_early_stop = logical(exact_early_stop);
    info.selected_norm_sq = sum(r0.^2);
    info.activation = activation;

    fprintf( ...
        'Frequency initialization (%s) r = (%.6f, %.6f)\n', ...
        activation,r0(1),r0(2));

    fprintf( ...
        'Frequency FFT candidates = %d / %d, batches = %d\n', ...
        num_fft_candidates,num_candidate,num_batches);
end


function d2 = activation_second_derivative(S,activation)

    switch activation

        case 'tanh'
            T = tanh(S);
            d2 = -2*T.*(1-T.^2);

        case 'sin'
            d2 = -sin(S);

        otherwise
            error('This helper is intended for tanh or sin.');
    end
end


function activation = normalize_activation_name(activation)

    activation = lower(strtrim(char(activation)));

    switch activation
        case {'gaussian','gauss'}
            activation = 'gaussian';
        case 'tanh'
            activation = 'tanh';
        case {'sin','sine'}
            activation = 'sin';
        otherwise
            error('Unknown activation: %s',activation);
    end
end


function value = get_field(s,name,default_value)

    if isfield(s,name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = default_value;
    end
end
