function [p0,info] = ...
    frequency_initialize_gaussian_2d(cfg,rhs_fun)
%FREQUENCY_INITIALIZE_GAUSSIAN_2D Common 2-D Gaussian FFT initializer.
%
% The problem-dependent caller supplies only the right-hand side:
%
%   rhs_value = rhs_fun(X_frequency).
%
% The candidate set, feasibility rule, batched FFT implementation,
% exact early stopping, and minimum-norm tie-breaking are shared by the
% Poisson and Helmholtz initializers.

    domain = cfg.domain;

    x_min = domain(1,1);
    x_max = domain(1,2);
    y_min = domain(2,1);
    y_max = domain(2,2);

    max_r = cfg.initialization.frequency.max_r;
    n_cand_1d = cfg.initialization.frequency.num_candidates;
    Fs = cfg.initialization.frequency.Fs;

    batch_size = get_field( ...
        cfg.initialization.frequency,'batch_size',64);

    exact_early_stop = get_field( ...
        cfg.initialization.frequency,'exact_early_stop',true);

    T = 1/Fs;

    %% Frequency grid
    x1 = x_min:T:x_max;
    x2 = y_min:T:y_max;

    [X,Y] = meshgrid(x1,x2);

    L1 = numel(x1);
    L2 = numel(x2);

    n1 = floor(L2/2)+1;
    n2 = floor(L1/2)+1;

    X_frequency = [X(:),Y(:)];

    %% Right-hand-side spectrum
    rhs_value = rhs_fun(X_frequency);

    P_rhs = abs(fft2(reshape(rhs_value,L2,L1)));
    P_rhs = P_rhs(1:n1,1:n2);

    [~,idx_rhs] = max(P_rhs,[],'all','linear');
    [row_rhs,col_rhs] = ind2sub([n1,n2],idx_rhs);

    %% Original two-dimensional candidate grid
    W = linspace(0,max_r,n_cand_1d+1);
    W = W(2:end);

    [W1,W2] = meshgrid(W,W);

    candidates = [W1(:).';W2(:).'];
    num_candidates = size(candidates,2);

    candidate_norm_sq = sum(candidates.^2,1);
    original_index = 1:num_candidates;

    if exact_early_stop
        order_table = [candidate_norm_sq(:),original_index(:)];
        [~,order] = sortrows(order_table,[1,2]);
    else
        order = original_index(:);
    end

    best_norm_sq = Inf;
    best_original_index = Inf;
    best_candidate = [];

    num_fft_candidates = 0;
    num_batches = 0;

    pos = 1;

    while pos <= num_candidates

        if exact_early_stop && isfinite(best_norm_sq)
            next_norm = candidate_norm_sq(order(pos));
            tol_norm = 64*eps(max(1,best_norm_sq));

            if next_norm > best_norm_sq+tol_norm
                break;
            end
        end

        last = min(pos+batch_size-1,num_candidates);
        ids = order(pos:last).';

        if exact_early_stop && isfinite(best_norm_sq)
            tol_norm = 64*eps(max(1,best_norm_sq));
            ids = ids(candidate_norm_sq(ids) <= best_norm_sq+tol_norm);

            if isempty(ids)
                break;
            end
        end

        num_batches = num_batches+1;
        num_fft_candidates = num_fft_candidates+numel(ids);

        nb = numel(ids);

        w1 = reshape(candidates(1,ids),1,1,nb);
        w2 = reshape(candidates(2,ids),1,1,nb);

        % Established Gaussian frequency indicator:
        % response = -d^2 phi/dx^2 for phi = exp(-S^2).
        S = X.*w1 + Y.*w2;
        E = exp(-S.^2);

        response = ...
            -(w1.^2).*(4*S.^2-2).*E;

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
            norm_sq = candidate_norm_sq(id);

            if isfinite(best_norm_sq)
                tol_best = 64*eps(max([1,norm_sq,best_norm_sq]));
            else
                tol_best = 0;
            end

            if ~isfinite(best_norm_sq) || ...
                    norm_sq < best_norm_sq-tol_best || ...
                    (abs(norm_sq-best_norm_sq) <= tol_best && ...
                     id < best_original_index)

                best_norm_sq = norm_sq;
                best_original_index = id;
                best_candidate = candidates(:,id);
            end
        end

        pos = last+1;
    end

    if isempty(best_candidate)

        warning( ...
            ['Frequency initialization found no admissible candidate. ', ...
             'Using the problem-specific manual fallback.']);

        p0 = cfg.initialization.manual_p(:);
        used_fallback = true;

    else

        p0 = best_candidate(:);
        used_fallback = false;
    end

    info = struct();
    info.used_frequency_search = true;
    info.used_manual_fallback = used_fallback;
    info.rhs_peak_index = [col_rhs-1;row_rhs-1];
    info.num_fft_candidates = num_fft_candidates;
    info.num_batches = num_batches;
    info.total_candidates = num_candidates;
    info.exact_early_stop = logical(exact_early_stop);
    info.selected_norm_sq = sum(p0.^2);
end


function value = get_field(s,name,default_value)

    if isfield(s,name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = default_value;
    end
end
