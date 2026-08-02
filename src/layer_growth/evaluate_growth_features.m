function out = evaluate_growth_features( ...
    X,model,rho,base_state,max_order)
%EVALUATE_GROWTH_FEATURES Evaluate localized layer-growth features.
%
% out = evaluate_growth_features(X,model,rho,base_state,max_order)
%
% base_state is the FROZEN base solution used to define the growth block:
%   base_state.u       : N-by-1
%   base_state.d1{k}   : first derivative in coordinate k
%   base_state.d2{k}   : second derivative in coordinate k
%
% max_order = 0, 1, or 2.
%
% Feature:
%
%   psi_j = exp(-rho^2 S_j)
%
% where
%
%   S_j = ||beta_j||^2 (u_0-u_0(c_j))^2
%         + 1/2 sum_k beta_{k,j}^2 (x_k-c_{j,k})^2.
%
% Analytic derivatives:
%
%   d_k psi = -rho^2 (d_k S) psi,
%
%   d_kk psi = [rho^4(d_k S)^2-rho^2 d_kk S] psi.

    if nargin < 5 || isempty(max_order)
        max_order = 2;
    end

    if ~ismember(max_order,[0,1,2])
        error('max_order must be 0, 1, or 2.');
    end

    rho = rho(:);

    if numel(rho) ~= 1 || ~isfinite(rho) || rho <= 0
        error('rho must be one finite positive scalar.');
    end

    N = size(X,1);

    if size(X,2) ~= model.dim
        error('X dimension does not match growth model.');
    end

    if ~isstruct(base_state) || ~isfield(base_state,'u')
        error('base_state.u is required.');
    end

    u0 = base_state.u(:);

    if numel(u0) ~= N
        error('base_state.u must contain N values.');
    end

    du = u0-model.center_values.';

    S = du.^2 .* model.direction_norm_sq.';

    for k = 1:model.dim
        dx = X(:,k)-model.centers(:,k).';
        S = S + 0.5*dx.^2.*(model.directions(k,:).^2);
    end

    S = max(S,0);

    phi = exp(-(rho.^2)*S);

    out = struct();
    out.phi = phi;
    out.S = S;
    out.d1 = cell(model.dim,1);
    out.d2 = cell(model.dim,1);

    if max_order == 0
        return;
    end

    if ~isfield(base_state,'d1') || ...
            numel(base_state.d1) < model.dim
        error('base_state.d1 is required for max_order >= 1.');
    end

    if max_order >= 2 && ...
            (~isfield(base_state,'d2') || ...
             numel(base_state.d2) < model.dim)
        error('base_state.d2 is required for max_order = 2.');
    end

    norm_sq = model.direction_norm_sq.';

    for k = 1:model.dim

        uk = base_state.d1{k}(:);

        if numel(uk) ~= N
            error('base_state.d1{%d} must contain N values.',k);
        end

        dx = X(:,k)-model.centers(:,k).';
        beta_sq = model.directions(k,:).^2;

        dS = ...
            2*du.*uk.*norm_sq + ...
            dx.*beta_sq;

        out.d1{k} = ...
            -(rho.^2)*dS.*phi;

        if max_order >= 2

            ukk = base_state.d2{k}(:);

            if numel(ukk) ~= N
                error('base_state.d2{%d} must contain N values.',k);
            end

            d2S = ...
                2*(uk.^2 + du.*ukk).*norm_sq + ...
                beta_sq;

            out.d2{k} = ...
                ((rho.^4)*(dS.^2) - ...
                 (rho.^2)*d2S).*phi;
        end
    end
end
