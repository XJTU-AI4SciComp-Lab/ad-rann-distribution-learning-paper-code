function [objective,grad,info] = ...
    evaluate_helmholtz_reduced_fast(p,cache,lambda,ls_opts)
%EVALUATE_HELMHOLTZ_REDUCED_FAST Fast Gaussian Helmholtz PDAD evaluator.
%
% The full dM/dp matrices are never stored. Only the two derivative actions
%
%   (dM/dp_1) w_lambda,  (dM/dp_2) w_lambda
%
% are formed. The ridge solve and Adam optimizer are the common src versions.

    if nargin < 4 || isempty(ls_opts)
        ls_opts = struct();
    end

    p = p(:);

    if numel(p) ~= 2 || any(~isfinite(p)) || any(p <= 0)
        objective = Inf;
        grad = [NaN;NaN];
        info = struct('residual_mse',Inf);
        return;
    end

    p1 = p(1);
    p2 = p(2);

    %% Interior feature values and preactivation derivatives

    Si = p1*cache.Qi1 + p2*cache.Qi2;
    Ai = activation_derivatives(Si,'gaussian',3);

    q = ...
        p1^2*cache.Z1sq + ...
        p2^2*cache.Z2sq;

    M_i = ...
        Ai.d2.*q + ...
        cache.k^2*Ai.phi;

    %% Boundary block

    Sb = p1*cache.Qb1 + p2*cache.Qb2;
    Ab = activation_derivatives(Sb,'gaussian',1);

    M_b = cache.boundary_penalty*Ab.phi;

    M = [M_i;M_b];

    %% Common ridge solve

    [w,ridge_info] = solve_ridge(M,cache.y,lambda,ls_opts);

    if isa(w,'gpuArray')
        w = gather(w);
    end

    w = w(:);

    residual = M*w-cache.y;
    N = cache.num_rows;

    residual_sq = real(residual'*residual);
    coefficient_sq = real(w'*w);

    objective = ...
        (residual_sq + lambda*coefficient_sq)/(2*N);

    %% Analytic envelope-gradient actions

    dq1 = 2*p1*cache.Z1sq;
    dq2 = 2*p2*cache.Z2sq;

    qw = q(:).*w;

    v_i1 = ...
        Ai.d2*(dq1(:).*w) + ...
        (Ai.d3.*cache.Qi1)*qw + ...
        cache.k^2*((Ai.d1.*cache.Qi1)*w);

    v_i2 = ...
        Ai.d2*(dq2(:).*w) + ...
        (Ai.d3.*cache.Qi2)*qw + ...
        cache.k^2*((Ai.d1.*cache.Qi2)*w);

    v_b1 = ...
        cache.boundary_penalty*((Ab.d1.*cache.Qb1)*w);

    v_b2 = ...
        cache.boundary_penalty*((Ab.d1.*cache.Qb2)*w);

    v1 = [v_i1;v_b1];
    v2 = [v_i2;v_b2];

    grad = [ ...
        real(residual'*v1)/N; ...
        real(residual'*v2)/N];

    info = struct();
    info.w = w;
    info.residual_mse = residual_sq/N;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;
    info.num_rows = N;
    info.num_features = cache.num_features;

    if isfield(ls_opts,'compute_spectrum') && ls_opts.compute_spectrum

        singular_values = svd(M,'econ');

        info.sigma_max = max(singular_values);
        info.sigma_min = min(singular_values);

        if lambda > 0
            info.ridge_condition_number = ...
                (info.sigma_max^2+lambda) / ...
                (info.sigma_min^2+lambda);
        else
            info.ridge_condition_number = Inf;
        end
    end
end
