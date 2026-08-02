function [objective,grad,info] = ...
    evaluate_growth_ddad_reduced_fast(rho,cache,lambda,ls_opts)
%EVALUATE_GROWTH_DDAD_REDUCED_FAST Reduced DDAD objective for growth scale.
%
% Inner problem:
%
%   w_lambda(rho)
%     = argmin_w || [Phi_base,Psi(rho)] w-y ||_2^2
%                 + lambda ||w||_2^2,
%
%   Psi(rho) = exp(-rho^2 S).
%
% Reduced objective:
%
%   f(rho)
%     = (||res||_2^2 + lambda||w_lambda||_2^2)/(2N).
%
% Only dPsi/drho acting on the growth coefficients is formed.

    if nargin < 4 || isempty(ls_opts)
        ls_opts = struct();
    end

    rho = rho(:);

    if numel(rho) ~= 1 || ~isfinite(rho) || rho <= 0
        error('rho must be one finite positive scalar.');
    end

    if ~isscalar(lambda) || ~isfinite(lambda) || lambda < 0
        error('lambda must be a finite nonnegative scalar.');
    end

    N = cache.num_rows;

    Psi = exp(-(rho.^2)*cache.S);
    M = [cache.base_features,Psi];

    [w,ridge_info] = solve_ridge(M,cache.y,lambda,ls_opts);

    if isa(w,'gpuArray')
        w = gather(w);
    end

    w = w(:);

    res = M*w-cache.y;

    residual_sq = real(res'*res);
    coefficient_sq = real(w'*w);

    objective = ...
        (residual_sq + lambda*coefficient_sq)/(2*N);

    ng = cache.num_growth_features;
    wg = w(end-ng+1:end);

    dPsi = -2*rho*cache.S.*Psi;
    derivative_action = dPsi*wg;

    grad = real(res'*derivative_action)/N;
    grad = grad(:);

    info = struct();
    info.residual_mse = residual_sq/N;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;
    info.rho = rho;
end
