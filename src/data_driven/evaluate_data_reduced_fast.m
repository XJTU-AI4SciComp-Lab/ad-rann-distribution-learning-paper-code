function [objective,grad,info] = ...
    evaluate_data_reduced_fast(p,cache,lambda,ls_opts,activation)
%EVALUATE_DATA_REDUCED_FAST Fast dimension-independent DDAD evaluator.
%
% Data fitting problem:
%
%   min_w ||Phi(p)w-y||_2^2 + lambda||w||_2^2,
%
% where
%
%   S(p)   = sum_k p(k)Q{k},
%   Phi(p) = sigma(S(p)).
%
% The reduced gradient uses
%
%   dPhi/dp_k = sigma'(S).*Q{k},
%
% but only (dPhi/dp_k)*w is formed; full derivative matrices are not
% stored. The implementation is independent of the input dimension.

    if nargin < 4 || isempty(ls_opts)
        ls_opts = struct();
    end

    if nargin < 5 || isempty(activation)
        activation = 'gaussian';
    end

    p = p(:);
    d = cache.dim;

    if numel(p) ~= d
        error('numel(p) must equal cache.dim.');
    end

    if any(~isfinite(p))
        error('p contains NaN or Inf.');
    end

    if ~isscalar(lambda) || ~isfinite(lambda) || lambda < 0
        error('lambda must be a finite nonnegative scalar.');
    end

    N = cache.num_rows;
    m = cache.num_features;

    S = zeros(N,m,'like',cache.Q{1});

    for k = 1:d
        S = S + p(k)*cache.Q{k};
    end

    A = activation_derivatives(S,activation,1);
    Phi = A.phi;

    [w,ridge_info] = ...
        solve_ridge(Phi,cache.y,lambda,ls_opts);

    if isa(w,'gpuArray')
        w = gather(w);
    end

    w = w(:);

    res = Phi*w-cache.y;

    residual_sq = real(res'*res);
    coefficient_sq = real(w'*w);

    objective = ...
        (residual_sq + lambda*coefficient_sq)/(2*N);

    grad = zeros(d,1);

    for k = 1:d
        derivative_action = ...
            (A.d1.*cache.Q{k})*w;

        grad(k) = real(res'*derivative_action)/N;
    end

    info.w = w;
    info.residual_mse = residual_sq/N;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;
    info.activation = lower(char(activation));
    info.dim = d;
    info.num_rows = N;
    info.num_features = m;

    if isfield(ls_opts,'compute_spectrum') && ls_opts.compute_spectrum

        s = svd(Phi,'econ');

        info.sigma_max = max(s);
        info.sigma_min = min(s);

        if lambda > 0
            info.ridge_condition_number = ...
                (info.sigma_max^2+lambda) / ...
                (info.sigma_min^2+lambda);
        else
            info.ridge_condition_number = Inf;
        end
    end
end
