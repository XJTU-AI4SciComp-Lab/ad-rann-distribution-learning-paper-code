function [objective, grad, info] = reduced_objective_gradient(M, y, dM, lambda, opts)
%REDUCED_OBJECTIVE_GRADIENT Ridge-reduced objective and analytic gradient.

    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    y = y(:);

    [w,ridge_info] = solve_ridge(M,y,lambda,opts);

    if isa(w,'gpuArray')
        w = gather(w);
    end

    w = w(:);

    res = M*w-y;
    N = size(M,1);

    residual_sq = real(res'*res);
    coefficient_sq = real(w'*w);

    objective = ...
        (residual_sq + lambda*coefficient_sq)/(2*N);

    grad = zeros(numel(dM),1);

    for k = 1:numel(dM)
        direction = dM{k}*w;
        grad(k) = real(res'*direction)/N;
    end

    info.w = w;
    info.residual = res;
    info.residual_mse = residual_sq/N;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;

    if isfield(opts,'compute_spectrum') && opts.compute_spectrum

        s = svd(M,'econ');

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
