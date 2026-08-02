function [w, info] = solve_ridge(M, y, lambda, opts)
%SOLVE_RIDGE Ridge solve via an augmented least-squares system.

    if nargin < 4 || isempty(opts)
        opts = struct();
    end

    if isvector(y)
        y = y(:);
    end

    m = size(M,2);

    t = tic;

    if lambda > 0
        Aaug = [M; sqrt(lambda)*eye(m,'like',M)];
        baug = [y; zeros(m,size(y,2),'like',y)];
    else
        Aaug = M;
        baug = y;
    end

    build_augmented_system_time = toc(t);

    [w,ls_info] = solve_least_squares(Aaug,baug,opts);

    res = M*w-y;

    info.lambda = lambda;
    info.build_augmented_system_time = build_augmented_system_time;
    info.least_squares = ls_info;
    info.residual_norm = norm(res,'fro');
    info.coefficient_norm = norm(w,'fro');
end
