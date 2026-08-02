function [objective, grad, info] = ...
    evaluate_poisson_reduced_fast(p, cache, lambda, ls_opts)
%EVALUATE_POISSON_REDUCED_FAST Fast analytic Poisson reduced evaluator.
%
% No finite-difference gradient is used.
% Full dM matrices are not stored; only (dM/dp_k)*w is formed.

    p = p(:);

    p1 = p(1);
    p2 = p(2);

    %% Interior
    Si = p1*cache.Qi1 + p2*cache.Qi2;

    Ei = exp(-Si.^2);

    Hi = (4*Si.^2-2).*Ei;
    Hip = (12*Si-8*Si.^3).*Ei;

    %% Boundary
    Sb = p1*cache.Qb1 + p2*cache.Qb2;

    Eb = exp(-Sb.^2);
    Ebp = -2*Sb.*Eb;

    %% Poisson matrix
    q = ...
        (p1^2)*cache.Z1sq + ...
        (p2^2)*cache.Z2sq;

    A = -Hi.*q;
    B = cache.eta*Eb;

    M = [A;B];

    %% Ridge least squares
    [w,ridge_info] = ...
        solve_ridge(M,cache.y,lambda,ls_opts);

    if isa(w,'gpuArray')
        w = gather(w);
    end

    w = w(:);

    %% Objective
    res = M*w-cache.y;

    N = cache.num_rows;

    residual_sq = real(res'*res);
    coefficient_sq = real(w'*w);

    objective = ...
        (residual_sq + lambda*coefficient_sq)/(2*N);

    %% Analytic derivative actions
    dq1 = 2*p1*cache.Z1sq;
    dq2 = 2*p2*cache.Z2sq;

    qw = q(:).*w;

    vA1 = ...
        -Hi*(dq1(:).*w) ...
        -(Hip.*cache.Qi1)*qw;

    vA2 = ...
        -Hi*(dq2(:).*w) ...
        -(Hip.*cache.Qi2)*qw;

    vB1 = ...
        cache.eta*((Ebp.*cache.Qb1)*w);

    vB2 = ...
        cache.eta*((Ebp.*cache.Qb2)*w);

    v1 = [vA1;vB1];
    v2 = [vA2;vB2];

    grad = [ ...
        real(res'*v1)/N; ...
        real(res'*v2)/N];

    %% Info
    info.w = w;
    info.residual_mse = residual_sq/N;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;

    if isfield(ls_opts,'compute_spectrum') && ls_opts.compute_spectrum

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
