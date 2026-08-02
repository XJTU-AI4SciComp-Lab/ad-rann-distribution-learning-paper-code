function [objective,grad,info] = ...
    evaluate_poisson_lshape_mixture_reduced_fast( ...
        p,cache,lambda,ls_opts)
%EVALUATE_POISSON_LSHAPE_MIXTURE_REDUCED_FAST Exact four-parameter gradient.
%
% The output coefficients are eliminated by ridge least squares.  Only
% (dM/dp_k)*w is formed, so four full derivative matrices are never stored.

    p = p(:);
    if numel(p) ~= 4
        error('Mixture parameter p must have four components.');
    end

    [Mnear,state_near] = build_block(p(1:2),cache.near);
    [Mfar,state_far] = build_block(p(3:4),cache.far);
    M = [Mnear,Mfar];

    [w,ridge_info] = solve_ridge(M,cache.y,lambda,ls_opts);
    if isa(w,'gpuArray')
        w = gather(w);
    end
    w = w(:);

    wnear = w(1:cache.num_near);
    wfar = w(cache.num_near+1:end);
    residual = M*w-cache.y;
    N = cache.num_rows;

    residual_sq = real(residual'*residual);
    coefficient_sq = real(w'*w);
    objective = (residual_sq+lambda*coefficient_sq)/(2*N);

    v1 = derivative_action(p(1:2),cache.near,state_near,wnear,1);
    v2 = derivative_action(p(1:2),cache.near,state_near,wnear,2);
    v3 = derivative_action(p(3:4),cache.far,state_far,wfar,1);
    v4 = derivative_action(p(3:4),cache.far,state_far,wfar,2);

    grad = real([ ...
        residual'*v1; residual'*v2; ...
        residual'*v3; residual'*v4])/N;

    info.w = w;
    info.residual_mse = residual_sq/N;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;
end


function [M,state] = build_block(p,cache)
    Si = p(1)*cache.Qi1+p(2)*cache.Qi2;
    Ei = exp(-Si.^2);
    Hi = (4*Si.^2-2).*Ei;
    Hip = (12*Si-8*Si.^3).*Ei;

    Sb = p(1)*cache.Qb1+p(2)*cache.Qb2;
    Eb = exp(-Sb.^2);
    Ebp = -2*Sb.*Eb;

    q = p(1)^2*cache.Z1sq+p(2)^2*cache.Z2sq;
    M = [-Hi.*q; cache.eta*Eb];

    state.Hi = Hi;
    state.Hip = Hip;
    state.Ebp = Ebp;
    state.q = q;
end


function v = derivative_action(p,cache,state,w,k)
    if k == 1
        Q = cache.Qi1;
        Qb = cache.Qb1;
        dq = 2*p(1)*cache.Z1sq;
    else
        Q = cache.Qi2;
        Qb = cache.Qb2;
        dq = 2*p(2)*cache.Z2sq;
    end

    qw = state.q(:).*w;
    vi = -state.Hi*(dq(:).*w)-(state.Hip.*Q)*qw;
    vb = cache.eta*((state.Ebp.*Qb)*w);
    v = [vi;vb];
end
