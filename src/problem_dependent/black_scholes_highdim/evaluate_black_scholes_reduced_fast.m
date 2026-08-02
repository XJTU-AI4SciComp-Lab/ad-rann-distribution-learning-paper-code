function [objective,grad,info] = evaluate_black_scholes_reduced_fast( ...
        p,cache,lambda,ls_opts,activation)
%EVALUATE_BLACK_SCHOLES_REDUCED_FAST Two-parameter PDAD objective.
%
% The returned gradient is always with respect to physical
% p=[r_s;r_t].  When p=exp(s) is enabled, the common Adam routine applies
% the chain rule grad_s=grad_p.*p.

    p = p(:);

    if numel(p) ~= 2 || any(p <= 0)
        error('p must be the positive vector [r_s;r_t].');
    end

    rs = p(1);
    rt = p(2);

    S_i = rs*cache.Qs_i+rt*cache.Qt_i;
    A_i = activation_derivatives(S_i,activation,3);

    Wt = rt*cache.Zt;
    D = rs*cache.D0;
    E = rs^2*cache.E0;

    M_i = A_i.d1.*(Wt-D)-A_i.d2.*E;

    S_c = rs*cache.Qs_c+rt*cache.Qt_c;
    A_c = activation_derivatives(S_c,activation,1);
    M_c = cache.constraint_penalty*A_c.phi;

    M = [M_i;M_c];
    y = cache.y;

    [w,ridge_info] = solve_ridge(M,y,lambda,ls_opts);

    if isa(w,'gpuArray')
        w = gather(w);
    end
    w = w(:);

    res = M*w-y;
    Ntot = size(M,1);

    residual_sq = real(res'*res);
    coefficient_sq = real(w'*w);

    objective = ...
        (residual_sq+lambda*coefficient_sq)/(2*Ntot);

    % d/dr_s
    dM_i_rs = ...
        A_i.d2.*cache.Qs_i.*(Wt-D) ...
        -A_i.d1.*cache.D0 ...
        -A_i.d3.*cache.Qs_i.*E ...
        -A_i.d2.*(2*rs*cache.E0);

    dM_c_rs = ...
        cache.constraint_penalty*A_c.d1.*cache.Qs_c;

    % d/dr_t
    dM_i_rt = ...
        A_i.d2.*cache.Qt_i.*(Wt-D) ...
        +A_i.d1.*cache.Zt ...
        -A_i.d3.*cache.Qt_i.*E;

    dM_c_rt = ...
        cache.constraint_penalty*A_c.d1.*cache.Qt_c;

    action_rs = [dM_i_rs;dM_c_rs]*w;
    action_rt = [dM_i_rt;dM_c_rt]*w;

    grad = [ ...
        real(res'*action_rs)/Ntot; ...
        real(res'*action_rt)/Ntot];

    info = struct();
    info.w = w;
    info.residual_mse = residual_sq/Ntot;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;
    info.num_rows = Ntot;
    info.num_features = size(M,2);
end
