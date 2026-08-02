function [objective,grad,info] = ...
    evaluate_burgers_pdad_reduced_fast( ...
        p,Xi,Xb,basis,u_l,u_ll,dt,nu,eta,lambda,ls_opts)
%EVALUATE_BURGERS_PDAD_REDUCED_FAST PDAD objective for one BDF2 step.
%
% At a fixed physical time step k >= 2, the BDF2/extrapolated system is
%
%   3 u^k
%   + 2 dt [ (2u^{k-1}-u^{k-2}) u_x^k - nu u_xx^k ]
%   = 4u^{k-1}-u^{k-2}.
%
% The reduced ridge problem is
%
%   w_lambda(p)
%     = argmin_w ||M_k(p)w-y_k||_2^2 + lambda||w||_2^2,
%
% and
%
%   f_lambda(p)
%     = (||M_k(p)w_lambda-y_k||_2^2
%        + lambda||w_lambda||_2^2)/(2N).
%
% The envelope gradient is
%
%   df/dp = r^T (dM/dp w_lambda) / N.
%
% Boundary rows are included in the ridge training system using the same
% penalty eta as the final PDE least-squares solve.

    if nargin < 11 || isempty(ls_opts)
        ls_opts = struct();
    end

    p = p(:);

    if numel(p) ~= 1
        error('p must be scalar for this 1-D example.');
    end

    Xi = Xi(:);
    Xb = Xb(:);
    u_l = u_l(:);
    u_ll = u_ll(:);

    Ni = numel(Xi);
    Nb = numel(Xb);

    if numel(u_l) ~= Ni || numel(u_ll) ~= Ni
        error('u_l and u_ll must have one entry per interior collocation point.');
    end

    if ~isscalar(lambda) || lambda < 0 || ~isfinite(lambda)
        error('lambda must be a finite nonnegative scalar.');
    end

    % =====================================================================
    % Interior feature values and spatial derivatives
    % =====================================================================
    S = build_preactivation(Xi,p,basis);
    A = activation_derivatives(S,'gaussian',3);

    Z = basis.Z(1,:);
    C = basis.C(1,:);

    W = p(1)*Z;
    Q = (Xi-C).*Z;

    phi = A.phi;
    ux = A.d1.*W;
    uxx = A.d2.*(W.^2);

    adv = 2*u_l-u_ll;

    M_i = ...
        3*phi + ...
        2*dt*(adv.*ux - nu*uxx);

    y_i = 4*u_l-u_ll;

    % =====================================================================
    % Analytic p derivatives
    % =====================================================================
    dphi = A.d1.*Q;

    dux = ...
        A.d2.*Q.*W + ...
        A.d1.*Z;

    duxx = ...
        A.d3.*Q.*(W.^2) + ...
        A.d2.*(2*W.*Z);

    dM_i = ...
        3*dphi + ...
        2*dt*(adv.*dux - nu*duxx);

    % =====================================================================
    % Boundary block
    % =====================================================================
    S_b = build_preactivation(Xb,p,basis);
    A_b = activation_derivatives(S_b,'gaussian',1);

    Q_b = (Xb-C).*Z;

    Phi_b = A_b.phi;
    dPhi_b = A_b.d1.*Q_b;

    M_b = eta*Phi_b;
    dM_b = eta*dPhi_b;

    % Homogeneous Dirichlet data.
    y_b = zeros(Nb,1);

    M = [M_i;M_b];
    y = [y_i;eta*y_b];

    % =====================================================================
    % Ridge inner solve
    % =====================================================================
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
        (residual_sq + lambda*coefficient_sq)/(2*Ntot);

    derivative_action = [dM_i;dM_b]*w;

    grad = real(res'*derivative_action)/Ntot;
    grad = grad(:);

    info = struct();
    info.w = w;
    info.residual_mse = residual_sq/Ntot;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;
    info.num_rows = Ntot;
    info.num_features = size(M,2);
end
