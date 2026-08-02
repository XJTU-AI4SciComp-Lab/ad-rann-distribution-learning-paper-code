function [objective,grad,info] = ...
    evaluate_ac_pdad_reduced_fast( ...
        p,Xi,Xb,basis,u_l,u_ll,dt,epsilon_ac,alpha,eta,lambda,ls_opts)
%EVALUATE_AC_PDAD_REDUCED_FAST PDAD reduced objective for Allen-Cahn.
%
% BDF2/IMEX system:
%
%   3u^k - 2dt epsilon u_xx^k
%     = 4u^{k-1}-u^{k-2}
%       -2dt alpha [2f(u^{k-1})-f(u^{k-2})],
%
%   f(u)=u^3-u.
%
% Periodic value and derivative boundary rows are included in the ridge
% training system.

    if nargin < 12 || isempty(ls_opts)
        ls_opts = struct();
    end

    p = p(:);

    if numel(p) ~= 1
        error('p must be scalar.');
    end

    Xi = Xi(:);
    Xb = Xb(:);
    u_l = u_l(:);
    u_ll = u_ll(:);

    % =====================================================================
    % Interior features
    % =====================================================================
    S = build_preactivation(Xi,p,basis);
    A = activation_derivatives(S,'gaussian',3);

    Z = basis.Z(1,:);
    C = basis.C(1,:);

    W = p(1)*Z;
    Q = (Xi-C).*Z;

    phi = A.phi;
    uxx = A.d2.*(W.^2);

    M_i = 3*phi - 2*dt*epsilon_ac*uxx;

    f_l = u_l.^3-u_l;
    f_ll = u_ll.^3-u_ll;

    y_i = ...
        4*u_l-u_ll ...
        -2*dt*alpha*(2*f_l-f_ll);

    % =====================================================================
    % p derivatives
    % =====================================================================
    dphi = A.d1.*Q;

    duxx = ...
        A.d3.*Q.*(W.^2) + ...
        A.d2.*(2*W.*Z);

    dM_i = ...
        3*dphi ...
        -2*dt*epsilon_ac*duxx;

    % =====================================================================
    % Periodic boundary rows and derivatives
    % =====================================================================
    S_b = build_preactivation(Xb,p,basis);
    A_b = activation_derivatives(S_b,'gaussian',2);

    Q_b = (Xb-C).*Z;

    phi_b = A_b.phi;
    ux_b = A_b.d1.*W;

    dphi_b = A_b.d1.*Q_b;

    dux_b = ...
        A_b.d2.*Q_b.*W + ...
        A_b.d1.*Z;

    M_b = eta*[ ...
        phi_b(1,:)-phi_b(2,:); ...
        ux_b(1,:)-ux_b(2,:)];

    dM_b = eta*[ ...
        dphi_b(1,:)-dphi_b(2,:); ...
        dux_b(1,:)-dux_b(2,:)];

    y_b = zeros(2,1);

    M = [M_i;M_b];
    y = [y_i;y_b];

    % =====================================================================
    % Ridge inner problem
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
