function [objective,grad,info] = ...
    evaluate_ac2d_pdad_reduced_fast( ...
        p,Xi,Xb,basis,rhs_i,a0,diffusion_weight, ...
        eta,lambda,ls_opts)
%EVALUATE_AC2D_PDAD_REDUCED_FAST
%
% Reduced PDAD objective for a current linear IMEX time step:
%
%   (a0*Phi - diffusion_weight*DeltaPhi) w = rhs_i
%
% with homogeneous Dirichlet boundary rows eta*Phi_b*w=0.

    if nargin < 10 || isempty(ls_opts)
        ls_opts = struct();
    end

    p = p(:);
    rhs_i = rhs_i(:);

    if numel(p) ~= 2
        error('p must contain two values.');
    end

    if size(Xi,2) ~= 2 || size(Xb,2) ~= 2
        error('Xi and Xb must be N-by-2 arrays.');
    end

    if numel(rhs_i) ~= size(Xi,1)
        error('rhs_i length must equal size(Xi,1).');
    end

    % =====================================================================
    % Interior feature values and derivatives
    % =====================================================================
    S = build_preactivation(Xi,p,basis);
    A = activation_derivatives(S,'gaussian',3);

    Zx = basis.Z(1,:);
    Zy = basis.Z(2,:);

    Cx = basis.C(1,:);
    Cy = basis.C(2,:);

    Wx = p(1)*Zx;
    Wy = p(2)*Zy;

    Qx = (Xi(:,1)-Cx).*Zx;
    Qy = (Xi(:,2)-Cy).*Zy;

    Wsq = Wx.^2+Wy.^2;

    Phi = A.phi;
    Lap = A.d2.*Wsq;

    M_i = ...
        a0*Phi ...
        -diffusion_weight*Lap;

    dPhi_x = A.d1.*Qx;
    dPhi_y = A.d1.*Qy;

    dLap_x = ...
        A.d3.*Qx.*Wsq + ...
        A.d2.*(2*Wx.*Zx);

    dLap_y = ...
        A.d3.*Qy.*Wsq + ...
        A.d2.*(2*Wy.*Zy);

    dM_x = ...
        a0*dPhi_x ...
        -diffusion_weight*dLap_x;

    dM_y = ...
        a0*dPhi_y ...
        -diffusion_weight*dLap_y;

    % =====================================================================
    % Homogeneous Dirichlet boundary values and derivatives
    % =====================================================================
    S_b = build_preactivation(Xb,p,basis);
    A_b = activation_derivatives(S_b,'gaussian',1);

    Qbx = (Xb(:,1)-Cx).*Zx;
    Qby = (Xb(:,2)-Cy).*Zy;

    M_b = eta*A_b.phi;

    dMb_x = eta*(A_b.d1.*Qbx);
    dMb_y = eta*(A_b.d1.*Qby);

    M = [M_i;M_b];
    y = [rhs_i;zeros(size(Xb,1),1)];

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

    grad = zeros(2,1);

    grad(1) = ...
        real(res'*([dM_x;dMb_x]*w))/Ntot;

    grad(2) = ...
        real(res'*([dM_y;dMb_y]*w))/Ntot;

    info = struct();

    info.w = w;
    info.residual_mse = residual_sq/Ntot;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;

    info.num_rows = Ntot;
    info.num_features = size(M,2);
end
