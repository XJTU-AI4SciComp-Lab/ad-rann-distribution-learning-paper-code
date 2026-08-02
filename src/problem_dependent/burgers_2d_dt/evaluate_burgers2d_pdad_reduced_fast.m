function [objective,grad,info] = ...
    evaluate_burgers2d_pdad_reduced_fast( ...
        p,Xi,Xb,basis,qhat,rhs_i,boundary_values, ...
        a0,convection_weight,diffusion_weight, ...
        eta,lambda,ls_opts)
%EVALUATE_BURGERS2D_PDAD_REDUCED_FAST Chunked PDAD reduced objective.
%
% The frozen-coefficient spatial system is
%
%   a0*u + convection_weight*qhat*(u_x+u_y)
%        - diffusion_weight*Delta u = rhs_i.
%
% The full system matrix M is assembled once.  After the ridge solve, the
% envelope-gradient actions (dM/dp_j)w are evaluated by chunks, avoiding
% storage of full derivative matrices for p_x and p_y.

    if nargin < 13 || isempty(ls_opts)
        ls_opts = struct();
    end

    p = p(:);
    qhat = qhat(:);
    rhs_i = rhs_i(:);
    boundary_values = boundary_values(:);

    if numel(p) ~= 2
        error('p must contain [p_x;p_y].');
    end

    if size(Xi,2) ~= 2 || size(Xb,2) ~= 2
        error('Xi and Xb must be N-by-2 arrays.');
    end

    Ni = size(Xi,1);
    Nb = size(Xb,1);
    m = size(basis.Z,2);

    if numel(qhat) ~= Ni || numel(rhs_i) ~= Ni
        error('qhat and rhs_i must match the interior point count.');
    end

    if numel(boundary_values) ~= Nb
        error('boundary_values must match the boundary point count.');
    end

    if isfield(ls_opts,'feature_chunk_rows') && ...
       ~isempty(ls_opts.feature_chunk_rows)
        chunk_rows = max(1,floor(ls_opts.feature_chunk_rows));
    else
        chunk_rows = 500;
    end

    Zx = basis.Z(1,:);
    Zy = basis.Z(2,:);
    Cx = basis.C(1,:);
    Cy = basis.C(2,:);

    Wx = p(1)*Zx;
    Wy = p(2)*Zy;

    Wsum = Wx+Wy;
    Wsq = Wx.^2+Wy.^2;

    % =====================================================================
    % Assemble the ridge system without derivative matrices
    % =====================================================================
    M = zeros(Ni+Nb,m);

    for first = 1:chunk_rows:Ni

        last = min(first+chunk_rows-1,Ni);
        rows = first:last;

        S = build_preactivation(Xi(rows,:),p,basis);
        A = activation_derivatives(S,'gaussian',3);

        Phi = A.phi;
        Dsum = A.d1.*Wsum;
        Lap = A.d2.*Wsq;

        M(rows,:) = ...
            a0*Phi + ...
            convection_weight*(qhat(rows).*Dsum) - ...
            diffusion_weight*Lap;
    end

    boundary_rows = Ni+(1:Nb);

    for first = 1:chunk_rows:Nb

        last = min(first+chunk_rows-1,Nb);
        local_rows = first:last;
        rows = Ni+local_rows;

        S_b = build_preactivation(Xb(local_rows,:),p,basis);
        A_b = activation_derivatives(S_b,'gaussian',1);

        M(rows,:) = eta*A_b.phi;
    end

    y = [rhs_i;eta*boundary_values];

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

    % =====================================================================
    % Chunked envelope-gradient actions
    % =====================================================================
    grad_numerator = zeros(2,1);

    for first = 1:chunk_rows:Ni

        last = min(first+chunk_rows-1,Ni);
        rows = first:last;

        Xchunk = Xi(rows,:);

        S = build_preactivation(Xchunk,p,basis);
        A = activation_derivatives(S,'gaussian',3);

        Qx = (Xchunk(:,1)-Cx).*Zx;
        Qy = (Xchunk(:,2)-Cy).*Zy;

        dPhi_x = A.d1.*Qx;
        dDsum_x = A.d2.*Qx.*Wsum + A.d1.*Zx;
        dLap_x = A.d3.*Qx.*Wsq + A.d2.*(2*Wx.*Zx);

        action_x = ...
            a0*(dPhi_x*w) + ...
            convection_weight*( ...
                qhat(rows).*(dDsum_x*w)) - ...
            diffusion_weight*(dLap_x*w);

        clear dPhi_x dDsum_x dLap_x;

        dPhi_y = A.d1.*Qy;
        dDsum_y = A.d2.*Qy.*Wsum + A.d1.*Zy;
        dLap_y = A.d3.*Qy.*Wsq + A.d2.*(2*Wy.*Zy);

        action_y = ...
            a0*(dPhi_y*w) + ...
            convection_weight*( ...
                qhat(rows).*(dDsum_y*w)) - ...
            diffusion_weight*(dLap_y*w);

        grad_numerator(1) = ...
            grad_numerator(1)+real(res(rows)'*action_x);
        grad_numerator(2) = ...
            grad_numerator(2)+real(res(rows)'*action_y);
    end

    for first = 1:chunk_rows:Nb

        last = min(first+chunk_rows-1,Nb);
        local_rows = first:last;
        rows = boundary_rows(local_rows);

        Xchunk = Xb(local_rows,:);

        S_b = build_preactivation(Xchunk,p,basis);
        A_b = activation_derivatives(S_b,'gaussian',1);

        Qbx = (Xchunk(:,1)-Cx).*Zx;
        Qby = (Xchunk(:,2)-Cy).*Zy;

        action_x = eta*((A_b.d1.*Qbx)*w);
        action_y = eta*((A_b.d1.*Qby)*w);

        grad_numerator(1) = ...
            grad_numerator(1)+real(res(rows)'*action_x);
        grad_numerator(2) = ...
            grad_numerator(2)+real(res(rows)'*action_y);
    end

    grad = grad_numerator/Ntot;

    info = struct();

    info.w = w;
    info.residual_mse = residual_sq/Ntot;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.ridge = ridge_info;
    info.num_rows = Ntot;
    info.num_features = m;
    info.feature_chunk_rows = chunk_rows;
end
