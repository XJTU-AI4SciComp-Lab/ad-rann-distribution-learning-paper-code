function op = prepare_burgers2d_base_operator(p,problem,basis)
%PREPARE_BURGERS2D_BASE_OPERATOR Fixed feature/operator matrices at p.
%
% The assembly is chunked and stores only the matrices used by the physical
% Burgers solve: Phi, d_x Phi+d_y Phi, and Delta Phi.

    p = p(:);

    if numel(p) ~= 2
        error('p must contain [p_x;p_y].');
    end

    Xi = problem.Xi;
    Xb = problem.Xb;

    Ni = size(Xi,1);
    Nb = size(Xb,1);
    m = size(basis.Z,2);

    chunk_rows = 500;

    Wx = p(1)*basis.Z(1,:);
    Wy = p(2)*basis.Z(2,:);

    Phi_i = zeros(Ni,m);
    Dsum_i = zeros(Ni,m);
    Lap_i = zeros(Ni,m);

    for first = 1:chunk_rows:Ni

        last = min(first+chunk_rows-1,Ni);
        rows = first:last;

        S = build_preactivation(Xi(rows,:),p,basis);
        A = activation_derivatives(S,'gaussian',2);

        Phi_i(rows,:) = A.phi;
        Dsum_i(rows,:) = A.d1.*(Wx+Wy);
        Lap_i(rows,:) = A.d2.*(Wx.^2+Wy.^2);
    end

    Phi_b = zeros(Nb,m);

    for first = 1:chunk_rows:Nb

        last = min(first+chunk_rows-1,Nb);
        rows = first:last;

        S = build_preactivation(Xb(rows,:),p,basis);
        A = activation_derivatives(S,'gaussian',2);
        Phi_b(rows,:) = A.phi;
    end

    op = struct();

    op.p = p;
    op.Phi_i = Phi_i;
    op.Dsum_i = Dsum_i;
    op.Lap_i = Lap_i;
    op.Phi_b = Phi_b;
end
