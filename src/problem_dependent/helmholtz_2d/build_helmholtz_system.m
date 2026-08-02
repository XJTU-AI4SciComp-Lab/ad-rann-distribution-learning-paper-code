function [M,y,dM] = build_helmholtz_system( ...
        p,problem,basis,need_derivatives,chunk_rows)
%BUILD_HELMHOLTZ_SYSTEM Assemble Delta u + k^2 u = q and Dirichlet rows.
%
%   [M,y] = build_helmholtz_system(p,problem,basis,false,chunk_rows)
%   [M,y,dM] = build_helmholtz_system(p,problem,basis,true,chunk_rows)
%
% The implementation reuses common build_preactivation and
% activation_derivatives. Row chunking limits temporary memory.

    if nargin < 4 || isempty(need_derivatives)
        need_derivatives = (nargout >= 3);
    end

    if nargin < 5 || isempty(chunk_rows) || chunk_rows <= 0
        chunk_rows = Inf;
    end

    p = p(:);

    if numel(p) ~= 2
        error('p must contain [p_x;p_y].');
    end

    m = size(basis.Z,2);
    Ni = size(problem.Xi,1);
    Nb = size(problem.Xb,1);
    Ntot = Ni+Nb;

    M = zeros(Ntot,m);

    if need_derivatives
        dM1 = zeros(Ntot,m);
        dM2 = zeros(Ntot,m);
    end

    Z1 = basis.Z(1,:);
    Z2 = basis.Z(2,:);
    C1 = basis.C(1,:);
    C2 = basis.C(2,:);

    W1 = p(1)*Z1;
    W2 = p(2)*Z2;

    q = W1.^2+W2.^2;
    dq1 = 2*W1.*Z1;
    dq2 = 2*W2.*Z2;

    if isinf(chunk_rows)
        chunk_rows = max(Ni,Nb);
    end

    %% Interior rows

    for first = 1:chunk_rows:Ni

        last = min(first+chunk_rows-1,Ni);
        rows = first:last;
        X = problem.Xi(rows,:);

        S = build_preactivation(X,p,basis);

        if need_derivatives
            A = activation_derivatives(S,'gaussian',3);
        else
            A = activation_derivatives(S,'gaussian',2);
        end

        M(rows,:) = A.d2.*q + problem.k^2*A.phi;

        if need_derivatives

            Q1 = (X(:,1)-C1).*Z1;
            Q2 = (X(:,2)-C2).*Z2;

            dM1(rows,:) = ...
                A.d3.*Q1.*q + ...
                A.d2.*dq1 + ...
                problem.k^2*A.d1.*Q1;

            dM2(rows,:) = ...
                A.d3.*Q2.*q + ...
                A.d2.*dq2 + ...
                problem.k^2*A.d1.*Q2;
        end
    end

    %% Boundary rows

    eta = problem.boundary_penalty;

    for first = 1:chunk_rows:Nb

        last = min(first+chunk_rows-1,Nb);
        local_rows = first:last;
        global_rows = Ni+local_rows;
        X = problem.Xb(local_rows,:);

        S = build_preactivation(X,p,basis);
        A = activation_derivatives(S,'gaussian',1);

        M(global_rows,:) = eta*A.phi;

        if need_derivatives

            Q1 = (X(:,1)-C1).*Z1;
            Q2 = (X(:,2)-C2).*Z2;

            dM1(global_rows,:) = eta*A.d1.*Q1;
            dM2(global_rows,:) = eta*A.d1.*Q2;
        end
    end

    y = problem.y;

    if need_derivatives
        dM = {dM1,dM2};
    else
        dM = {};
    end
end
