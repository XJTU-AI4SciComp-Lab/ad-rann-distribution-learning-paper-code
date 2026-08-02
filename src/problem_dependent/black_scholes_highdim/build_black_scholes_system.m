function [M,y] = build_black_scholes_system( ...
        p,split,basis,cfg,constraint_penalty,chunk_rows)
%BUILD_BLACK_SCHOLES_SYSTEM Assemble the full unregularized PDE system.

    p = p(:);
    m = size(basis.Z,2);

    Xi = double(split.interior_xt);
    Xc = double([split.boundary_xt;split.initial_xt]);

    Ni = size(Xi,1);
    Nc = size(Xc,1);

    estimated_gb = 8*(Ni+Nc)*m/1024^3;
    fprintf('Allocating final LS matrix: %d x %d, about %.3f GB.\n', ...
        Ni+Nc,m,estimated_gb);

    M = zeros(Ni+Nc,m);

    for first = 1:chunk_rows:Ni
        rows = first:min(first+chunk_rows-1,Ni);
        M(rows,:) = black_scholes_operator_block( ...
            Xi(rows,:),p,basis,cfg);
    end

    for first = 1:chunk_rows:Nc
        rows_local = first:min(first+chunk_rows-1,Nc);
        rows_global = Ni+rows_local;

        Xhat = normalize_black_scholes_points(Xc(rows_local,:),cfg);
        p_full = black_scholes_expand_parameter(p,cfg.dimension);
        S = build_preactivation(Xhat,p_full,basis);
        A = activation_derivatives(S,cfg.activation,0);
        M(rows_global,:) = constraint_penalty*A.phi;
    end

    y = [ ...
        zeros(Ni,1); ...
        constraint_penalty*double(split.boundary_values(:)); ...
        constraint_penalty*double(split.initial_values(:))];
end
