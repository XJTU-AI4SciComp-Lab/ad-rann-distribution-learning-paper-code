function value = evaluate_black_scholes_network( ...
        xt,p,basis,coef,cfg,chunk_rows)
%EVALUATE_BLACK_SCHOLES_NETWORK Evaluate the fitted network in chunks.

    xt = double(xt);
    coef = coef(:);
    N = size(xt,1);
    value = zeros(N,1);

    p_full = black_scholes_expand_parameter(p,cfg.dimension);

    for first = 1:chunk_rows:N
        rows = first:min(first+chunk_rows-1,N);
        Xhat = normalize_black_scholes_points(xt(rows,:),cfg);
        S = build_preactivation(Xhat,p_full,basis);
        A = activation_derivatives(S,cfg.activation,0);
        value(rows) = A.phi*coef;
    end
end
