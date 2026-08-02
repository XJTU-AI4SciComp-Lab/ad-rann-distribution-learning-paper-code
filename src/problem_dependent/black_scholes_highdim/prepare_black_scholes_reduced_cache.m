function cache = prepare_black_scholes_reduced_cache( ...
        split,basis,cfg,constraint_penalty)
%PREPARE_BLACK_SCHOLES_REDUCED_CACHE p-independent training quantities.

    d = cfg.dimension;
    m = size(basis.Z,2);

    Xi_phys = double(split.interior_xt);
    Xc_phys = double([split.boundary_xt;split.initial_xt]);

    Xi_hat = normalize_black_scholes_points(Xi_phys,cfg);
    Xc_hat = normalize_black_scholes_points(Xc_phys,cfg);

    Zs = basis.Z(1:d,:);
    Cs = basis.C(1:d,:);
    Zt = basis.Z(d+1,:);
    Ct = basis.C(d+1,:);

    cache.Qs_i = Xi_hat(:,1:d)*Zs-sum(Cs.*Zs,1);
    cache.Qt_i = Xi_hat(:,end)*Zt-Ct.*Zt;

    cache.Qs_c = Xc_hat(:,1:d)*Zs-sum(Cs.*Zs,1);
    cache.Qt_c = Xc_hat(:,end)*Zt-Ct.*Zt;

    x = Xi_phys(:,1:d);
    mu_x = x.*cfg.mu;
    sigma2_x2 = x.^2.*(cfg.sigma(:).'.^2);

    cache.D0 = (mu_x*Zs)/cfg.normalization.x_scale;
    cache.E0 = 0.5*(sigma2_x2*(Zs.^2)) / ...
        (cfg.normalization.x_scale^2);

    cache.Zt = Zt;
    cache.constraint_penalty = constraint_penalty;

    yi = zeros(size(Xi_phys,1),1);
    yc = constraint_penalty*double([ ...
        split.boundary_values(:);split.initial_values(:)]);

    cache.y = [yi;yc];
    cache.num_interior = size(Xi_phys,1);
    cache.num_constraints = size(Xc_phys,1);
    cache.num_features = m;
end
