function M = black_scholes_operator_block(Xphys,p,basis,cfg)
%BLACK_SCHOLES_OPERATOR_BLOCK Apply the BS operator to hidden features.

    d = cfg.dimension;
    Xphys = double(Xphys);
    Xhat = normalize_black_scholes_points(Xphys,cfg);

    p_full = black_scholes_expand_parameter(p,d);
    S = build_preactivation(Xhat,p_full,basis);
    A = activation_derivatives(S,cfg.activation,2);

    rs = p(1);
    rt = p(2);

    Zs = basis.Z(1:d,:);
    Zt = basis.Z(d+1,:);

    Wx = (rs/cfg.normalization.x_scale)*Zs;
    Wt = rt*Zt;

    x = Xphys(:,1:d);

    drift_coefficient = (x.*cfg.mu)*Wx;
    diffusion_coefficient = ...
        0.5*(x.^2.*(cfg.sigma(:).'.^2))*(Wx.^2);

    M = ...
        A.d1.*Wt ...
        -A.d1.*drift_coefficient ...
        -A.d2.*diffusion_coefficient;
end
