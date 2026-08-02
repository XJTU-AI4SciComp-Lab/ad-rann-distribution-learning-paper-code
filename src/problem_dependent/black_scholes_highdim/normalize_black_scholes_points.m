function xhat = normalize_black_scholes_points(xt,cfg)
%NORMALIZE_BLACK_SCHOLES_POINTS Map x in [90,110] to [0,1]; keep t.

    d = cfg.dimension;

    if size(xt,2) ~= d+1
        error('xt must have d+1 columns.');
    end

    xhat = double(xt);
    xhat(:,1:d) = ...
        (xhat(:,1:d)-cfg.normalization.x_shift) / ...
        cfg.normalization.x_scale;
end
