function Phi = gaussian_features(X, p, basis)
%GAUSSIAN_FEATURES Evaluate exp(-s^2) randomized Gaussian features.

    p = p(:);

    W = basis.Z .* p;
    b = -sum(basis.C .* W,1);

    S = X*W + b;

    Phi = exp(-S.^2);
end
