function D = feature_derivatives_2d(X,p,basis,activation)
%FEATURE_DERIVATIVES_2D Generic 2-D feature/spatial/p-derivatives.
%
%   D = feature_derivatives_2d(X,p,basis,activation)
%
% phi(x,y;p) = sigma(S),
% S = p1*Q1 + p2*Q2,
% Q1 = (x-C1).*Z1,
% Q2 = (y-C2).*Z2.
%
% The activation-specific formulas are supplied by activation_derivatives.
% This routine supports gaussian, tanh, and sin without duplicating the
% spatial/parameter chain-rule formulas.

    p = p(:);

    if numel(p) ~= 2
        error('feature_derivatives_2d requires p with exactly two components.');
    end

    if size(X,2) ~= 2
        error('X must be N-by-2.');
    end

    if size(basis.Z,1) ~= 2 || ~isequal(size(basis.Z),size(basis.C))
        error('basis.Z and basis.C must both be 2-by-m.');
    end

    Z1 = basis.Z(1,:);
    Z2 = basis.Z(2,:);

    C1 = basis.C(1,:);
    C2 = basis.C(2,:);

    p1 = p(1);
    p2 = p(2);

    W1 = p1*Z1;
    W2 = p2*Z2;

    Q1 = (X(:,1)-C1).*Z1;
    Q2 = (X(:,2)-C2).*Z2;

    S = p1*Q1 + p2*Q2;

    A = activation_derivatives(S,activation,3);

    d1 = A.d1;
    d2 = A.d2;
    d3 = A.d3;

    D.phi = A.phi;

    D.x = d1.*W1;
    D.y = d1.*W2;

    D.xx = d2.*(W1.^2);
    D.xy = d2.*(W1.*W2);
    D.yy = d2.*(W2.^2);

    D.dp = cell(2,1);

    % p1 derivatives
    D.dp{1}.phi = d1.*Q1;
    D.dp{1}.x = d2.*Q1.*W1 + d1.*Z1;
    D.dp{1}.y = d2.*Q1.*W2;
    D.dp{1}.xx = d2.*(2*W1.*Z1) + d3.*Q1.*(W1.^2);
    D.dp{1}.xy = d2.*(Z1.*W2) + d3.*Q1.*(W1.*W2);
    D.dp{1}.yy = d3.*Q1.*(W2.^2);

    % p2 derivatives
    D.dp{2}.phi = d1.*Q2;
    D.dp{2}.x = d2.*Q2.*W1;
    D.dp{2}.y = d2.*Q2.*W2 + d1.*Z2;
    D.dp{2}.xx = d3.*Q2.*(W1.^2);
    D.dp{2}.xy = d2.*(W1.*Z2) + d3.*Q2.*(W1.*W2);
    D.dp{2}.yy = d2.*(2*W2.*Z2) + d3.*Q2.*(W2.^2);

    D.S = S;
end
