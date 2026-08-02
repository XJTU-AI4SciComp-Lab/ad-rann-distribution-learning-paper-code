function [ux,uy] = poisson_lshape_exact_gradient(X)
%POISSON_LSHAPE_EXACT_GRADIENT Gradient of the corner singular function.
%
% For alpha=2/3,
%   ux = alpha*r^(alpha-1)*sin((alpha-1)*theta),
%   uy = alpha*r^(alpha-1)*cos((alpha-1)*theta).
%
% The gradient is singular at r=0. NaN is returned at that one point.

    if size(X,2) ~= 2
        error('X must be N-by-2.');
    end

    x = X(:,1);
    y = X(:,2);

    r = hypot(x,y);
    theta = atan2(y,x);
    theta(theta < 0) = theta(theta < 0) + 2*pi;

    alpha = 2/3;
    factor = alpha*r.^(alpha-1);

    ux = factor.*sin((alpha-1)*theta);
    uy = factor.*cos((alpha-1)*theta);

    ux(r == 0) = NaN;
    uy(r == 0) = NaN;
end
