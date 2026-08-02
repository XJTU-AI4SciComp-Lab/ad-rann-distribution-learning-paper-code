function u = poisson_lshape_exact_solution(X)
%POISSON_LSHAPE_EXACT_SOLUTION Re-entrant-corner singular function.
%
% u(r,theta) = r^(2/3) sin(2 theta / 3),
% with theta in [0,3*pi/2] on the L-shaped domain.

    if size(X,2) ~= 2
        error('X must be N-by-2.');
    end

    x = X(:,1);
    y = X(:,2);

    r = hypot(x,y);
    theta = atan2(y,x);
    theta(theta < 0) = theta(theta < 0) + 2*pi;

    alpha = 2/3;
    u = r.^alpha .* sin(alpha*theta);
    u(r == 0) = 0;
end
