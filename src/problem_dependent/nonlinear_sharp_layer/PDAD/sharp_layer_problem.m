function problem = sharp_layer_problem(cfg)
%SHARP_LAYER_PROBLEM Manufactured nonlinear sharp-layer problem.
%
%   -Delta u + u^2 = f   in Omega=(0,1)^2,
%              u = g     on boundary.
%
% Exact solution:
%
%   u(x,y) = tanh((0.5-r)/epsilon),
%   r = sqrt((x-0.5)^2+(y-0.5)^2).

    problem = struct();

    problem.domain = cfg.domain;
    problem.boundary_penalty = cfg.boundary_penalty;

    problem.Xi = make_tensor_grid( ...
        cfg.domain, ...
        cfg.interior_grid, ...
        1e-6);

    problem.fi = sharp_rhs( ...
        problem.Xi, ...
        cfg.epsilon_layer);

    nB = cfg.boundary_points_per_side;

    x = linspace(cfg.domain(1,1),cfg.domain(1,2),nB)';
    y = linspace(cfg.domain(2,1),cfg.domain(2,2),nB)';

    problem.Xb = [ ...
        cfg.domain(1,1)*ones(nB,1), y; ...
        cfg.domain(1,2)*ones(nB,1), y; ...
        x, cfg.domain(2,1)*ones(nB,1); ...
        x, cfg.domain(2,2)*ones(nB,1)];

    problem.gb = sharp_exact( ...
        problem.Xb, ...
        cfg.epsilon_layer);

    problem.Xtest = make_tensor_grid( ...
        cfg.domain, ...
        cfg.test_grid, ...
        0);

    problem.utest = sharp_exact( ...
        problem.Xtest, ...
        cfg.epsilon_layer);

    problem.grad_utest = sharp_exact_gradient( ...
        problem.Xtest, ...
        cfg.epsilon_layer);

    problem.exact = @(X) sharp_exact(X,cfg.epsilon_layer);
    problem.grad_exact = @(X) sharp_exact_gradient(X,cfg.epsilon_layer);
    problem.rhs = @(X) sharp_rhs(X,cfg.epsilon_layer);
end


function X = make_tensor_grid(domain,n,offset)

    if nargin < 3
        offset = 0;
    end

    x = linspace( ...
        domain(1,1)+offset, ...
        domain(1,2)-offset, ...
        n(1));

    y = linspace( ...
        domain(2,1)+offset, ...
        domain(2,2)-offset, ...
        n(2));

    [X1,X2] = meshgrid(x,y);

    X = [X1(:),X2(:)];
end


function u = sharp_exact(X,epsilon)

    x = X(:,1);
    y = X(:,2);

    r = sqrt((x-0.5).^2+(y-0.5).^2);

    u = tanh((0.5-r)/epsilon);
end


function grad_u = sharp_exact_gradient(X,epsilon)
%SHARP_EXACT_GRADIENT Exact spatial gradient of the manufactured solution.
%
% At the circle center r=0 the radial direction is undefined.  The exact
% solution is essentially flat there for the present epsilon, and we assign
% the limiting numerical value grad u = 0 at that single grid point.

    x = X(:,1);
    y = X(:,2);

    dx = x-0.5;
    dy = y-0.5;

    r = sqrt(dx.^2+dy.^2);

    g = (0.5-r)/epsilon;
    u = tanh(g);
    sech2 = 1-u.^2;

    grad_u = zeros(size(X));

    mask = r > 1e-12;

    grad_u(mask,1) = ...
        -sech2(mask).*dx(mask) ./ ...
        (epsilon*r(mask));

    grad_u(mask,2) = ...
        -sech2(mask).*dy(mask) ./ ...
        (epsilon*r(mask));
end


function f = sharp_rhs(X,epsilon)

    x = X(:,1);
    y = X(:,2);

    dx = x-0.5;
    dy = y-0.5;

    r = sqrt(dx.^2+dy.^2);
    r_safe = max(r,1e-12);

    g = (0.5-r_safe)/epsilon;

    u = tanh(g);
    sech2 = 1-u.^2;

    gx = -(dx./r_safe)/epsilon;
    gy = -(dy./r_safe)/epsilon;

    gxx = -(dy.^2./r_safe.^3)/epsilon;
    gyy = -(dx.^2./r_safe.^3)/epsilon;

    uxx = ...
        sech2.*gxx - ...
        2*u.*sech2.*gx.^2;

    uyy = ...
        sech2.*gyy - ...
        2*u.*sech2.*gy.^2;

    f = -uxx-uyy+u.^2;
end
