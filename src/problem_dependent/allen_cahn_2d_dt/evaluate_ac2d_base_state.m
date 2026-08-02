function state = evaluate_ac2d_base_state(X,p,basis,coef,max_order)
%EVALUATE_AC2D_BASE_STATE Gaussian first-layer values and derivatives.
%
% Without coef, state contains feature matrices:
%
%   phi, dx, dy, dxx, dyy, lap
%
% With coef, it additionally contains the frozen solution state required by
% src/layer_growth:
%
%   u
%   d1{1}, d1{2}
%   d2{1}, d2{2}

    if nargin < 4
        coef = [];
    end

    if nargin < 5 || isempty(max_order)
        max_order = 2;
    end

    if size(X,2) ~= 2
        error('X must be N-by-2.');
    end

    p = p(:);

    if numel(p) ~= 2
        error('p must contain [p_x;p_y].');
    end

    derivative_order = max(2,max_order);

    S = build_preactivation(X,p,basis);
    A = activation_derivatives(S,'gaussian',derivative_order);

    Wx = p(1)*basis.Z(1,:);
    Wy = p(2)*basis.Z(2,:);

    dx = A.d1.*Wx;
    dy = A.d1.*Wy;

    dxx = A.d2.*(Wx.^2);
    dyy = A.d2.*(Wy.^2);

    state = struct();

    state.phi = A.phi;
    state.dx = dx;
    state.dy = dy;
    state.dxx = dxx;
    state.dyy = dyy;
    state.lap = dxx+dyy;

    if ~isempty(coef)

        coef = coef(:);

        state.u = state.phi*coef;

        state.d1 = { ...
            dx*coef; ...
            dy*coef};

        state.d2 = { ...
            dxx*coef; ...
            dyy*coef};
    end
end
