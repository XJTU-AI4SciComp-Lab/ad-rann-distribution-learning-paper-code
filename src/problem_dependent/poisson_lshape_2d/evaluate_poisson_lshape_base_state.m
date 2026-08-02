function state = evaluate_poisson_lshape_base_state( ...
        X,p,basis,coef,max_order)
%EVALUATE_POISSON_LSHAPE_BASE_STATE First-layer values and derivatives.
%
% With output coefficients, this also returns the frozen base solution
% state required by the generic src/layer_growth implementation:
%
%   state.u
%   state.d1{1}, state.d1{2}
%   state.d2{1}, state.d2{2}

    if nargin < 4
        coef = [];
    end

    if nargin < 5 || isempty(max_order)
        max_order = 2;
    end

    if size(X,2) ~= 2
        error('X must be N-by-2.');
    end

    if ~ismember(max_order,[0,1,2])
        error('max_order must be 0, 1, or 2.');
    end

    p = p(:);

    if numel(p) ~= 2
        error('p must contain [p_x;p_y].');
    end

    S = build_preactivation(X,p,basis);
    A = activation_derivatives(S,'gaussian',max_order);

    state = struct();
    state.phi = A.phi;

    if max_order >= 1
        Wx = p(1)*basis.Z(1,:);
        Wy = p(2)*basis.Z(2,:);

        state.dx = A.d1.*Wx;
        state.dy = A.d1.*Wy;
    end

    if max_order >= 2
        state.dxx = A.d2.*(Wx.^2);
        state.dyy = A.d2.*(Wy.^2);
        state.lap = state.dxx+state.dyy;
    end

    if ~isempty(coef)
        coef = coef(:);
        state.u = state.phi*coef;

        if max_order >= 1
            state.d1 = {state.dx*coef;state.dy*coef};
        end

        if max_order >= 2
            state.d2 = {state.dxx*coef;state.dyy*coef};
        end
    end
end
