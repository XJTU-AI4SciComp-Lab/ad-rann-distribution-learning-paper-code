function state = evaluate_burgers2d_base_state(X,p,basis,coef,max_order)
%EVALUATE_BURGERS2D_BASE_STATE Gaussian values and spatial derivatives.
%
% Without coef, feature matrices are returned:
%   phi, dx, dy, dsum, dxx, dyy, lap.
%
% With coef, the full feature matrix phi and the solution-state vectors
% required by src/layer_growth are returned:
%   u, d1{1}, d1{2}, d2{1}, d2{2}.
%
% Assembly is chunked to avoid simultaneously storing several temporary
% N-by-m activation-derivative matrices.

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

    if ~ismember(max_order,[0,1,2])
        error('max_order must be 0, 1, or 2.');
    end

    N = size(X,1);
    m = size(basis.Z,2);
    chunk_rows = 500;

    Wx = p(1)*basis.Z(1,:);
    Wy = p(2)*basis.Z(2,:);

    state = struct();
    state.phi = zeros(N,m);

    if isempty(coef)

        if max_order >= 1
            state.dx = zeros(N,m);
            state.dy = zeros(N,m);
            state.dsum = zeros(N,m);
        end

        if max_order >= 2
            state.dxx = zeros(N,m);
            state.dyy = zeros(N,m);
            state.lap = zeros(N,m);
        end

    else

        coef = coef(:);

        if numel(coef) ~= m
            error('coef must contain one value per base feature.');
        end

        state.u = zeros(N,1);

        if max_order >= 1
            state.d1 = {zeros(N,1);zeros(N,1)};
        end

        if max_order >= 2
            state.d2 = {zeros(N,1);zeros(N,1)};
        end
    end

    derivative_order = max(2,max_order);

    for first = 1:chunk_rows:N

        last = min(first+chunk_rows-1,N);
        rows = first:last;

        S = build_preactivation(X(rows,:),p,basis);
        A = activation_derivatives(S,'gaussian',derivative_order);

        state.phi(rows,:) = A.phi;

        if isempty(coef)

            if max_order >= 1
                dx = A.d1.*Wx;
                dy = A.d1.*Wy;

                state.dx(rows,:) = dx;
                state.dy(rows,:) = dy;
                state.dsum(rows,:) = dx+dy;
            end

            if max_order >= 2
                dxx = A.d2.*(Wx.^2);
                dyy = A.d2.*(Wy.^2);

                state.dxx(rows,:) = dxx;
                state.dyy(rows,:) = dyy;
                state.lap(rows,:) = dxx+dyy;
            end

        else

            state.u(rows) = A.phi*coef;

            if max_order >= 1
                state.d1{1}(rows) = (A.d1.*Wx)*coef;
                state.d1{2}(rows) = (A.d1.*Wy)*coef;
            end

            if max_order >= 2
                state.d2{1}(rows) = ...
                    (A.d2.*(Wx.^2))*coef;
                state.d2{2}(rows) = ...
                    (A.d2.*(Wy.^2))*coef;
            end
        end
    end
end
