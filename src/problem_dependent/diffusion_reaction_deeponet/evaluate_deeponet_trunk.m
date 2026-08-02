function T = evaluate_deeponet_trunk(Y,p_trunk,basis,activation)
%EVALUATE_DEEPONET_TRUNK Hard-constrained trunk features and derivatives.
%
% The multiplier c(x,t)=t*x*(1-x) enforces homogeneous initial and
% Dirichlet boundary conditions.

    if nargin < 4 || isempty(activation)
        activation = 'tanh';
    end

    if size(Y,2) ~= 2
        error('Y must be N-by-2 with columns [x,t].');
    end

    p_trunk = p_trunk(:);

    if numel(p_trunk) ~= 2 || any(~isfinite(p_trunk)) || ...
            any(p_trunk <= 0)
        error('p_trunk=[rx;rt] must contain two finite positive values.');
    end

    D = feature_derivatives_2d(Y,p_trunk,basis,activation);

    x = Y(:,1);
    t = Y(:,2);

    c = t.*x.*(1-x);
    ct = x.*(1-x);
    cx = t.*(1-2*x);
    cxx = -2*t;

    T.value = c.*D.phi;
    T.t = ct.*D.phi+c.*D.y;
    T.x = cx.*D.phi+c.*D.x;
    T.xx = cxx.*D.phi+2*cx.*D.x+c.*D.xx;

    T.drx = c.*D.dp{1}.phi;
    T.drt = c.*D.dp{2}.phi;

    T.raw = D.phi;
    T.multiplier = c;
end
