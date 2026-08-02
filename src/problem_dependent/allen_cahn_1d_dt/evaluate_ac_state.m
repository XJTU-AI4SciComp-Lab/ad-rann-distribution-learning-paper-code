function state = evaluate_ac_state(X,p,basis,coef)
%EVALUATE_AC_STATE Gaussian feature values and derivatives in 1-D.
%
% Reuses common src:
%
%   build_preactivation
%   activation_derivatives

    if nargin < 4
        coef = [];
    end

    X = X(:);
    p = p(:);

    if numel(p) ~= 1
        error('This example requires one scalar distribution parameter.');
    end

    S = build_preactivation(X,p,basis);
    A = activation_derivatives(S,'gaussian',2);

    W = p(1)*basis.Z(1,:);

    state = struct();

    state.phi = A.phi;
    state.ux = A.d1.*W;
    state.uxx = A.d2.*(W.^2);

    if ~isempty(coef)

        coef = coef(:);

        state.u = state.phi*coef;
        state.du = state.ux*coef;
        state.d2u = state.uxx*coef;
    end
end
