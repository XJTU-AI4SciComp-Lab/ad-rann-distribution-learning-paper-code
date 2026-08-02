function state = evaluate_burgers_state(X,p,basis,coef)
%EVALUATE_BURGERS_STATE Gaussian feature matrices for the 1-D Burgers solve.
%
% This routine deliberately reuses the common src functions
%
%   build_preactivation
%   activation_derivatives
%
% rather than maintaining a separate Gaussian derivative implementation.
%
% Returned fields:
%   phi  : feature values
%   ux   : first spatial derivatives of the features
%   uxx  : second spatial derivatives of the features
%
% If coef is supplied, also return
%   u, du, d2u.

    if nargin < 4
        coef = [];
    end

    X = X(:);
    p = p(:);

    if numel(p) ~= 1
        error('This Burgers example requires one scalar distribution parameter.');
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

        if numel(coef) ~= size(state.phi,2)
            error('coef length must equal the number of features.');
        end

        state.u = state.phi*coef;
        state.du = state.ux*coef;
        state.d2u = state.uxx*coef;
    end
end
