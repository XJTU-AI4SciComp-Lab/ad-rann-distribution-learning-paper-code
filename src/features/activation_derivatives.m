function A = activation_derivatives(S,activation,max_order)
%ACTIVATION_DERIVATIVES Values and derivatives with respect to preactivation.
%
%   A = activation_derivatives(S,activation,max_order)
%
% A.phi = sigma(S)
% A.d1  = sigma'(S)       if max_order >= 1
% A.d2  = sigma''(S)      if max_order >= 2
% A.d3  = sigma'''(S)     if max_order >= 3
%
% Supported activations: gaussian, tanh, sin.

    if nargin < 2 || isempty(activation)
        activation = 'gaussian';
    end

    if nargin < 3 || isempty(max_order)
        max_order = 3;
    end

    if ~isscalar(max_order) || max_order < 0 || max_order > 3 || max_order ~= floor(max_order)
        error('max_order must be an integer in {0,1,2,3}.');
    end

    activation = lower(strtrim(char(activation)));

    switch activation

        case {'gaussian','gauss'}
            E = exp(-S.^2);
            A.phi = E;

            if max_order >= 1
                A.d1 = -2*S.*E;
            end

            if max_order >= 2
                A.d2 = (4*S.^2-2).*E;
            end

            if max_order >= 3
                A.d3 = (12*S-8*S.^3).*E;
            end

        case 'tanh'
            T = tanh(S);
            R = 1-T.^2;

            A.phi = T;

            if max_order >= 1
                A.d1 = R;
            end

            if max_order >= 2
                A.d2 = -2*T.*R;
            end

            if max_order >= 3
                A.d3 = R.*(6*T.^2-2);
            end

        case {'sin','sine'}
            A.phi = sin(S);

            if max_order >= 1
                A.d1 = cos(S);
            end

            if max_order >= 2
                A.d2 = -A.phi;
            end

            if max_order >= 3
                A.d3 = -cos(S);
            end

        otherwise
            error(['Unknown activation "%s". Supported activations are ', ...
                   'gaussian, tanh, and sin.'],activation);
    end
end
