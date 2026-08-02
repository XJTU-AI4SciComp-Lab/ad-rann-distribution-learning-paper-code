function Phi = evaluate_activation(S,activation)
%EVALUATE_ACTIVATION Evaluate a supported activation function.
%
% Supported activation names:
%   'gaussian' or 'gauss' : exp(-S.^2)
%   'tanh'                : tanh(S)
%   'sin' or 'sine'       : sin(S)

    if nargin < 2 || isempty(activation)
        activation = 'gaussian';
    end

    activation = lower(strtrim(char(activation)));

    switch activation

        case {'gaussian','gauss'}
            Phi = exp(-S.^2);

        case 'tanh'
            Phi = tanh(S);

        case {'sin','sine'}
            Phi = sin(S);

        otherwise
            error(['Unknown activation "%s". Supported activations are ', ...
                   'gaussian, tanh, and sin.'],activation);
    end
end
