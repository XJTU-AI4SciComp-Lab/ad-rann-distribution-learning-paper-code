function Phi = activation_features(X,p,basis,activation)
%ACTIVATION_FEATURES Dimension-independent randomized features.
%
%   Phi = activation_features(X,p,basis,activation)
%
% Supported activations: gaussian, tanh, sin.

    S = build_preactivation(X,p,basis);
    Phi = evaluate_activation(S,activation);
end
