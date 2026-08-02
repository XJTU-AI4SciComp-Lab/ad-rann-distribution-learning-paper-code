function Phi = sin_features(X,p,basis)
%SIN_FEATURES Evaluate randomized sine features.

    S = build_preactivation(X,p,basis);
    Phi = evaluate_activation(S,'sin');
end
