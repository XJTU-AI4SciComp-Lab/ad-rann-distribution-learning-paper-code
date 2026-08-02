function Phi = tanh_features(X,p,basis)
%TANH_FEATURES Evaluate randomized tanh features.

    S = build_preactivation(X,p,basis);
    Phi = evaluate_activation(S,'tanh');
end
