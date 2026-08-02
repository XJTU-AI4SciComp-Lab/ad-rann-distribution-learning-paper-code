function B = evaluate_deeponet_branch(F,rb,basis,activation)
%EVALUATE_DEEPONET_BRANCH Branch features and scale derivative.

    if nargin < 4 || isempty(activation)
        activation = 'tanh';
    end

    if ~isscalar(rb) || ~isfinite(rb) || rb <= 0
        error('rb must be a finite positive scalar.');
    end

    num_sensors = size(F,2);

    if size(basis.Z,1) ~= num_sensors
        error('Branch input width does not match the frozen basis.');
    end

    p_branch = rb*ones(num_sensors,1);
    S = build_preactivation(F,p_branch,basis);
    A = activation_derivatives(S,activation,1);

    dS_drb = F*basis.Z-sum(basis.C.*basis.Z,1);

    B.value = A.phi;
    B.drb = A.d1.*dS_drb;
end
