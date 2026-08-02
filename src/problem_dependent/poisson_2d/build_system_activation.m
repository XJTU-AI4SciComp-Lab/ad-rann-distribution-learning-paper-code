function [M,y,dM] = build_system_activation(p,problem,basis,activation)
%BUILD_SYSTEM_ACTIVATION 2-D Poisson system for tanh/sin/gaussian features.
%
% This is the activation-general reference assembly used by the generic
% PDE-driven path. The existing build_system.m is retained unchanged for
% the legacy Gaussian path.

    if nargin < 4 || isempty(activation)
        activation = 'gaussian';
    end

    Di = feature_derivatives_2d( ...
        problem.Xi,p,basis,activation);

    A = -Di.xx-Di.yy;

    dA1 = -Di.dp{1}.xx-Di.dp{1}.yy;
    dA2 = -Di.dp{2}.xx-Di.dp{2}.yy;

    Db = feature_derivatives_2d( ...
        problem.Xb,p,basis,activation);

    eta = problem.boundary_penalty;

    B = eta*Db.phi;

    dB1 = eta*Db.dp{1}.phi;
    dB2 = eta*Db.dp{2}.phi;

    M = [A;B];

    if isfield(problem,'y')
        y = problem.y;
    else
        y = [problem.fi;eta*problem.gb];
    end

    dM = { ...
        [dA1;dB1], ...
        [dA2;dB2]};
end
