function [M, y, dM] = build_system(p, problem, basis)
%BUILD_SYSTEM Generic Poisson residual/boundary system.

    Di = gaussian_derivatives(problem.Xi,p,basis);

    A = -Di.xx-Di.yy;

    dA1 = -Di.dp{1}.xx-Di.dp{1}.yy;
    dA2 = -Di.dp{2}.xx-Di.dp{2}.yy;

    Db = gaussian_derivatives(problem.Xb,p,basis);

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
