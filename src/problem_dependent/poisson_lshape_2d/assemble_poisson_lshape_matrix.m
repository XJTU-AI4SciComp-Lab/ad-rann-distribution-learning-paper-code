function [M,y,parts] = assemble_poisson_lshape_matrix( ...
        p,problem,basis,activation)
%ASSEMBLE_POISSON_LSHAPE_MATRIX Assemble -Delta and Dirichlet rows only.
%
% This final-refit assembly avoids constructing dM/dp. It reuses the common
% build_preactivation and activation_derivatives routines.

    p = p(:);

    Si = build_preactivation(problem.Xi,p,basis);
    Ai = activation_derivatives(Si,activation,2);

    W = basis.Z.*p;
    weight_norm2 = sum(W.^2,1);

    Mi = -Ai.d2.*weight_norm2;

    Sb = build_preactivation(problem.Xb,p,basis);
    Ab = activation_derivatives(Sb,activation,0);

    Mb = problem.boundary_penalty*Ab.phi;

    M = [Mi;Mb];
    y = problem.y;

    if nargout >= 3
        parts = struct();
        parts.Mi = Mi;
        parts.Mb = Mb;
    end
end
