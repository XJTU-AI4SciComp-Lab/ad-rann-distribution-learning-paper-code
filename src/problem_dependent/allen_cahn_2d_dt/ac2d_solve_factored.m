function coef = ac2d_solve_factored(factor,rhs)
%AC2D_SOLVE_FACTORED Solve a cached least-squares system.

    rhs = rhs(:);

    switch factor.mode

        case 'qr'

            coef = factor.R \ (factor.Q'*rhs);

        case 'project_solver'

            coef = solve_least_squares( ...
                factor.M,rhs,factor.ls_opts);

        otherwise

            error('Unknown factorization mode: %s',factor.mode);
    end

    if isa(coef,'gpuArray')
        coef = gather(coef);
    end

    coef = coef(:);
end
