function [model,info] = fit_deeponet_physics_picard( ...
    F,Y,rhs,p,basis,cfg,W_initial)
%FIT_DEEPONET_PHYSICS_PICARD Physics-informed coefficient solve.
%
% Solves the Picard linearization of
%
%   u_t-D*u_xx-k*u^2=f(x)
%
% with u=t*x*(1-x)*G, which enforces zero initial and boundary data.

    if nargin < 7 || isempty(W_initial)
        W_initial = zeros( ...
            basis.num_branch,basis.num_trunk);
    end

    rhs = rhs(:);

    B = evaluate_deeponet_branch( ...
        F,p(1),basis.branch,cfg.activation);
    T = evaluate_deeponet_trunk( ...
        Y,p(2:3),basis.trunk,cfg.activation);

    if ~isequal(size(W_initial), ...
            [size(B.value,2),size(T.value,2)])
        error('W_initial size does not match the full basis.');
    end

    W = W_initial;
    u_old = sum((B.value*W).*T.value,2);

    maxit = cfg.physics.max_picard_iterations;
    relaxation = cfg.physics.relaxation;
    tolerance = cfg.physics.relative_tolerance;

    residual_history = nan(maxit,1);
    change_history = nan(maxit,1);
    solver_history = cell(maxit,1);

    completed = 0;

    for it = 1:maxit

        L = T.t ...
            -cfg.physics.diffusion*T.xx ...
            -cfg.physics.reaction*u_old.*T.value;

        A = build_deeponet_design_matrix(B.value,L);

        if cfg.physics.lambda > 0
            [w,solver_info] = solve_ridge( ...
                A,rhs,cfg.physics.lambda,cfg.linear_solver);
        else
            [w,solver_info] = solve_least_squares( ...
                A,rhs,cfg.linear_solver);
        end

        W_candidate = reshape( ...
            w,size(B.value,2),size(T.value,2));

        W_new = relaxation*W_candidate+(1-relaxation)*W;
        u_new = sum((B.value*W_new).*T.value,2);

        change = norm(u_new-u_old)/max(norm(u_new),eps);
        residual = norm(A*w-rhs)/max(norm(rhs),eps);

        residual_history(it) = residual;
        change_history(it) = change;
        solver_history{it} = solver_info;
        completed = it;

        W = W_new;
        u_old = u_new;

        if cfg.verbose
            fprintf([ ...
                'PI Picard %d/%d | PDE rel residual=%.6e | ', ...
                'relative change=%.6e\n'], ...
                it,maxit,residual,change);
        end

        if change <= tolerance
            break;
        end
    end

    model = struct();
    model.p = p(:);
    model.basis = basis;
    model.W = W;
    model.training_mode = 'PI_RANN_DDAD';

    info = struct();
    info.num_iterations = completed;
    info.residual_history = residual_history(1:completed);
    info.change_history = change_history(1:completed);
    info.solver_history = solver_history(1:completed);
    info.final_relative_residual = residual_history(completed);
end
