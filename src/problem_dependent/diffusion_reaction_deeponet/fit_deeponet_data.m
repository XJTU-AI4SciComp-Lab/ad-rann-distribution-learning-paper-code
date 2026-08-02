function [model,info] = fit_deeponet_data( ...
    F,Y,target,p,basis,cfg)
%FIT_DEEPONET_DATA Fit output coefficients to supervised solution values.

    B = evaluate_deeponet_branch( ...
        F,p(1),basis.branch,cfg.activation);
    T = evaluate_deeponet_trunk( ...
        Y,p(2:3),basis.trunk,cfg.activation);

    A = build_deeponet_design_matrix(B.value,T.value);

    if cfg.data_fit.lambda > 0
        [w,solve_info] = solve_ridge( ...
            A,target,cfg.data_fit.lambda,cfg.linear_solver);
    else
        [w,solve_info] = solve_least_squares( ...
            A,target,cfg.linear_solver);
    end

    w = w(:);
    residual = A*w-target;

    model = struct();
    model.p = p(:);
    model.basis = basis;
    model.W = reshape(w,size(B.value,2),size(T.value,2));
    model.training_mode = 'RANN_DDAD';

    info = struct();
    info.residual_mse = mean(residual.^2);
    info.relative_l2 = relative_l2(A*w,target);
    info.solver = solve_info;
end
