function [model,info] = fit_burgers_deeponet(training,basis,p,cfg)
%FIT_BURGERS_DEEPONET Fit only the linear output coefficients.

    p = p(:);

    if numel(p) ~= 3 || any(~isfinite(p)) || any(p <= 0)
        error('p must be [rb;rx;rt] with finite positive entries.');
    end

    scales = [p(1);p(2);p(2);p(3)];

    B = evaluate_burgers_deeponet_branch( ...
        training.branch_inputs,p(1),basis.branch,cfg.activation);
    T = evaluate_burgers_deeponet_trunk( ...
        training.Y,scales(2:4),basis.trunk,cfg.activation);

    [W,solve_info] = solve_grouped_tensor_least_squares( ...
        B,T,training.function_index,training.target, ...
        cfg.training.ridge_lambda,cfg.linear_solver);

    model = struct();
    model.W = W;
    model.basis = basis;
    model.p = p;
    model.scales = scales;
    model.activation = cfg.activation;
    model.num_branch = size(W,1);
    model.num_trunk = size(W,2);
    model.training_mode = 'data-driven RaNN-DeepONet + DDAD';

    info = solve_info;
    info.category_counts = training.category_counts;
    info.parameter_order = {'rb','rx','rt'};
end
