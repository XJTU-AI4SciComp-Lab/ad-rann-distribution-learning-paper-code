function [objective,grad,info] = ...
    evaluate_burgers_deeponet_ddad_reduced( ...
        p,training,basis,cfg)
%EVALUATE_BURGERS_DEEPONET_DDAD_REDUCED Exact supervised DDAD gradient.
%
% Independent parameter order: p=[rb;rx;rt].  The periodic trunk uses
% expanded scales [rx;rx;rt] for [cos(2*pi*x),sin(2*pi*x),t].

    p = p(:);

    if numel(p) ~= 3 || any(~isfinite(p)) || any(p <= 0)
        error('p must be [rb;rx;rt] with finite positive entries.');
    end

    F = training.branch_inputs;
    Y = training.Y;
    target = training.target(:);
    group_index = training.function_index;
    N = numel(target);

    branch_scale = p(1)*ones(size(F,2),1);
    branch_argument = build_preactivation( ...
        F,branch_scale,basis.branch);
    branch_activation = activation_derivatives( ...
        branch_argument,cfg.activation,1);

    B = branch_activation.phi;
    dargument_drb = ...
        F*basis.branch.Z- ...
        sum(basis.branch.C.*basis.branch.Z,1);
    dB_drb = branch_activation.d1.*dargument_drb;

    x = Y(:,1);
    t = Y(:,2);
    periodic_input = [cos(2*pi*x),sin(2*pi*x),t];
    trunk_scales = [p(2);p(2);p(3)];

    trunk_argument = build_preactivation( ...
        periodic_input,trunk_scales,basis.trunk);
    trunk_activation = activation_derivatives( ...
        trunk_argument,cfg.activation,1);

    T = trunk_activation.phi;

    dargument_drx = ...
        periodic_input(:,1:2)*basis.trunk.Z(1:2,:)- ...
        sum(basis.trunk.C(1:2,:).*basis.trunk.Z(1:2,:),1);
    dargument_drt = ...
        periodic_input(:,3)*basis.trunk.Z(3,:)- ...
        basis.trunk.C(3,:).*basis.trunk.Z(3,:);

    dT_drx = trunk_activation.d1.*dargument_drx;
    dT_drt = trunk_activation.d1.*dargument_drt;

    [W,solve_info] = solve_grouped_tensor_least_squares( ...
        B,T,group_index,target,cfg.ddad.ridge_lambda, ...
        cfg.ddad.linear_solver);

    grouped_coefficients = B*W;
    prediction = sum( ...
        T.*grouped_coefficients(group_index,:),2);
    residual = prediction-target;

    coefficient_sq = real(W(:)'*W(:));
    residual_sq = real(residual'*residual);

    objective = ...
        (residual_sq+cfg.ddad.ridge_lambda*coefficient_sq)/(2*N);

    grouped_drb = dB_drb*W;
    du_drb = sum(T.*grouped_drb(group_index,:),2);
    du_drx = sum( ...
        dT_drx.*grouped_coefficients(group_index,:),2);
    du_drt = sum( ...
        dT_drt.*grouped_coefficients(group_index,:),2);

    grad = real([ ...
        residual'*du_drb; ...
        residual'*du_drx; ...
        residual'*du_drt])/N;

    info = struct();
    info.residual_mse = residual_sq/N;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.solver = solve_info;
    info.parameter_order = {'rb','rx','rt'};
end
