function [objective,grad,info] = evaluate_deeponet_ddad_reduced( ...
    p,F,Y,target,basis,lambda,ls_opts,activation)
%EVALUATE_DEEPONET_DDAD_REDUCED Exact reduced DDAD objective and gradient.
%
% Parameter order is p=[rb;rx;rt].  The ridge-reduced objective is
%
%   (||A(p)w-y||^2 + lambda*||w||^2)/(2*N).
%
% The envelope gradient is exact for this objective.

    if nargin < 8 || isempty(activation)
        activation = 'tanh';
    end

    p = p(:);
    target = target(:);

    if numel(p) ~= 3 || any(~isfinite(p)) || any(p <= 0)
        error('p must be [rb;rx;rt] with finite positive entries.');
    end

    if size(F,1) ~= size(Y,1) || size(F,1) ~= numel(target)
        error('F, Y, and target row counts must agree.');
    end

    N = size(F,1);

    B = evaluate_deeponet_branch(F,p(1),basis.branch,activation);
    T = evaluate_deeponet_trunk(Y,p(2:3),basis.trunk,activation);

    A = build_deeponet_design_matrix(B.value,T.value);
    [w,ridge_info] = solve_ridge(A,target,lambda,ls_opts);
    w = w(:);

    res = A*w-target;
    residual_sq = real(res'*res);
    coefficient_sq = real(w'*w);

    objective = ...
        (residual_sq+lambda*coefficient_sq)/(2*N);

    W = reshape(w,size(B.value,2),size(T.value,2));
    BW = B.value*W;

    du_drb = sum((B.drb*W).*T.value,2);
    du_drx = sum(BW.*T.drx,2);
    du_drt = sum(BW.*T.drt,2);

    grad = [ ...
        real(res'*du_drb); ...
        real(res'*du_drx); ...
        real(res'*du_drt)]/N;

    info = struct();
    info.residual_mse = residual_sq/N;
    info.coefficient_norm = sqrt(coefficient_sq);
    info.w = w;
    info.ridge = ridge_info;
    info.parameter_order = {'rb','rx','rt'};
end
