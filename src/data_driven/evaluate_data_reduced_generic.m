function [objective,grad,info] = ...
    evaluate_data_reduced_generic(p,cache,lambda,ls_opts,activation)
%EVALUATE_DATA_REDUCED_GENERIC Generic reference DDAD evaluator.
%
% This implementation explicitly builds every dM/dp_k matrix and then
% calls reduced_objective_gradient. It is intended as a clear reference
% implementation and for verification. For repeated optimization use
% evaluate_data_reduced_fast, which forms only derivative actions.

    if nargin < 4 || isempty(ls_opts)
        ls_opts = struct();
    end

    if nargin < 5 || isempty(activation)
        activation = 'gaussian';
    end

    [M,y,dM] = build_data_system(p,cache,activation);

    [objective,grad,info] = ...
        reduced_objective_gradient(M,y,dM,lambda,ls_opts);
end
