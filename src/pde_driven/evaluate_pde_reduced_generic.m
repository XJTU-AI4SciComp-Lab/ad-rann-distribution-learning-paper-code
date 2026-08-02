function [objective,grad,info] = ...
    evaluate_pde_reduced_generic(p,system_builder,lambda,ls_opts)
%EVALUATE_PDE_REDUCED_GENERIC Generic PDE-driven AD-RaNN evaluator.
%
%   [objective,grad,info] = evaluate_pde_reduced_generic( ...
%       p,system_builder,lambda,ls_opts)
%
% system_builder is a function handle with interface
%
%   [M,y,dM] = system_builder(p)
%
% where dM is a cell array containing dM/dp_k. Neither the PDE, the input
% dimension, nor the activation is hard-coded here. This generic path is
% intentionally simple and can be slower because it stores all dM matrices.
% Problem-specific fast evaluators may instead form only (dM/dp_k)*w.

    if nargin < 4 || isempty(ls_opts)
        ls_opts = struct();
    end

    if ~isa(system_builder,'function_handle')
        error('system_builder must be a function handle.');
    end

    p = p(:);

    [M,y,dM] = system_builder(p);

    if ~iscell(dM) || numel(dM) ~= numel(p)
        error('system_builder must return one dM cell entry per p component.');
    end

    [objective,grad,info] = ...
        reduced_objective_gradient(M,y,dM,lambda,ls_opts);
end
