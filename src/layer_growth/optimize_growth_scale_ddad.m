function [rho_best,history] = optimize_growth_scale_ddad( ...
    rho0,cache,lambda,ls_opts,optimizer_opts)
%OPTIMIZE_GROWTH_SCALE_DDAD Optimize the scalar growth scale using DDAD.
%
% This routine intentionally reuses the project's common
% optimize_distribution_adam implementation.  There is no separate Adam
% implementation for layer growth.

    if nargin < 5 || isempty(optimizer_opts)
        optimizer_opts = struct();
    end

    if ~isfield(optimizer_opts,'lower_bound') || ...
            isempty(optimizer_opts.lower_bound)
        optimizer_opts.lower_bound = 1e-3;
    end

    if ~isfield(optimizer_opts,'upper_bound') || ...
            isempty(optimizer_opts.upper_bound)
        optimizer_opts.upper_bound = 100;
    end

    if ~isfield(optimizer_opts,'selection_metric') || ...
            isempty(optimizer_opts.selection_metric)
        optimizer_opts.selection_metric = 'residual_mse';
    end

    objective_fun = @(rho) ...
        evaluate_growth_ddad_reduced_fast( ...
            rho,cache,lambda,ls_opts);

    [rho_best,history] = ...
        optimize_distribution_adam( ...
            rho0,objective_fun,optimizer_opts);

    rho_best = rho_best(1);
end
