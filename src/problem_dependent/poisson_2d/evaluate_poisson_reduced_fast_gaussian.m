function [objective,grad,info] = ...
    evaluate_poisson_reduced_fast_gaussian(p,cache,lambda,ls_opts)
%EVALUATE_POISSON_REDUCED_FAST_GAUSSIAN Named Gaussian compatibility wrapper.
%
% The main Gaussian run intentionally calls evaluate_poisson_reduced_fast.m
% directly so the established inner-loop path is unchanged.

    [objective,grad,info] = ...
        evaluate_poisson_reduced_fast(p,cache,lambda,ls_opts);
end
