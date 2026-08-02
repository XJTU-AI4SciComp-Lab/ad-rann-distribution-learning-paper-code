function [rel_l2,rel_h1,rel_linf] = ...
    sharp_layer_error_metrics(pred,grad_pred,exact,grad_exact)
%SHARP_LAYER_ERROR_METRICS Relative L2, H1, and Linf errors.
%
% Discrete relative H1 norm on the common uniform test grid:
%
%   ||e||_H1 / ||u||_H1
%
% = sqrt( ||e||_2^2 + ||grad e||_2^2 )
%   ------------------------------------------------
%   sqrt( ||u||_2^2 + ||grad u||_2^2 ).
%
% Since numerator and denominator use the same uniform grid, the common
% quadrature factor cancels in the relative error.

    pred = pred(:);
    exact = exact(:);

    if size(grad_pred,1) ~= numel(pred) || ...
            size(grad_exact,1) ~= numel(exact)
        error('Gradient arrays must have one row per test point.');
    end

    if size(grad_pred,2) ~= size(grad_exact,2)
        error('Predicted and exact gradients must have matching dimensions.');
    end

    e = pred-exact;
    grad_e = grad_pred-grad_exact;

    rel_l2 = ...
        norm(e) / ...
        max(norm(exact),eps);

    h1_num_sq = ...
        sum(e.^2) + ...
        sum(grad_e(:).^2);

    h1_den_sq = ...
        sum(exact.^2) + ...
        sum(grad_exact(:).^2);

    rel_h1 = ...
        sqrt(h1_num_sq) / ...
        max(sqrt(h1_den_sq),eps);

    rel_linf = ...
        max(abs(e)) / ...
        max(max(abs(exact)),eps);
end
