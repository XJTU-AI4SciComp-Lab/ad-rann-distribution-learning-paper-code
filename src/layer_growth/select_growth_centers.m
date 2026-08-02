function [centers,indices,scores] = ...
    select_growth_centers(X,residual,num_centers,policy)
%SELECT_GROWTH_CENTERS Select layer-growth centers from a residual indicator.
%
% [centers,indices,scores] = ...
%     select_growth_centers(X,residual,num_centers,policy)
%
% Inputs:
%   X           : N-by-d candidate coordinates.
%   residual    : N-by-1 residual/indicator values.
%   num_centers : number of centers to select.
%   policy      : currently 'top_abs' (default).
%
% The selection is deliberately independent of the PDE.  Any residual or
% externally supplied indicator can be used.

    if nargin < 4 || isempty(policy)
        policy = 'top_abs';
    end

    if size(X,1) ~= numel(residual)
        error('size(X,1) must equal numel(residual).');
    end

    residual = residual(:);

    if any(~isfinite(X(:))) || any(~isfinite(residual))
        error('X and residual must contain only finite values.');
    end

    N = size(X,1);

    if ~isscalar(num_centers) || num_centers < 1 || ...
            num_centers ~= floor(num_centers)
        error('num_centers must be a positive integer.');
    end

    num_centers = min(num_centers,N);

    switch lower(char(policy))
        case 'top_abs'
            indicator = abs(residual);

            % maxk avoids sorting the entire vector when num_centers << N.
            [scores,indices] = maxk(indicator,num_centers);

        otherwise
            error('Unknown growth-center policy: %s',policy);
    end

    centers = X(indices,:);
end
