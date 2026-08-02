function mask = poisson_lshape_is_inside(X,include_boundary,tol)
%POISSON_LSHAPE_IS_INSIDE Membership in the L-shaped domain or its closure.
%
% Omega = (-1,1)^2 \ ([0,1] x [-1,0]).

    if nargin < 2 || isempty(include_boundary)
        include_boundary = false;
    end

    if nargin < 3 || isempty(tol)
        tol = 1e-12;
    end

    if size(X,2) ~= 2
        error('X must be N-by-2.');
    end

    x = X(:,1);
    y = X(:,2);

    if include_boundary
        in_box = ...
            x >= -1-tol & x <= 1+tol & ...
            y >= -1-tol & y <= 1+tol;

        % Remove only the strict interior of the cut-out quadrant, thereby
        % retaining the two re-entrant boundary segments.
        in_removed_open_quadrant = x > tol & y < -tol;
        mask = in_box & ~in_removed_open_quadrant;
    else
        in_box = ...
            x > -1+tol & x < 1-tol & ...
            y > -1+tol & y < 1-tol;

        % The cut-out and its two edges are not PDE-interior points.
        in_removed_closed_quadrant = x >= -tol & y <= tol;
        mask = in_box & ~in_removed_closed_quadrant;
    end
end
