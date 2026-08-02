function u = burgers2d_exact(X,t,epsilon)
%BURGERS2D_EXACT Exact travelling-wave solution.
%
%   u(x,y,t) = 1/(1+exp((x+y-t)/(2*epsilon))).

    if size(X,2) ~= 2
        error('X must be N-by-2.');
    end

    if ~isscalar(t) || ~isfinite(t)
        error('t must be a finite scalar.');
    end

    if ~isscalar(epsilon) || epsilon <= 0 || ~isfinite(epsilon)
        error('epsilon must be a finite positive scalar.');
    end

    z = (X(:,1)+X(:,2)-t)/(2*epsilon);

    % Stable logistic evaluation.
    u = zeros(size(z));
    positive = z >= 0;

    ez = exp(-z(positive));
    u(positive) = ez./(1+ez);

    ez = exp(z(~positive));
    u(~positive) = 1./(1+ez);
end
