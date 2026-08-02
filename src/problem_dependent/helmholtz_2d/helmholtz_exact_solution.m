function u = helmholtz_exact_solution(X,problem)
%HELMHOLTZ_EXACT_SOLUTION Exact oscillatory Helmholtz solution.
%
%   u(x,y)=sin(a1*pi*x) sin(a2*pi*y).

    if size(X,2) ~= 2
        error('X must be N-by-2.');
    end

    x = X(:,1);
    y = X(:,2);

    u = ...
        sin(problem.a1*pi*x) .* ...
        sin(problem.a2*pi*y);
end
