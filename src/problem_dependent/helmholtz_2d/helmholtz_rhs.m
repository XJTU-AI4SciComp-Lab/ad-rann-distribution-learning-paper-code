function q = helmholtz_rhs(X,problem)
%HELMHOLTZ_RHS Source term for Delta u + k^2 u = q.

    u = helmholtz_exact_solution(X,problem);

    multiplier = ...
        problem.k^2 - ...
        pi^2*(problem.a1^2+problem.a2^2);

    q = multiplier*u;
end
