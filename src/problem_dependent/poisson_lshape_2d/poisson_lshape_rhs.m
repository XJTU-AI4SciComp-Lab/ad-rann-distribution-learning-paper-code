function f = poisson_lshape_rhs(X)
%POISSON_LSHAPE_RHS The singular function is harmonic away from the corner.

    if size(X,2) ~= 2
        error('X must be N-by-2.');
    end

    f = zeros(size(X,1),1);
end
