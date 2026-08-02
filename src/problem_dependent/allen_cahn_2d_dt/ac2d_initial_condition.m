function u0 = ac2d_initial_condition(X)
%AC2D_INITIAL_CONDITION
%
%   u0(x,y) = cos(pi*x/2)*cos(pi*y/2).

    if size(X,2) ~= 2
        error('X must be N-by-2.');
    end

    u0 = ...
        cos(0.5*pi*X(:,1)) .* ...
        cos(0.5*pi*X(:,2));
end
