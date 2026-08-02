function S = build_preactivation(X,p,basis)
%BUILD_PREACTIVATION Dimension-independent randomized-feature preactivation.
%
%   S = build_preactivation(X,p,basis)
%
% X       : N-by-d input points
% p       : d-by-1 distribution parameters
% basis.Z : d-by-m frozen base random weights
% basis.C : d-by-m frozen centers
%
% The parameterized hidden weights and bias are
%
%   W = diag(p)*Z,
%   b = -sum(C.*W,1),
%
% and S = X*W+b.

    p = p(:);

    if ~isnumeric(X) || ~ismatrix(X) || isempty(X)
        error('X must be a nonempty numeric N-by-d matrix.');
    end

    if any(~isfinite(X(:))) || any(~isfinite(p))
        error('X and p must be finite.');
    end

    d = size(X,2);

    if numel(p) ~= d
        error('numel(p) must equal size(X,2).');
    end

    if ~isstruct(basis) || ~isfield(basis,'Z') || ~isfield(basis,'C')
        error('basis must contain fields Z and C.');
    end

    if size(basis.Z,1) ~= d
        error('size(basis.Z,1) must equal size(X,2).');
    end

    if ~isequal(size(basis.Z),size(basis.C))
        error('basis.Z and basis.C must have the same size.');
    end

    W = basis.Z .* p;
    b = -sum(basis.C .* W,1);

    S = X*W + b;
end
