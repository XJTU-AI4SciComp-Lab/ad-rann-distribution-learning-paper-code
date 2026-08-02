function cache = prepare_data_cache(X,y,basis)
%PREPARE_DATA_CACHE Dimension-independent cache for data-driven AD-RaNN.
%
%   cache = prepare_data_cache(X,y,basis)
%
% X : N-by-d input coordinates. Examples:
%       1-D data       -> X = t(:)
%       2-D data       -> X = [x,y]
%       space-time     -> X = [x,t]
%       2-D space-time -> X = [x,y,t]
% y : N-by-1 target data
%
% basis.Z and basis.C must be d-by-m.
%
% The cache stores Q{k} such that
%
%   S(p) = sum_k p(k)*Q{k}
%
% and therefore the same cache can be used for gaussian, tanh, and sin.

    if nargin < 3
        error('prepare_data_cache requires X, y, and basis.');
    end

    if ~isstruct(basis) || ~isfield(basis,'Z') || ~isfield(basis,'C')
        error('basis must contain fields Z and C.');
    end

    if ~isequal(size(basis.Z),size(basis.C))
        error('basis.Z and basis.C must have the same size.');
    end

    if ~isnumeric(X) || isempty(X)
        error('X must be a nonempty numeric array.');
    end

    if isvector(X) && size(basis.Z,1) == 1
        X = X(:);
    end

    if ~ismatrix(X)
        error('X must be an N-by-d matrix.');
    end

    if any(~isfinite(X(:)))
        error('X contains NaN or Inf.');
    end

    y = y(:);

    if numel(y) ~= size(X,1)
        error('The number of target values must equal size(X,1).');
    end

    if any(~isfinite(y))
        error('y contains NaN or Inf.');
    end

    d = size(X,2);

    if size(basis.Z,1) ~= d
        error('size(basis.Z,1) must equal size(X,2).');
    end

    N = size(X,1);
    m = size(basis.Z,2);

    Q = cell(d,1);

    for k = 1:d
        Q{k} = ...
            (X(:,k)-basis.C(k,:)).*basis.Z(k,:);
    end

    cache.Q = Q;
    cache.y = y;
    cache.num_rows = N;
    cache.num_features = m;
    cache.dim = d;
end
