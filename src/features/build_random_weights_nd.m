function basis = build_random_weights_nd(num_features,domain,seed)
%BUILD_RANDOM_WEIGHTS_ND Dimension-independent frozen random-feature basis.
%
%   basis = build_random_weights_nd(num_features,domain,seed)
%
% domain is d-by-2:
%
%   [x1_min x1_max
%    x2_min x2_max
%       ...     ...]
%
% Z ~ U(-1,1) in R^{d x m}; centers C are sampled uniformly in domain.
%
% For d=2 this uses the same two matrix draws and the same draw order as
% build_random_weights.m, so with the same seed it produces the same Z and C.

    if nargin < 3
        error('build_random_weights_nd requires num_features, domain, and seed.');
    end

    if size(domain,2) ~= 2 || isempty(domain)
        error('domain must be a nonempty d-by-2 matrix.');
    end

    if any(~isfinite(domain(:))) || any(domain(:,1) >= domain(:,2))
        error('Each domain row must contain finite lower/upper bounds with lower < upper.');
    end

    if ~isscalar(num_features) || num_features < 1 || num_features ~= floor(num_features)
        error('num_features must be a positive integer.');
    end

    d = size(domain,1);

    stream = RandStream('mt19937ar','Seed',seed);

    % Keep the random draw order identical to the existing 2-D routine.
    Z = 2*rand(stream,d,num_features)-1;
    U = rand(stream,d,num_features);

    C = domain(:,1) + (domain(:,2)-domain(:,1)).*U;

    basis.Z = Z;
    basis.C = C;
    basis.num_features = num_features;
    basis.seed = seed;
    basis.domain = domain;
    basis.dim = d;
end
