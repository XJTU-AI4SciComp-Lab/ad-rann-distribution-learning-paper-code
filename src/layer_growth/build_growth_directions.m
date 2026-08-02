function directions = build_growth_directions(dim,num_features,seed)
%BUILD_GROWTH_DIRECTIONS Random directions for a layer-growth block.
%
% directions = build_growth_directions(dim,num_features,seed)
%
% Output:
%   directions : dim-by-num_features, entries sampled from U(-1,1).
%
% The global RNG state is restored before returning, so this helper does not
% disturb the random stream used by the calling experiment.

    if nargin < 3 || isempty(seed)
        seed = 42;
    end

    if ~isscalar(dim) || dim < 1 || dim ~= floor(dim)
        error('dim must be a positive integer.');
    end

    if ~isscalar(num_features) || num_features < 1 || ...
            num_features ~= floor(num_features)
        error('num_features must be a positive integer.');
    end

    old_rng = rng;
    cleanup = onCleanup(@() rng(old_rng)); %#ok<NASGU>

    rng(seed,'twister');

    directions = 2*rand(dim,num_features)-1;

    % Avoid a practically zero direction.
    nrm = sqrt(sum(directions.^2,1));
    tiny = nrm < 1e-12;

    if any(tiny)
        directions(1,tiny) = 1;
    end
end
