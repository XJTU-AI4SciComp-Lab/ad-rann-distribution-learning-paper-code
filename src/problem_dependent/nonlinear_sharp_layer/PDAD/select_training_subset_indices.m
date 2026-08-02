function indices = select_training_subset_indices(N,n_select,seed)
%SELECT_TRAINING_SUBSET_INDICES Deterministic subset of existing points.
%
% The returned indices always refer to the already configured collocation
% set.  No new physical sample locations are generated.
%
% If n_select >= N, all points are returned in their original order.
%
% The global RNG state is restored before returning.

    if nargin < 3 || isempty(seed)
        seed = 42;
    end

    if ~isscalar(N) || N < 1 || N ~= floor(N)
        error('N must be a positive integer.');
    end

    if ~isscalar(n_select) || n_select < 1 || n_select ~= floor(n_select)
        error('n_select must be a positive integer.');
    end

    n_select = min(n_select,N);

    if n_select == N
        indices = (1:N).';
        return;
    end

    old_rng = rng;
    cleanup = onCleanup(@() rng(old_rng)); %#ok<NASGU>

    rng(seed,'twister');

    indices = randperm(N,n_select).';
    indices = sort(indices);
end
