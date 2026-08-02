function cache = prepare_poisson_cache(problem, basis)
%PREPARE_POISSON_CACHE Precompute all p-independent Gaussian factors.

    Z1 = basis.Z(1,:);
    Z2 = basis.Z(2,:);

    C1 = basis.C(1,:);
    C2 = basis.C(2,:);

    cache.Qi1 = (problem.Xi(:,1)-C1).*Z1;
    cache.Qi2 = (problem.Xi(:,2)-C2).*Z2;

    cache.Qb1 = (problem.Xb(:,1)-C1).*Z1;
    cache.Qb2 = (problem.Xb(:,2)-C2).*Z2;

    cache.Z1sq = Z1.^2;
    cache.Z2sq = Z2.^2;

    cache.eta = problem.boundary_penalty;

    if isfield(problem,'y')
        cache.y = problem.y;
    else
        cache.y = [problem.fi;cache.eta*problem.gb];
    end

    cache.num_rows = numel(cache.y);
    cache.num_features = size(basis.Z,2);
end
