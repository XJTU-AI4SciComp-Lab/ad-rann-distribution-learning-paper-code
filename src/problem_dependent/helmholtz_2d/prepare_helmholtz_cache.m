function cache = prepare_helmholtz_cache(problem,basis)
%PREPARE_HELMHOLTZ_CACHE Cache all p-independent Gaussian factors.

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

    cache.k = problem.k;
    cache.boundary_penalty = problem.boundary_penalty;
    cache.y = problem.y;

    cache.num_rows = numel(problem.y);
    cache.num_features = size(basis.Z,2);
end
