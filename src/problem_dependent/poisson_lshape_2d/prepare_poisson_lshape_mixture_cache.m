function cache = prepare_poisson_lshape_mixture_cache(problem,basis)
%PREPARE_POISSON_LSHAPE_MIXTURE_CACHE Frozen factors for both feature blocks.

    cache.near = prepare_poisson_cache(problem,basis.near);
    cache.far = prepare_poisson_cache(problem,basis.far);
    cache.y = problem.y;
    cache.num_rows = numel(problem.y);
    cache.num_near = basis.num_near;
    cache.num_far = basis.num_far;
    cache.radius_split = basis.radius_split;
end
