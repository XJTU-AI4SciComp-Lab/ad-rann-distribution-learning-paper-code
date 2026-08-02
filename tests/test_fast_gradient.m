function test_fast_gradient()
%TEST_FAST_GRADIENT Analytic gradient versus centered finite difference.

    cfg = config();

    cfg.num_features = 35;
    cfg.interior_grid = [7 8];
    cfg.boundary_points_per_side = 8;
    cfg.linear_solver.use_gpu = false;
    cfg.compute_spectrum = false;

    basis = build_random_weights( ...
        cfg.num_features,cfg.domain,cfg.seed);

    problem.domain = cfg.domain;
    problem.boundary_penalty = cfg.boundary_penalty;

    problem.Xi = tensor_grid( ...
        cfg.domain,cfg.interior_grid,1e-6);

    problem.fi = rhs(problem.Xi);

    nB = cfg.boundary_points_per_side;

    x = linspace(cfg.domain(1,1),cfg.domain(1,2),nB)';
    y = linspace(cfg.domain(2,1),cfg.domain(2,2),nB)';

    problem.Xb = [ ...
        cfg.domain(1,1)*ones(nB,1), y; ...
        cfg.domain(1,2)*ones(nB,1), y; ...
        x, cfg.domain(2,1)*ones(nB,1); ...
        x, cfg.domain(2,2)*ones(nB,1)];

    problem.gb = exact_solution(problem.Xb);

    problem.y = [ ...
        problem.fi; ...
        cfg.boundary_penalty*problem.gb];

    cache = prepare_poisson_cache(problem,basis);

    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    p = [4.3;7.8];

    [obj,grad,~] = ...
        evaluate_poisson_reduced_fast( ...
            p,cache,cfg.lambda,ls_opts);

    h = 1e-6;
    grad_fd = zeros(2,1);

    for k = 1:2

        pp = p;
        pm = p;

        pp(k) = pp(k)+h;
        pm(k) = pm(k)-h;

        fp = evaluate_poisson_reduced_fast( ...
            pp,cache,cfg.lambda,ls_opts);

        fm = evaluate_poisson_reduced_fast( ...
            pm,cache,cfg.lambda,ls_opts);

        grad_fd(k) = (fp-fm)/(2*h);
    end

    rel = norm(grad-grad_fd)/max(1,norm(grad_fd));

    fprintf( ...
        'analytic gradient test: obj=%.3e, relative diff=%.3e\n', ...
        obj,rel);

    assert(rel < 5e-6);
end
