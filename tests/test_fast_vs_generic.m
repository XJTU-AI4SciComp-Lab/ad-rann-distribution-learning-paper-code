function test_fast_vs_generic()
%TEST_FAST_VS_GENERIC Compare fast and generic analytic implementations.

    cfg = config();

    cfg.num_features = 40;
    cfg.interior_grid = [8 9];
    cfg.boundary_points_per_side = 10;
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

    p = [4.2;8.1];

    ls_opts = cfg.linear_solver;
    ls_opts.compute_spectrum = false;

    [M,y_rhs,dM] = build_system(p,problem,basis);

    [obj_g,grad_g,~] = ...
        reduced_objective_gradient( ...
            M,y_rhs,dM,cfg.lambda,ls_opts);

    cache = prepare_poisson_cache(problem,basis);

    [obj_f,grad_f,~] = ...
        evaluate_poisson_reduced_fast( ...
            p,cache,cfg.lambda,ls_opts);

    rel_obj = abs(obj_f-obj_g)/max(1,abs(obj_g));
    rel_grad = norm(grad_f-grad_g)/max(1,norm(grad_g));

    fprintf( ...
        'fast-vs-generic: rel obj = %.3e, rel grad = %.3e\n', ...
        rel_obj,rel_grad);

    assert(rel_obj < 1e-11);
    assert(rel_grad < 1e-10);
end
