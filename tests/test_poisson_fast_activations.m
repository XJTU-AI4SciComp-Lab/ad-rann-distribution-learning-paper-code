function test_poisson_fast_activations()
%TEST_POISSON_FAST_ACTIVATIONS Tanh/sin fast path versus generic PDE path.

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

    p = [2.3;5.7];

    activations = {'tanh','sin'};

    for j = 1:numel(activations)

        activation = activations{j};

        system_builder = @(q) ...
            build_system_activation(q,problem,basis,activation);

        [obj_g,grad_g,~] = ...
            evaluate_pde_reduced_generic( ...
                p,system_builder,cfg.lambda,ls_opts);

        switch activation
            case 'tanh'
                [obj_f,grad_f,~] = ...
                    evaluate_poisson_reduced_fast_tanh( ...
                        p,cache,cfg.lambda,ls_opts);
            case 'sin'
                [obj_f,grad_f,~] = ...
                    evaluate_poisson_reduced_fast_sin( ...
                        p,cache,cfg.lambda,ls_opts);
        end

        rel_obj = abs(obj_f-obj_g)/max(1,abs(obj_g));
        rel_grad = norm(grad_f-grad_g)/max(1,norm(grad_g));

        fprintf( ...
            '%s fast-vs-generic: rel obj=%.3e, rel grad=%.3e\n', ...
            activation,rel_obj,rel_grad);

        assert(rel_obj < 1e-11);
        assert(rel_grad < 1e-10);
    end
end
