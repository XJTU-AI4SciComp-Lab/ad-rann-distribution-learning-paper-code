function problem = build_helmholtz_problem(cfg)
%BUILD_HELMHOLTZ_PROBLEM Construct collocation and boundary data.

    problem = struct();

    problem.domain = cfg.domain;
    problem.k = cfg.k;
    problem.a1 = cfg.a1;
    problem.a2 = cfg.a2;
    problem.boundary_penalty = cfg.boundary_penalty;

    problem.Xi = tensor_grid( ...
        cfg.domain,cfg.interior_grid,cfg.interior_inset);

    problem.fi = helmholtz_rhs(problem.Xi,problem);

    nB = cfg.boundary_points_per_side;

    x = linspace(cfg.domain(1,1),cfg.domain(1,2),nB).';
    y = linspace(cfg.domain(2,1),cfg.domain(2,2),nB).';

    problem.Xb = [ ...
        cfg.domain(1,1)*ones(nB,1), y; ...
        cfg.domain(1,2)*ones(nB,1), y; ...
        x, cfg.domain(2,1)*ones(nB,1); ...
        x, cfg.domain(2,2)*ones(nB,1)];

    problem.gb = helmholtz_exact_solution(problem.Xb,problem);

    problem.y = [ ...
        problem.fi; ...
        cfg.boundary_penalty*problem.gb];
end
