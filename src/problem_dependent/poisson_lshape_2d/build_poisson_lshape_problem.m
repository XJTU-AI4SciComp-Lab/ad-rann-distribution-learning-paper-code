function problem = build_poisson_lshape_problem(cfg)
%BUILD_POISSON_LSHAPE_PROBLEM Collocation points and Dirichlet data.

    Xbox = tensor_grid( ...
        cfg.domain,cfg.interior_grid,cfg.interior_offset);

    interior_mask = poisson_lshape_is_inside( ...
        Xbox,false,cfg.geometry_tolerance);

    Xi = Xbox(interior_mask,:);
    fi = poisson_lshape_rhs(Xi);

    n = cfg.boundary_intervals_per_unit;

    s_full = linspace(-1,1,2*n+1).';
    s_minus = linspace(-1,0,n+1).';
    s_plus = linspace(0,1,n+1).';

    Xb = [ ...
        -ones(size(s_full)), s_full; ...      % x=-1
        s_full, ones(size(s_full)); ...       % y=1
        ones(size(s_plus)), s_plus; ...       % x=1, y in [0,1]
        s_minus, -ones(size(s_minus)); ...    % y=-1, x in [-1,0]
        zeros(size(s_minus)), s_minus; ...    % x=0, y in [-1,0]
        s_plus, zeros(size(s_plus))];          % y=0, x in [0,1]

    Xb = unique(Xb,'rows','stable');

    gb = poisson_lshape_exact_solution(Xb);

    problem = struct();
    problem.domain = cfg.domain;
    problem.Xi = Xi;
    problem.fi = fi;
    problem.Xb = Xb;
    problem.gb = gb;
    problem.boundary_penalty = cfg.boundary_penalty;

    problem.y = [ ...
        fi; ...
        cfg.boundary_penalty*gb];

    problem.num_interior = size(Xi,1);
    problem.num_boundary = size(Xb,1);
end
