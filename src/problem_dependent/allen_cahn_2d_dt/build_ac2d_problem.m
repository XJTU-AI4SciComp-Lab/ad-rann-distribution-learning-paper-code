function problem = build_ac2d_problem(cfg)
%BUILD_AC2D_PROBLEM Construct interior and Dirichlet boundary points.

    xa = cfg.domain(1,1);
    xb = cfg.domain(1,2);

    ya = cfg.domain(2,1);
    yb = cfg.domain(2,2);

    nx = cfg.num_collocation_x;
    ny = cfg.num_collocation_y;

    offset = 1e-6;

    x = linspace(xa+offset,xb-offset,nx);
    y = linspace(ya+offset,yb-offset,ny);

    [X,Y] = ndgrid(x,y);

    Xi = [X(:),Y(:)];

    nb = cfg.num_boundary_per_side;
    sx = linspace(xa,xb,nb).';
    sy = linspace(ya,yb,nb).';

    Xbottom = [sx,ya*ones(nb,1)];
    Xtop = [sx,yb*ones(nb,1)];

    Xleft = [xa*ones(nb,1),sy];
    Xright = [xb*ones(nb,1),sy];

    % Corners are deliberately repeated, matching the four-side penalty
    % assembly used in the historical AC2D implementation.
    Xb = [Xbottom;Xtop;Xleft;Xright];

    problem = struct();

    problem.Xi = Xi;
    problem.Xb = Xb;

    problem.Xbottom = Xbottom;
    problem.Xtop = Xtop;
    problem.Xleft = Xleft;
    problem.Xright = Xright;

    problem.u0 = ac2d_initial_condition(Xi);
    problem.boundary_values = zeros(size(Xb,1),1);

    problem.num_interior = size(Xi,1);
    problem.num_boundary = size(Xb,1);
end
