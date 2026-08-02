function ref = build_burgers2d_reference(cfg)
%BUILD_BURGERS2D_REFERENCE Exact fields on the fixed evaluation grid.

    n = cfg.evaluation_grid_size;

    x = linspace(cfg.domain(1,1),cfg.domain(1,2),n).';
    y = linspace(cfg.domain(2,1),cfg.domain(2,2),n).';

    [Xe,Ye] = ndgrid(x,y);
    points = [Xe(:),Ye(:)];

    tt = cfg.t_domain(1) + ...
        (1:cfg.num_saved_snapshots).' * ...
        diff(cfg.t_domain)/cfg.num_saved_snapshots;

    UU = zeros(n,n,cfg.num_saved_snapshots);

    for j = 1:cfg.num_saved_snapshots
        values = burgers2d_exact( ...
            points,tt(j),cfg.epsilon_burgers);
        UU(:,:,j) = reshape(values,n,n);
    end

    ref = struct();

    ref.Xe = Xe;
    ref.Ye = Ye;
    ref.points = points;
    ref.tt = tt;
    ref.UU = UU;
    ref.U = UU(:,:,end);
    ref.num_snapshots = cfg.num_saved_snapshots;
    ref.description = 'Analytic travelling-wave solution';
end
