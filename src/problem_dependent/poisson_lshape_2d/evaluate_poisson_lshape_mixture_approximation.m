function evaluation = evaluate_poisson_lshape_mixture_approximation( ...
        cfg,p,basis,coef)
%EVALUATE_POISSON_LSHAPE_MIXTURE_APPROXIMATION Overall and regional errors.

    nx = cfg.test_grid(1);
    ny = cfg.test_grid(2);
    x = linspace(cfg.domain(1,1),cfg.domain(1,2),nx);
    y = linspace(cfg.domain(2,1),cfg.domain(2,2),ny);
    [Xgrid,Ygrid] = meshgrid(x,y);
    Xbox = [Xgrid(:),Ygrid(:)];
    valid_vector = poisson_lshape_is_inside( ...
        Xbox,true,cfg.geometry_tolerance);
    X = Xbox(valid_vector,:);

    u_exact = poisson_lshape_exact_solution(X);
    [ux_exact,uy_exact] = poisson_lshape_exact_gradient(X);
    u_pred = zeros(size(u_exact));
    ux_pred = zeros(size(u_exact));
    uy_pred = zeros(size(u_exact));

    cnear = coef(1:basis.num_near);
    cfar = coef(basis.num_near+1:end);

    for first = 1:cfg.evaluation_chunk_rows:size(X,1)
        idx = first:min(first+cfg.evaluation_chunk_rows-1,size(X,1));
        Dnear = feature_derivatives_2d( ...
            X(idx,:),p(1:2),basis.near,cfg.activation);
        Dfar = feature_derivatives_2d( ...
            X(idx,:),p(3:4),basis.far,cfg.activation);

        u_pred(idx) = Dnear.phi*cnear+Dfar.phi*cfar;
        ux_pred(idx) = Dnear.x*cnear+Dfar.x*cfar;
        uy_pred(idx) = Dnear.y*cnear+Dfar.y*cfar;
    end

    evaluation = common_metrics( ...
        X,u_pred,u_exact,ux_pred,uy_pred,ux_exact,uy_exact, ...
        basis.radius_split);

    valid_matrix = reshape(valid_vector,ny,nx);
    U_exact = nan(ny,nx);
    U_pred = nan(ny,nx);
    U_error = nan(ny,nx);
    U_exact(valid_matrix) = u_exact;
    U_pred(valid_matrix) = u_pred;
    U_error(valid_matrix) = abs(u_pred-u_exact);

    evaluation.X = X;
    evaluation.u_exact = u_exact;
    evaluation.u_pred = u_pred;
    evaluation.grid.x = x(:);
    evaluation.grid.y = y(:);
    evaluation.grid.X = Xgrid;
    evaluation.grid.Y = Ygrid;
    evaluation.grid.valid = valid_matrix;
    evaluation.grid.U_exact = U_exact;
    evaluation.grid.U_pred = U_pred;
    evaluation.grid.U_error = U_error;
end


function evaluation = common_metrics( ...
        X,u,ue,ux,uy,uxe,uye,radius_split)

    evaluation.relative_l2 = relative_l2(u,ue);
    evaluation.relative_linf = relative_linf(u,ue);

    finite = isfinite(uxe) & isfinite(uye);
    ge2 = norm(ux(finite)-uxe(finite))^2 + ...
        norm(uy(finite)-uye(finite))^2;
    gr2 = norm(uxe(finite))^2+norm(uye(finite))^2;
    ve2 = norm(u-ue)^2;
    vr2 = norm(ue)^2;
    evaluation.relative_h1_seminorm = sqrt(ge2/max(gr2,eps));
    evaluation.relative_h1 = sqrt((ve2+ge2)/max(vr2+gr2,eps));

    radius = vecnorm(X,2,2);
    near = radius <= radius_split;
    far = radius > radius_split;
    evaluation.near = regional_metrics(u,ue,near);
    evaluation.far = regional_metrics(u,ue,far);
end


function region = regional_metrics(u,ue,mask)
    region.num_points = nnz(mask);
    region.relative_l2 = relative_l2(u(mask),ue(mask));
    region.relative_linf = relative_linf(u(mask),ue(mask));
    region.absolute_linf = norm(u(mask)-ue(mask),inf);
end
