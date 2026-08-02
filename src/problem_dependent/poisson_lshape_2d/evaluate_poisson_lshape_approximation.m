function evaluation = evaluate_poisson_lshape_approximation( ...
        cfg,p_base,base_basis,coef,rho,growth_basis)
%EVALUATE_POISSON_LSHAPE_APPROXIMATION L2, Linf and H1 errors in chunks.
%
% The plotting grid is built explicitly with MESHGRID. Therefore all stored
% field matrices have size ny-by-nx and can be passed directly to
% contourf(x,y,U) without a transpose. This guarantees that the removed
% quadrant [0,1] x [-1,0] is displayed in the lower-right corner.

    nx = cfg.test_grid(1);
    ny = cfg.test_grid(2);

    x = linspace( ...
        cfg.domain(1,1), ...
        cfg.domain(1,2), ...
        nx);

    y = linspace( ...
        cfg.domain(2,1), ...
        cfg.domain(2,2), ...
        ny);

    [Xgrid,Ygrid] = meshgrid(x,y);
    Xbox = [Xgrid(:),Ygrid(:)];

    valid_vector = poisson_lshape_is_inside( ...
        Xbox,true,cfg.geometry_tolerance);

    X = Xbox(valid_vector,:);

    u_exact = poisson_lshape_exact_solution(X);
    [ux_exact,uy_exact] = poisson_lshape_exact_gradient(X);

    n = size(X,1);
    chunk_rows = cfg.evaluation_chunk_rows;

    u_pred = zeros(n,1);
    ux_pred = zeros(n,1);
    uy_pred = zeros(n,1);

    m_base = size(base_basis.Z,2);
    coef_base = coef(1:m_base);

    has_growth = ~isempty(growth_basis);

    if has_growth
        coef_growth = coef(m_base+1:end);
        uses_common_growth = isfield(growth_basis,'centers') && ...
            isfield(growth_basis,'directions') && ...
            isfield(growth_basis,'center_values');

        if ~uses_common_growth
            % Backward compatibility for result files produced by the
            % former residual-centered first-layer enrichment.
            rho_vec = [rho(1);rho(1)];
        end
    else
        coef_growth = [];
        uses_common_growth = false;
        rho_vec = [];
    end

    for first = 1:chunk_rows:n

        last = min(first+chunk_rows-1,n);
        idx = first:last;
        Xc = X(idx,:);

        final_base_state = evaluate_poisson_lshape_base_state( ...
            Xc,p_base,base_basis,coef_base,1);

        u_chunk = final_base_state.u;
        ux_chunk = final_base_state.d1{1};
        uy_chunk = final_base_state.d1{2};

        if has_growth

            if uses_common_growth
                if ~isfield(growth_basis,'base_output_weights')
                    error([ ...
                        'The common L-shape growth model must store the ', ...
                        'frozen pre-growth base output weights.']);
                end

                frozen_base_state = ...
                    evaluate_poisson_lshape_base_state( ...
                        Xc,p_base,base_basis, ...
                        growth_basis.base_output_weights,1);

                growth_state = evaluate_growth_features( ...
                    Xc,growth_basis,rho,frozen_base_state,1);

                u_chunk = u_chunk + ...
                    growth_state.phi*coef_growth;
                ux_chunk = ux_chunk + ...
                    growth_state.d1{1}*coef_growth;
                uy_chunk = uy_chunk + ...
                    growth_state.d1{2}*coef_growth;
            else
                Phi_growth = activation_features( ...
                    Xc,rho_vec,growth_basis,cfg.activation);

                D_growth = feature_derivatives_2d( ...
                    Xc,rho_vec,growth_basis,cfg.activation);

                u_chunk = u_chunk + Phi_growth*coef_growth;
                ux_chunk = ux_chunk + D_growth.x*coef_growth;
                uy_chunk = uy_chunk + D_growth.y*coef_growth;
            end

        end

        u_pred(idx) = u_chunk;
        ux_pred(idx) = ux_chunk;
        uy_pred(idx) = uy_chunk;

    end

    rel_l2 = relative_l2(u_pred,u_exact);
    rel_linf = relative_linf(u_pred,u_exact);

    finite_gradient = ...
        isfinite(ux_exact) & isfinite(uy_exact);

    grad_error_sq = ...
        norm(ux_pred(finite_gradient)-ux_exact(finite_gradient))^2 + ...
        norm(uy_pred(finite_gradient)-uy_exact(finite_gradient))^2;

    grad_exact_sq = ...
        norm(ux_exact(finite_gradient))^2 + ...
        norm(uy_exact(finite_gradient))^2;

    value_error_sq = norm(u_pred-u_exact)^2;
    value_exact_sq = norm(u_exact)^2;

    relative_h1_seminorm = ...
        sqrt(grad_error_sq/max(grad_exact_sq,eps));

    relative_h1 = sqrt( ...
        (value_error_sq+grad_error_sq) / ...
        max(value_exact_sq+grad_exact_sq,eps));

    valid_matrix = reshape(valid_vector,ny,nx);

    U_exact = nan(ny,nx);
    U_pred = nan(ny,nx);
    U_error = nan(ny,nx);

    U_exact(valid_matrix) = u_exact;
    U_pred(valid_matrix) = u_pred;
    U_error(valid_matrix) = abs(u_pred-u_exact);

    evaluation = struct();

    evaluation.relative_l2 = rel_l2;
    evaluation.relative_linf = rel_linf;
    evaluation.relative_h1_seminorm = relative_h1_seminorm;
    evaluation.relative_h1 = relative_h1;

    evaluation.X = X;
    evaluation.u_exact = u_exact;
    evaluation.u_pred = u_pred;
    evaluation.ux_exact = ux_exact;
    evaluation.uy_exact = uy_exact;
    evaluation.ux_pred = ux_pred;
    evaluation.uy_pred = uy_pred;

    evaluation.grid.x = x(:);
    evaluation.grid.y = y(:);
    evaluation.grid.X = Xgrid;
    evaluation.grid.Y = Ygrid;
    evaluation.grid.valid = valid_matrix;
    evaluation.grid.U_exact = U_exact;
    evaluation.grid.U_pred = U_pred;
    evaluation.grid.U_error = U_error;

end
