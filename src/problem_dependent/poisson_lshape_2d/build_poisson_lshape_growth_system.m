function [M,y,parts] = build_poisson_lshape_growth_system( ...
        problem,M_base,y_rhs,model,rho,base_i,base_b)
%BUILD_POISSON_LSHAPE_GROWTH_SYSTEM Add the paper-consistent local block.
%
% The PDE-independent layer-growth features and their derivatives are
% evaluated by src/layer_growth/evaluate_growth_features.m. This wrapper
% only applies the Poisson operator and the boundary penalty.

    growth_i = evaluate_growth_features( ...
        problem.Xi,model,rho,base_i,2);

    growth_b = evaluate_growth_features( ...
        problem.Xb,model,rho,base_b,0);

    Mg_i = -(growth_i.d2{1}+growth_i.d2{2});
    Mg_b = problem.boundary_penalty*growth_b.phi;
    Mg = [Mg_i;Mg_b];

    M = [M_base,Mg];
    y = y_rhs;

    parts = struct();
    parts.Mg_i = Mg_i;
    parts.Mg_b = Mg_b;
    parts.Phi_i = growth_i.phi;
    parts.Phi_b = growth_b.phi;
end
