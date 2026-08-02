function op = prepare_ac2d_base_operator(p,problem,basis)
%PREPARE_AC2D_BASE_OPERATOR First-layer matrices at fixed p.

    st_i = evaluate_ac2d_base_state( ...
        problem.Xi,p,basis,[],2);

    st_b = evaluate_ac2d_base_state( ...
        problem.Xb,p,basis,[],0);

    op = struct();

    op.p = p(:);

    op.Phi_i = st_i.phi;
    op.Lap_i = st_i.lap;
    op.Phi_b = st_b.phi;
end
