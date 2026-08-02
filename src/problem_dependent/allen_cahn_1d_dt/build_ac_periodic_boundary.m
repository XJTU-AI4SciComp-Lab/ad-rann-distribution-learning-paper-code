function B = build_ac_periodic_boundary(Xb,p,basis,eta)
%BUILD_AC_PERIODIC_BOUNDARY Periodic value/derivative constraint matrix.
%
% Rows:
%
%   eta [phi(-1)-phi(1)]
%   eta [phi_x(-1)-phi_x(1)]

    if numel(Xb) ~= 2
        error('Xb must contain exactly the left and right endpoints.');
    end

    st = evaluate_ac_state(Xb(:),p,basis);

    B = eta*[ ...
        st.phi(1,:)-st.phi(2,:); ...
        st.ux(1,:)-st.ux(2,:)];
end
