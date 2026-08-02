function u0 = burgers_initial_condition(x)
%BURGERS_INITIAL_CONDITION Initial condition u(x,0) = -sin(pi x).

    u0 = -sin(pi*x);
end
