function u0 = ac_initial_condition(x)
%AC_INITIAL_CONDITION Historical Allen-Cahn initial condition.
%
% The first column of the supplied AC_new.mat is
%
%   u(x,0) = x^2 cos(pi x).

    u0 = x.^2 .* cos(pi*x);
end
