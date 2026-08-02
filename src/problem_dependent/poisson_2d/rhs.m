function f = rhs(X)
%RHS Right-hand side of -Delta u = f.

    x = X(:,1);
    y = X(:,2);

    f = (pi^2 + (5*pi)^2).*sin(pi*x).*sin(5*pi*y);
end
