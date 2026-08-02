function u = exact_solution(X)
%EXACT_SOLUTION u(x,y)=sin(pi*x)sin(5*pi*y).

    x = X(:,1);
    y = X(:,2);

    u = sin(pi*x).*sin(5*pi*y);
end
