function X = tensor_grid(domain, n, inset)
%TENSOR_GRID Tensor-product grid on a rectangular 2-D domain.

    if nargin < 3
        inset = 0;
    end

    if numel(n) == 1
        n = [n n];
    end

    x = linspace(domain(1,1)+inset,domain(1,2)-inset,n(1));
    y = linspace(domain(2,1)+inset,domain(2,2)-inset,n(2));

    [Xg,Yg] = meshgrid(x,y);

    X = [Xg(:),Yg(:)];
end
