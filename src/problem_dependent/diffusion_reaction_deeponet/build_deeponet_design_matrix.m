function A = build_deeponet_design_matrix(B,T)
%BUILD_DEEPONET_DESIGN_MATRIX Rowwise branch/trunk tensor products.
%
% With alpha=reshape(w,[kB,pT]), A*w equals
%
%   sum((B*alpha).*T,2).

    if size(B,1) ~= size(T,1)
        error('Branch and trunk features must have the same row count.');
    end

    N = size(B,1);
    kB = size(B,2);
    pT = size(T,2);

    A = zeros(N,kB*pT,'like',B);

    for j = 1:pT
        columns = (j-1)*kB+(1:kB);
        A(:,columns) = B.*T(:,j);
    end
end
