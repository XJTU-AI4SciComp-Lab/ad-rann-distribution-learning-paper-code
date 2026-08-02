function [M,y,dM] = build_data_system(p,cache,activation)
%BUILD_DATA_SYSTEM Generic data-driven least-squares system.
%
%   [M,y] = build_data_system(p,cache,activation)
%   [M,y,dM] = build_data_system(p,cache,activation)
%
% This routine is independent of the input dimension. Full dM matrices are
% created only when the third output is requested.

    if nargin < 3 || isempty(activation)
        activation = 'gaussian';
    end

    p = p(:);
    d = cache.dim;

    if numel(p) ~= d
        error('numel(p) must equal cache.dim.');
    end

    N = cache.num_rows;
    m = cache.num_features;

    S = zeros(N,m,'like',cache.Q{1});

    for k = 1:d
        S = S + p(k)*cache.Q{k};
    end

    M = evaluate_activation(S,activation);
    y = cache.y;

    if nargout >= 3
        A = activation_derivatives(S,activation,1);

        dM = cell(d,1);

        for k = 1:d
            dM{k} = A.d1.*cache.Q{k};
        end
    end
end
