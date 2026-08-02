function state = evaluate_gaussian_state( ...
    X,p,basis,coef,max_order,chunk_rows)
%EVALUATE_GAUSSIAN_STATE Evaluate Gaussian randomized-feature solution.
%
% state.u
% state.d1{k}
% state.d2{k}
%
% Only scalar solution/derivative vectors are returned.  Full derivative
% feature matrices are not stored.

    if nargin < 5 || isempty(max_order)
        max_order = 1;
    end

    if nargin < 6 || isempty(chunk_rows)
        chunk_rows = 1000;
    end

    if ~ismember(max_order,[0,1,2])
        error('max_order must be 0, 1, or 2.');
    end

    p = p(:);
    coef = coef(:);

    N = size(X,1);
    d = numel(p);
    m = size(basis.Z,2);

    if numel(coef) ~= m
        error('coef length must equal the number of base features.');
    end

    state = struct();
    state.u = zeros(N,1);
    state.d1 = cell(d,1);
    state.d2 = cell(d,1);

    if max_order >= 1
        for k = 1:d
            state.d1{k} = zeros(N,1);
        end
    end

    if max_order >= 2
        for k = 1:d
            state.d2{k} = zeros(N,1);
        end
    end

    for first = 1:chunk_rows:N

        rows = first:min(first+chunk_rows-1,N);
        Xc = X(rows,:);

        z = zeros(numel(rows),m);

        for k = 1:d
            z = z + ...
                (Xc(:,k)-basis.C(k,:)).* ...
                (p(k)*basis.Z(k,:));
        end

        phi = exp(-(z.^2));

        state.u(rows) = phi*coef;

        if max_order >= 1

            common1 = -2*z.*phi;

            for k = 1:d
                wk = p(k)*basis.Z(k,:);

                state.d1{k}(rows) = ...
                    (common1.*wk)*coef;
            end
        end

        if max_order >= 2

            common2 = (4*z.^2-2).*phi;

            for k = 1:d
                wk2 = (p(k)*basis.Z(k,:)).^2;

                state.d2{k}(rows) = ...
                    (common2.*wk2)*coef;
            end
        end
    end
end
