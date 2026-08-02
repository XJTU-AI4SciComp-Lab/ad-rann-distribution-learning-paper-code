function factor = ac2d_factorize_ls(M,ls_opts)
%AC2D_FACTORIZE_LS Cache an economy QR factorization of a tall matrix.

    if nargin < 2
        ls_opts = struct();
    end

    factor = struct();

    use_cache = ...
        ~isfield(ls_opts,'cache_qr') || ...
        logical(ls_opts.cache_qr);

    use_gpu = ...
        isfield(ls_opts,'use_gpu') && ...
        logical(ls_opts.use_gpu);

    if use_gpu || ~use_cache

        factor.mode = 'project_solver';
        factor.M = M;
        factor.ls_opts = ls_opts;
        return;
    end

    [Q,R] = qr(M,0);

    d = abs(diag(R));

    if isempty(d)
        error('The least-squares matrix has no columns.');
    end

    if isfield(ls_opts,'rank_tolerance')
        tol = ls_opts.rank_tolerance;
    else
        tol = 1e-12;
    end

    rank_estimate = nnz(d > tol*max(d));

    factor.rank_estimate = rank_estimate;
    factor.num_columns = size(M,2);

    if rank_estimate < size(M,2)

        warning([ ...
            'Cached QR detected numerical rank %d < %d. ', ...
            'Falling back to the project least-squares solver.'], ...
            rank_estimate,size(M,2));

        factor.mode = 'project_solver';
        factor.M = M;
        factor.ls_opts = ls_opts;

    else

        factor.mode = 'qr';
        factor.Q = Q;
        factor.R = R;
    end
end
