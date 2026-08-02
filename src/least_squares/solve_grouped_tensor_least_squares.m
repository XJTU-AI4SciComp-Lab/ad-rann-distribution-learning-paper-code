function [W,info] = solve_grouped_tensor_least_squares( ...
    B,T,group_index,target,lambda,opts)
%SOLVE_GROUPED_TENSOR_LEAST_SQUARES Matrix-free DeepONet coefficient fit.
%
% For row i, the design row is kron(T(i,:),B(group_index(i),:)).
% The design matrix is never formed explicitly.

    if nargin < 5 || isempty(lambda)
        lambda = 0;
    end

    if nargin < 6
        opts = struct();
    end

    tol = get_option(opts,'tolerance',1e-8);
    maxit = get_option(opts,'max_iterations',200);
    verbose = get_option(opts,'verbose',false);

    target = target(:);
    group_index = group_index(:);

    num_rows = size(T,1);
    num_groups = size(B,1);
    num_branch = size(B,2);
    num_trunk = size(T,2);
    num_coefficients = num_branch*num_trunk;

    if numel(target) ~= num_rows || numel(group_index) ~= num_rows
        error('T, group_index, and target must have the same row count.');
    end

    if any(group_index < 1) || any(group_index > num_groups) || ...
            any(group_index ~= floor(group_index))
        error('group_index must contain integer indices into the rows of B.');
    end

    if ~isscalar(lambda) || ~isfinite(lambda) || lambda < 0
        error('lambda must be a finite nonnegative scalar.');
    end

    group_sum = sparse( ...
        group_index,(1:num_rows)',1,num_groups,num_rows);

    if lambda > 0
        rhs = [target;zeros(num_coefficients,1,'like',target)];
    else
        rhs = target;
    end

    timer = tic;
    [w,flag,relres,iter,resvec] = lsqr( ...
        @apply_operator,rhs,tol,maxit);
    elapsed = toc(timer);

    W = reshape(w,num_branch,num_trunk);
    fitted = apply_prediction(W);
    residual = fitted-target;

    info = struct();
    info.method = 'matrix-free grouped LSQR';
    info.flag = flag;
    info.relative_residual = relres;
    info.iterations = iter;
    info.residual_history = resvec;
    info.residual_mse = mean(residual.^2);
    info.relative_l2 = norm(residual)/max(norm(target),eps);
    info.elapsed_time = elapsed;
    info.num_rows = num_rows;
    info.num_coefficients = num_coefficients;
    info.explicit_design_bytes_avoided = ...
        8*double(num_rows)*double(num_coefficients);

    if verbose
        fprintf(['Grouped LSQR: flag=%d, iter=%d, relres=%.3e, ', ...
            'residual MSE=%.3e\n'], ...
            flag,iter,relres,info.residual_mse);
    end


    function z = apply_operator(v,transp_flag)

        if strcmp(transp_flag,'notransp')
            coefficient_matrix = reshape( ...
                v(1:num_coefficients),num_branch,num_trunk);
            z_data = apply_prediction(coefficient_matrix);

            if lambda > 0
                z = [z_data;sqrt(lambda)*v(1:num_coefficients)];
            else
                z = z_data;
            end

        elseif strcmp(transp_flag,'transp')
            data_part = v(1:num_rows);
            grouped_trunk = group_sum*(data_part.*T);
            z = reshape(B'*grouped_trunk,num_coefficients,1);

            if lambda > 0
                z = z+sqrt(lambda)*v(num_rows+1:end);
            end

        else
            error('Unsupported transpose flag: %s',transp_flag);
        end
    end


    function prediction = apply_prediction(coefficient_matrix)

        grouped_coefficients = B*coefficient_matrix;
        prediction = sum( ...
            T.*grouped_coefficients(group_index,:),2);
    end
end


function value = get_option(opts,name,default_value)

    if isfield(opts,name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default_value;
    end
end
