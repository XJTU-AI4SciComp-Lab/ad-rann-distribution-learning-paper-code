function [x, info] = solve_least_squares(A, b, opts)
%SOLVE_LEAST_SQUARES Solve min_x ||A*x-b||_2.

    if nargin < 3 || isempty(opts)
        opts = struct();
    end

    method = lower(char(get_opt(opts,'method','linsolve')));
    use_gpu = logical(get_opt(opts,'use_gpu',false));

    if isvector(b)
        b = b(:);
    end

    if size(A,1) ~= size(b,1)
        error('solve_least_squares:SizeMismatch', ...
            'A and b must have the same number of rows.');
    end

    total_timer = tic;

    if use_gpu

        if ~canUseGPU
            error('solve_least_squares:GPUUnavailable', ...
                'GPU was requested, but MATLAB cannot use a supported GPU.');
        end

        t = tic;
        Ag = gpuArray(A);
        bg = gpuArray(b);
        transfer_to_gpu_time = toc(t);

        t = tic;
        xg = Ag\bg;
        wait(gpuDevice);
        solve_time = toc(t);

        t = tic;
        x = gather(xg);
        gather_time = toc(t);

        solver_used = 'gpu-mldivide';

        clear Ag bg xg

    else

        transfer_to_gpu_time = 0;
        gather_time = 0;

        t = tic;

        switch method
            case 'linsolve'
                ls_opts.RECT = true;
                x = linsolve(A,b,ls_opts);
                solver_used = 'cpu-linsolve';

            case 'mldivide'
                x = A\b;
                solver_used = 'cpu-mldivide';

            otherwise
                error('solve_least_squares:UnknownMethod', ...
                    'Unknown least-squares method: %s',method);
        end

        solve_time = toc(t);
    end

    info.method = solver_used;
    info.use_gpu = use_gpu;
    info.transfer_to_gpu_time = transfer_to_gpu_time;
    info.solve_time = solve_time;
    info.gather_time = gather_time;
    info.total_time = toc(total_timer);
    info.residual_norm = norm(A*x-b,'fro');
end


function value = get_opt(opts,name,default_value)

    if isfield(opts,name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default_value;
    end
end
