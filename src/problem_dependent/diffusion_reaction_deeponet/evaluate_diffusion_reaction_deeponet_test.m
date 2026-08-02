function metrics = evaluate_diffusion_reaction_deeponet_test( ...
    model,data,cfg)
%EVALUATE_DIFFUSION_REACTION_DEEPONET_TEST Stream test functions in batches.

    grid_shape = cfg.test.grid_shape(:).';
    points_per_function = prod(grid_shape);

    data_file = data.test_file;
    info = whos('-file',data_file,'u');
    total_rows = info.size(1);

    if mod(total_rows,points_per_function) ~= 0
        error('Test rows are not divisible by prod(cfg.test.grid_shape).');
    end

    num_functions = total_rows/points_per_function;
    if data.test_supports_partial
        source = matfile(data_file);
        memory = [];
        Ygrid = double(source.y(1:points_per_function,:));
    else
        warning([ ...
            'The test MAT file is not v7.3. Loading f, y, and u once ', ...
            'in their stored precision; convert it to v7.3 for true ', ...
            'disk-backed batching.']);
        memory = load(data_file,'f','y','u');
        source = [];
        Ygrid = double(memory.y(1:points_per_function,:));
    end
    T = evaluate_deeponet_trunk( ...
        Ygrid,model.p(2:3),model.basis.trunk,cfg.activation);

    error_sq = 0;
    truth_sq = 0;
    relative_each = nan(num_functions,1);
    first_true = [];
    first_pred = [];

    batch_size = cfg.test.function_batch_size;

    for first = 1:batch_size:num_functions

        last = min(first+batch_size-1,num_functions);
        ids = (first:last).';
        first_rows = (ids-1)*points_per_function+1;

        if data.test_supports_partial
            F = double(source.f(first_rows,:));
        else
            F = double(memory.f(first_rows,:));
        end
        B = evaluate_deeponet_branch( ...
            F,model.p(1),model.basis.branch,cfg.activation);

        U_pred = (B.value*model.W)*T.value.';

        row_first = (first-1)*points_per_function+1;
        row_last = last*points_per_function;
        if data.test_supports_partial
            u_block = double(source.u(row_first:row_last,1));
        else
            u_block = double(memory.u(row_first:row_last,1));
        end
        U_true = reshape( ...
            u_block,points_per_function,numel(ids)).';

        E = U_pred-U_true;
        error_sq = error_sq+sum(E(:).^2);
        truth_sq = truth_sq+sum(U_true(:).^2);

        numerator = sqrt(sum(E.^2,2));
        denominator = sqrt(sum(U_true.^2,2));
        relative_each(ids) = numerator./max(denominator,eps);
        plot_sample = 5;
        if plot_sample >= first && plot_sample <= last

            local_index = plot_sample-first+1;
        
            first_true = reshape( ...
                U_true(local_index,:),grid_shape);
        
            first_pred = reshape( ...
                U_pred(local_index,:),grid_shape);
        end

        if cfg.verbose
            fprintf('Test functions %d-%d / %d\n', ...
                first,last,num_functions);
        end
    end

    metrics = struct();
    metrics.overall_relative_l2 = ...
        sqrt(error_sq)/max(sqrt(truth_sq),eps);
    metrics.mean_relative_l2 = mean(relative_each);
    metrics.median_relative_l2 = median(relative_each);
    metrics.max_relative_l2 = max(relative_each);
    metrics.relative_l2_each = relative_each;
    metrics.num_functions = num_functions;
    metrics.grid_shape = grid_shape;
    metrics.Ygrid = Ygrid;
    metrics.first_true = first_true;
    metrics.first_prediction = first_pred;
    metrics.first_absolute_error = abs(first_pred-first_true);
end
