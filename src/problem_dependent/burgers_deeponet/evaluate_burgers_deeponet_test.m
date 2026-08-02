function test = evaluate_burgers_deeponet_test(data,model,cfg)
%EVALUATE_BURGERS_DEEPONET_TEST Evaluate all 100 full space-time solutions.

    nx = cfg.test.grid_shape(1);
    nt = cfg.test.grid_shape(2);
    x = linspace(0,1,nx);
    t = linspace(0,1,nt);
    [Tgrid,Xgrid] = meshgrid(t,x);
    Y = [Xgrid(:),Tgrid(:)];

    first_test = cfg.data.test_start_index;
    last_test = first_test+cfg.data.num_test_functions-1;
    test_ids = first_test:last_test;

    branch = data.branch_inputs(test_ids,:);
    B = evaluate_burgers_deeponet_branch( ...
        branch,model.scales(1),model.basis.branch,model.activation);
    trunk = evaluate_burgers_deeponet_trunk( ...
        Y,model.scales(2:4),model.basis.trunk,model.activation);

    prediction_rows = (B*model.W)*trunk';
    exact_rows = zeros(size(prediction_rows),'like',prediction_rows);

    for k = 1:numel(test_ids)
        % Stored output is time-by-space; plots use space-by-time.
        exact_field = squeeze(data.output(test_ids(k),:,:)).';
        exact_rows(k,:) = exact_field(:).';
    end

    error_rows = prediction_rows-exact_rows;
    relative_l2 = sqrt(sum(error_rows.^2,2))./ ...
        max(sqrt(sum(exact_rows.^2,2)),eps);

    [worst_error,worst_index] = max(relative_l2);

    test = struct();
    test.x = x;
    test.t = t;
    test.exact = exact_rows;
    test.prediction = prediction_rows;
    test.absolute_error = abs(error_rows);
    test.relative_l2 = relative_l2;
    test.mean_relative_l2 = mean(relative_l2);
    test.median_relative_l2 = median(relative_l2);
    test.worst_relative_l2 = worst_error;
    test.worst_index = worst_index;
    test.test_ids = test_ids;
    test.grid_shape = [nx,nt];
end
