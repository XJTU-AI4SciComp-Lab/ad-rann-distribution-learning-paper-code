function training = sample_burgers_deeponet_training(data,cfg)
%SAMPLE_BURGERS_DEEPONET_TRAINING Paper-proportional supervised sampling.

    nfun = cfg.data.num_training_functions;
    q_ic = cfg.data.initial_points_per_function;
    q_bc = cfg.data.boundary_points_per_function;
    q_ir = cfg.data.interior_points_per_function;

    total_requested = cfg.training.num_rows;
    total_per_function = q_ic+q_bc+q_ir;

    category_counts = round(total_requested*[q_ic,q_bc,q_ir]/ ...
        total_per_function);
    category_counts(3) = total_requested-sum(category_counts(1:2));

    stream = RandStream('mt19937ar','Seed',cfg.training.sample_seed);

    ic_pool = (1:nfun*q_ic)';

    bc_all = (1:nfun*q_bc)';
    spatial_boundary = ...
        data.boundary.Y(bc_all,1) == 0 | ...
        data.boundary.Y(bc_all,1) == 1;
    bc_pool = bc_all(spatial_boundary);

    ir_pool = (1:nfun*q_ir)';

    selected_ic = select_rows(stream,ic_pool,category_counts(1));
    selected_bc = select_rows(stream,bc_pool,category_counts(2));
    selected_ir = select_rows(stream,ir_pool,category_counts(3));

    Y = [ ...
        data.initial.Y(selected_ic,:); ...
        data.boundary.Y(selected_bc,:); ...
        data.interior.Y(selected_ir,:)];

    target = [ ...
        data.initial.u(selected_ic); ...
        data.boundary.u(selected_bc); ...
        data.interior.u(selected_ir)];

    function_index = [ ...
        floor((selected_ic-1)/q_ic)+1; ...
        floor((selected_bc-1)/q_bc)+1; ...
        floor((selected_ir-1)/q_ir)+1];

    order = randperm(stream,total_requested);

    training = struct();
    training.branch_inputs = data.branch_inputs(1:nfun,:);
    training.Y = Y(order,:);
    training.target = target(order);
    training.function_index = function_index(order);
    training.category_counts = category_counts;
    training.num_rows = total_requested;
end


function selected = select_rows(stream,pool,count)

    if count > numel(pool)
        error('Requested %d rows from a pool containing only %d.', ...
            count,numel(pool));
    end

    order = randperm(stream,numel(pool),count);
    selected = pool(order);
end
