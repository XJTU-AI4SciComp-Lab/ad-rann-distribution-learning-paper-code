function batch = sample_diffusion_reaction_rows( ...
    data_source,num_rows,seed,include_solution,points_per_function)
%SAMPLE_DIFFUSION_REACTION_ROWS Stratified sampling of flattened functions.
%
% Each input function occupies one contiguous coordinate block, and its
% branch sensor vector is repeated on every row of that block.  Reading
% one block at a time satisfies matfile's contiguous-indexing restriction.

    if nargin < 4
        include_solution = true;
    end

    if nargin < 5 || isempty(points_per_function)
        error('points_per_function is required.');
    end

    source_is_memory = isstruct(data_source);

    if source_is_memory
        if ~isfield(data_source,'f') || ~isfield(data_source,'y')
            error('In-memory data source must contain f and y.');
        end

        total_rows = size(data_source.f,1);
        num_sensors = size(data_source.f,2);

        if include_solution && ~isfield(data_source,'u')
            error('In-memory data source does not contain u.');
        end
    else
        info = whos('-file',data_source,'f');

        if isempty(info)
            error('Variable f not found in %s.',data_source);
        end

        total_rows = info.size(1);
        num_sensors = info.size(2);
        source = matfile(data_source);
    end

    if num_rows > total_rows
        error('Requested rows exceed the dataset size.');
    end

    if mod(total_rows,points_per_function) ~= 0
        error('Dataset rows are not divisible by points_per_function.');
    end

    num_functions = total_rows/points_per_function;
    stream = RandStream('mt19937ar','Seed',seed);

    F = zeros(num_rows,num_sensors);
    Y = zeros(num_rows,2);

    if include_solution
        u = zeros(num_rows,1);
    else
        u = [];
    end

    indices = zeros(num_rows,1);

    base_count = floor(num_rows/num_functions);
    extra_count = mod(num_rows,num_functions);
    counts = base_count*ones(num_functions,1);

    function_order = randperm(stream,num_functions);
    counts(function_order(1:extra_count)) = ...
        counts(function_order(1:extra_count))+1;

    cursor = 0;

    for function_id = 1:num_functions

        count = counts(function_id);

        if count == 0
            continue;
        end

        block_first = ...
            (function_id-1)*points_per_function+1;
        block_rows = ...
            block_first:(block_first+points_per_function-1);

        local_indices = sort( ...
            randperm(stream,points_per_function,count));
        output_rows = cursor+(1:count);
        selected_rows = block_first+local_indices-1;

        if source_is_memory
            branch_row = double(data_source.f(block_first,:));
            Yblock = double(data_source.y(block_rows,:));
        else
            branch_row = double(source.f(block_first,:));
            Yblock = double(source.y(block_rows,:));
        end

        F(output_rows,:) = repmat(branch_row,count,1);
        Y(output_rows,:) = Yblock(local_indices,:);
        indices(output_rows) = selected_rows;

        if include_solution
            if source_is_memory
                ublock = double(data_source.u(block_rows,1));
            else
                ublock = double(source.u(block_rows,1));
            end
            u(output_rows) = ublock(local_indices);
        end

        cursor = cursor+count;
    end

    batch = struct();
    batch.F = F;
    batch.Y = Y;
    batch.u = u;
    batch.indices = indices;
    batch.num_rows = num_rows;
    batch.num_functions = num_functions;
    batch.points_per_function = points_per_function;
end
