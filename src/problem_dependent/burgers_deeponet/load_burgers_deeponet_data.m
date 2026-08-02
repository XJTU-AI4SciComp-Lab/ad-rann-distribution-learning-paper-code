function data = load_burgers_deeponet_data(project_root,cfg)
%LOAD_BURGERS_DEEPONET_DATA Portable loader for the supplied Burgers data.

    data_dir = fullfile(project_root,'data',cfg.data.folder);

    files = struct();
    files.initial = fullfile(data_dir,cfg.data.initial_file);
    files.boundary = fullfile(data_dir,cfg.data.boundary_file);
    files.interior = fullfile(data_dir,cfg.data.interior_file);
    files.solution = fullfile(data_dir,cfg.data.solution_file);

    names = fieldnames(files);

    for k = 1:numel(names)
        if exist(files.(names{k}),'file') ~= 2
            error('Required Burgers data file not found: %s', ...
                files.(names{k}));
        end
    end

    initial = load(files.initial,'f','y','u');
    boundary = load(files.boundary,'y','u');
    interior = load(files.interior,'y','u');
    solution = load(files.solution,'output');

    num_functions = size(initial.f,1)/cfg.data.initial_points_per_function;

    if num_functions ~= floor(num_functions)
        error('ic.mat row count is not divisible by the IC block size.');
    end

    last_test_index = cfg.data.test_start_index+ ...
        cfg.data.num_test_functions-1;

    if num_functions < max(cfg.data.num_training_functions,last_test_index)
        error('The Burgers data do not contain enough realizations.');
    end

    first_rows = 1:cfg.data.initial_points_per_function:size(initial.f,1);
    branch_inputs = initial.f(first_rows,:);

    if size(branch_inputs,2) ~= cfg.model.num_sensors
        error('Expected %d branch sensors, found %d.', ...
            cfg.model.num_sensors,size(branch_inputs,2));
    end

    data = struct();
    data.files = files;
    data.branch_inputs = branch_inputs;
    data.initial.Y = initial.y;
    data.initial.u = initial.u(:);
    data.boundary.Y = boundary.y;
    data.boundary.u = boundary.u(:);
    data.interior.Y = interior.y;
    data.interior.u = interior.u(:);
    data.output = solution.output;
    data.num_functions = num_functions;

    if cfg.verbose
        fprintf('Burgers data directory: %s\n',data_dir);
        fprintf('Realizations: %d training + %d test\n', ...
            cfg.data.num_training_functions, ...
            cfg.data.num_test_functions);
        fprintf('Branch sensors: %d\n',size(branch_inputs,2));
    end
end
