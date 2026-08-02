function data = load_diffusion_reaction_deeponet_data(project_root,cfg)
%LOAD_DIFFUSION_REACTION_DEEPONET_DATA Resolve portable project data paths.

    train_file = fullfile( ...
        project_root,'data',cfg.data.training_file);
    test_file = fullfile( ...
        project_root,'data',cfg.data.test_file);

    validate_file(train_file,{'f','y','u'});
    validate_file(test_file,{'f','y','u'});

    train_size = variable_size(train_file,'f');
    train_y_size = variable_size(train_file,'y');
    train_u_size = variable_size(train_file,'u');

    test_size = variable_size(test_file,'f');
    test_y_size = variable_size(test_file,'y');
    test_u_size = variable_size(test_file,'u');

    if train_size(1) ~= train_y_size(1) || ...
       train_size(1) ~= train_u_size(1)
        error('Training f, y, and u must have equal row counts.');
    end

    if test_size(1) ~= test_y_size(1) || ...
       test_size(1) ~= test_u_size(1)
        error('Test f, y, and u must have equal row counts.');
    end

    if train_size(2) ~= cfg.model.num_sensors || ...
       test_size(2) ~= cfg.model.num_sensors
        error('Dataset sensor count does not match cfg.model.num_sensors.');
    end

    if train_y_size(2) ~= 2 || test_y_size(2) ~= 2
        error('Dataset y variables must have columns [x,t].');
    end

    if mod(train_size(1),cfg.data.points_per_function) ~= 0
        error(['Training rows are not divisible by ', ...
            'cfg.data.points_per_function.']);
    end

    data = struct();
    data.train_file = train_file;
    data.test_file = test_file;
    data.num_train_rows = train_size(1);
    data.num_test_rows = test_size(1);
    data.num_sensors = train_size(2);
    data.num_train_functions = ...
        train_size(1)/cfg.data.points_per_function;
    data.train_supports_partial = supports_partial_loading(train_file);
    data.test_supports_partial = supports_partial_loading(test_file);

    fprintf('Training data: %s\n',train_file);
    fprintf('Test data:     %s\n',test_file);
    fprintf('Training rows / sensors: %d / %d\n', ...
        data.num_train_rows,data.num_sensors);
    fprintf('Training functions / points each: %d / %d\n', ...
        data.num_train_functions,cfg.data.points_per_function);
    fprintf('Test rows: %d\n',data.num_test_rows);
    fprintf('Training/test partial loading: %d / %d\n', ...
        data.train_supports_partial,data.test_supports_partial);
end


function tf = supports_partial_loading(filename)
% MATLAB v7.3 MAT files use HDF5 and support efficient matfile indexing.

    fid = fopen(filename,'r');

    if fid < 0
        error('Could not open data file: %s',filename);
    end

    cleanup = onCleanup(@() fclose(fid));
    header = fread(fid,128,'*uint8').';

    hdf5_signature = uint8([137,72,68,70,13,10,26,10]);
    is_hdf5 = numel(header) >= 8 && ...
        isequal(header(1:8),hdf5_signature);
    is_mat73 = contains(char(header),'MATLAB 7.3 MAT-file');

    tf = is_hdf5 || is_mat73;
end


function validate_file(filename,required_variables)

    if exist(filename,'file') ~= 2
        error('Required data file not found: %s',filename);
    end

    info = whos('-file',filename);
    names = {info.name};

    for k = 1:numel(required_variables)
        if ~ismember(required_variables{k},names)
            error('File %s does not contain variable %s.', ...
                filename,required_variables{k});
        end
    end
end


function sz = variable_size(filename,name)

    info = whos('-file',filename,name);

    if isempty(info)
        error('Variable %s not found in %s.',name,filename);
    end

    sz = info.size;
end
