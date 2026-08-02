function [cfg,project_root] = config_burgers_deeponet()
%CONFIG_BURGERS_DEEPONET Paper-aligned data-driven Burgers configuration.

    project_root = initialize_paths(mfilename('fullpath'));

    cfg = struct();
    cfg.seed = 42;
    cfg.activation = 'tanh';
    cfg.verbose = true;

    cfg.data.folder = 'burgers_deeponet';
    cfg.data.initial_file = 'ic.mat';
    cfg.data.boundary_file = 'bc.mat';
    cfg.data.interior_file = 'ir.mat';
    cfg.data.solution_file = 'Burger_t.mat';
    cfg.data.num_test_functions = 100;
    cfg.data.test_start_index = 1001;
    cfg.data.initial_points_per_function = 101;
    cfg.data.boundary_points_per_function = 100;
    cfg.data.interior_points_per_function = 2500;

    
    cfg.data.num_training_functions = 1000;
    cfg.training.num_rows = 86432;


    % Final data-driven RaNN-DeepONet model.
    cfg.model.num_sensors = 101;
    cfg.model.num_branch = 200;
    cfg.model.num_trunk = 120;

    %cfg.data.num_training_functions = 500;
   cfg.training.sample_seed = 42;
   cfg.training.ridge_lambda = 0;

    % Reduced randomized model used only for DDAD.
    cfg.reduced.num_branch = 100;
    cfg.reduced.num_trunk = 60;
    cfg.reduced.num_rows = 21608;
    cfg.reduced.sample_seed = 42;

    % DDAD optimizes only [rb;rx;rt].  The periodic cosine and sine
    % coordinates share rx, so there are three independent parameters.
    cfg.ddad.initial_p = [1;1;1];
    cfg.ddad.ridge_lambda = 1e-7;
    cfg.ddad.optimizer = ddad_optimizer();
    cfg.ddad.linear_solver.tolerance = 1e-7;
    cfg.ddad.linear_solver.max_iterations = 80;
    cfg.ddad.linear_solver.verbose = false;

    % Final explicit, direct, unregularized least-squares refit.
    cfg.linear_solver.tolerance = 1e-14;
    cfg.linear_solver.max_iterations = 6000;
    cfg.linear_solver.verbose = true;

    cfg.test.grid_shape = [101,101];

    % Plot the middle test realization.
    cfg.plot.sample = 50;
    cfg.plot.num_levels = 30;
    cfg.plot.font_size = 12;
    cfg.plot.window_position = [100,100,560,450];
    cfg.plot.png_resolution = 600;

    cfg.output_root_name = fullfile('results','burgers_deeponet');
end


function opt = ddad_optimizer()

    opt = struct();
    opt.maxit = 70;
    opt.learning_rate = 0.2*[1;1;1];
    opt.beta1 = 0.9;
    opt.beta2 = 0.999;
    opt.epsilon = 1e-8;

    opt.lower_bound = [1e-3;1e-3;1e-3];
    opt.upper_bound = [300;300;300];
    opt.parameterization = 'log';
    opt.selection_metric = 'residual_mse';

    opt.grad_tol = 1e-10;
    opt.step_tol = 1e-11;
    opt.relative_obj_tol = 1e-11;
    opt.patience = Inf;
    opt.min_delta = 0;

    opt.store_moments = true;
    opt.store_full_info = false;
    opt.verbose = true;
end


function project_root = initialize_paths(config_file)

    example_dir = fileparts(config_file);
    project_root = fileparts(fileparts(example_dir));

    required_dirs = { ...
        fullfile(project_root,'src'), ...
        fullfile(project_root,'data'), ...
        fullfile(project_root,'examples')};

    for k = 1:numel(required_dirs)
        if exist(required_dirs{k},'dir') ~= 7
            error('Required project directory not found: %s', ...
                required_dirs{k});
        end
    end

    addpath(genpath(fullfile(project_root,'src')));
    rehash path;

    required_functions = { ...
    'build_random_weights_nd', ...
    'activation_features', ...
    'build_deeponet_design_matrix', ...
    'solve_least_squares', ...
    'solve_grouped_tensor_least_squares', ...
    'optimize_distribution_adam', ...
    'evaluate_burgers_deeponet_ddad_reduced', ...
    'burgers_deeponet_study', ...
    'plot_burgers_deeponet_result'};

    for k = 1:numel(required_functions)
        if isempty(which(required_functions{k}))
            error('Required project function not found: %s', ...
                required_functions{k});
        end
    end

    fprintf('Project root: %s\n',project_root);
    fprintf('Burgers DeepONet example: %s\n',example_dir);
end
