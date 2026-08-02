function [project_root,example_dir] = setup_burgers2d_paths()
%SETUP_BURGERS2D_PATHS Locate project root and add common src paths.

    example_dir = fileparts(mfilename('fullpath'));

    probe = example_dir;
    project_root = '';

    for level = 1:12

        has_src = exist(fullfile(probe,'src'),'dir') == 7;
        has_data = exist(fullfile(probe,'data'),'dir') == 7;
        has_examples = exist(fullfile(probe,'examples'),'dir') == 7;
        has_growth = ...
            exist(fullfile(probe,'src','layer_growth'),'dir') == 7;

        if has_src && has_data && has_examples && has_growth
            project_root = probe;
            break;
        end

        parent = fileparts(probe);

        if strcmp(parent,probe)
            break;
        end

        probe = parent;
    end

    if isempty(project_root)
        error([ ...
            'Could not locate a project root containing src/, data/, ', ...
            'examples/, and src/layer_growth/.']);
    end

    addpath(genpath(fullfile(project_root,'src')));
    addpath(example_dir);
    rehash path;

    required = { ...
        'build_random_weights_nd', ...
        'build_preactivation', ...
        'activation_derivatives', ...
        'prepare_data_cache', ...
        'evaluate_data_reduced_fast', ...
        'optimize_distribution_adam', ...
        'solve_ridge', ...
        'solve_least_squares', ...
        'relative_l2', ...
        'select_growth_centers', ...
        'build_growth_directions', ...
        'fit_growth_block_ddad', ...
        'evaluate_growth_features'};

    for k = 1:numel(required)

        if isempty(which(required{k}))
            error('Required project function not found: %s',required{k});
        end
    end

    fprintf('Project root: %s\n',project_root);
    fprintf('Burgers2D example: %s\n',example_dir);
    fprintf('Layer growth: %s\n', ...
        fullfile(project_root,'src','layer_growth'));
end
