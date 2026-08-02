function [project_root,examples_dir] = setup_project_paths()
%SETUP_PROJECT_PATHS Robustly locate the AD-RaNN project root.
%
% This helper does NOT assume that the current example is exactly a fixed
% number of folders below the project root.  It walks upward until it finds
%
%   src/layer_growth
%
% and then adds the whole src tree to the MATLAB path.

    this_dir = fileparts(mfilename('fullpath'));

    probe = this_dir;
    project_root = '';

    for level = 1:10

        has_src = exist(fullfile(probe,'src'),'dir') == 7;
        has_growth = exist(fullfile(probe,'src','layer_growth'),'dir') == 7;

        if has_src && has_growth
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
            'Could not locate the project root containing src/layer_growth. ', ...
            'Please place this example somewhere inside the AD-RaNN project, ', ...
            'or copy src/layer_growth into the project src folder.']);
    end

    src_dir = fullfile(project_root,'src');

    addpath(genpath(src_dir));

    examples_dir = fullfile(project_root,'examples');

    poisson_dir = fullfile(examples_dir,'poisson_2d');

    if exist(poisson_dir,'dir') == 7
        addpath(poisson_dir);
    end

    addpath(this_dir);

    % Refresh MATLAB's path/file cache.
    rehash path;

    % Fail early with a useful message instead of failing deep in the run.
    required = { ...
        'select_growth_centers', ...
        'fit_growth_block_ddad', ...
        'evaluate_growth_features'};

    for k = 1:numel(required)
        if isempty(which(required{k}))
            error('Required function %s is still not on the MATLAB path.', ...
                required{k});
        end
    end

    fprintf('Project root: %s\n',project_root);
    fprintf('Layer-growth module: %s\n', ...
        fullfile(project_root,'src','layer_growth'));
end
