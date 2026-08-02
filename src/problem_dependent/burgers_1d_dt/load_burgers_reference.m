function ref = load_burgers_reference(project_root)
%LOAD_BURGERS_REFERENCE Load burgers.mat from the project data directory.
%
% Preferred location:
%   <project-root>/data/burgers.mat
%
% Fallback:
%   <project-root>/data/burgers_1d_dt/burgers.mat
%
% Required variables:
%   xx      : spatial reference coordinates
%   exact   : reference snapshots, one snapshot per column

    candidates = { ...
        fullfile(project_root,'data','burgers.mat'), ...
        fullfile(project_root,'data','burgers_1d_dt','burgers.mat')};

    data_file = '';

    for k = 1:numel(candidates)
        if exist(candidates{k},'file') == 2
            data_file = candidates{k};
            break;
        end
    end

    if isempty(data_file)
        error([ ...
            'Could not find burgers.mat. Expected either:\n  %s\nor:\n  %s'], ...
            candidates{1},candidates{2});
    end

    S = load(data_file);

    if ~isfield(S,'xx') || ~isfield(S,'exact')
        error('burgers.mat must contain variables xx and exact.');
    end

    xx = S.xx(:);
    exact = S.exact;

    if size(exact,1) ~= numel(xx)
        error('size(exact,1) must equal numel(xx).');
    end

    if any(~isfinite(xx)) || any(~isfinite(exact(:)))
        error('burgers.mat contains NaN or Inf.');
    end

    ref = struct();
    ref.xx = xx;
    ref.exact = exact;
    ref.num_snapshots = size(exact,2);
    ref.file = data_file;

    fprintf('Reference data: %s\n',data_file);
    fprintf('Reference grid points: %d\n',numel(xx));
    fprintf('Reference snapshots: %d\n',ref.num_snapshots);
end
