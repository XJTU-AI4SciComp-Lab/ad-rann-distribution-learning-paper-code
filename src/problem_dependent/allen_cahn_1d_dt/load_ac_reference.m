function ref = load_ac_reference(project_root)
%LOAD_AC_REFERENCE Load Allen--Cahn reference snapshots.
%
% Search order:
%
%   1. data/AC_new
%   2. data/AC_new.mat
%
% Required:
%
%   xx : Nx-by-1
%   uu : Nx-by-Ns
%
% Optional:
%
%   tt : Ns-by-1
%
% If tt is absent:
%
%   Ns=25 -> infer tt=(1:25)'/25
%   Ns=5  -> infer tt=linspace(0,1,5)'

    candidates = { ...
        fullfile(project_root,'data', ...
            'AC_new'), ...
        fullfile(project_root,'data','AC_new.mat')};

    data_file = '';

    for k = 1:numel(candidates)

        if exist(candidates{k},'file') == 2
            data_file = candidates{k};
            break;
        end
    end

    if isempty(data_file)

        error([ ...
            'Could not find an Allen-Cahn reference file.\n', ...
            'Expected either:\n  %s\nor:\n  %s'], ...
            candidates{1},candidates{2});
    end

    S = load(data_file);

    if ~isfield(S,'xx') || ~isfield(S,'uu')
        error('Reference file must contain variables xx and uu.');
    end

    xx = S.xx(:);
    uu = S.uu;

    if size(uu,1) ~= numel(xx)
        error('size(uu,1) must equal numel(xx).');
    end

    nsnap = size(uu,2);

    if isfield(S,'tt')

        tt = S.tt(:);

        if numel(tt) ~= nsnap
            error('numel(tt) must equal size(uu,2).');
        end

    elseif nsnap == 25

        tt = (1:25).'/25;

        warning([ ...
            'Reference file has 25 snapshots but no tt. ', ...
            'Using tt=(1:25)/25.']);

    elseif nsnap == 5

        tt = linspace(0,1,5).';

        warning([ ...
            'Reference file has 5 snapshots but no tt. ', ...
            'Using tt=[0,0.25,0.5,0.75,1].']);

    else

        error([ ...
            'Reference file has %d snapshots and no tt. ', ...
            'The temporal locations cannot be inferred safely.'], ...
            nsnap);
    end

    if any(~isfinite(xx)) || ...
       any(~isfinite(uu(:))) || ...
       any(~isfinite(tt))

        error('Reference data contain NaN or Inf.');
    end

    if any(diff(tt) <= 0)
        error('Reference times tt must be strictly increasing.');
    end

    ref = struct();

    ref.xx = xx;
    ref.uu = uu;
    ref.tt = tt;

    ref.num_snapshots = nsnap;
    ref.file = data_file;

    fprintf('Reference data: %s\n',data_file);
    fprintf('Reference grid points: %d\n',numel(xx));
    fprintf('Reference snapshots: %d\n',nsnap);
    fprintf('Reference time range: %.8f to %.8f\n', ...
        tt(1),tt(end));
end
