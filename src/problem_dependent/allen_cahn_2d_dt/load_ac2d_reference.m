function ref = load_ac2d_reference(project_root)
%LOAD_AC2D_REFERENCE Load the 50x50x10 sine-spectral reference.

    candidates = { ...
        fullfile(project_root,'data', ...
            'AC_2D_Dirichlet_spectral_10snap.mat'), ...
        fullfile(project_root,'data','allen_cahn_2d_dt', ...
            'AC_2D_Dirichlet_spectral_10snap.mat')};

    data_file = '';

    for k = 1:numel(candidates)

        if exist(candidates{k},'file') == 2
            data_file = candidates{k};
            break;
        end
    end

    if isempty(data_file)

        error([ ...
            'Could not find AC_2D_Dirichlet_spectral_10snap.mat.\n', ...
            'Expected:\n  %s'], ...
            candidates{1});
    end

    S = load(data_file);

    required = {'Xe','Ye','U','UU','tt'};

    for k = 1:numel(required)

        if ~isfield(S,required{k})
            error('Reference file is missing variable %s.',required{k});
        end
    end

    Xe = double(S.Xe);
    Ye = double(S.Ye);
    U = double(S.U);
    UU = double(S.UU);
    tt = double(S.tt(:));

    if ~isequal(size(Xe),[50,50]) || ...
       ~isequal(size(Ye),[50,50]) || ...
       ~isequal(size(U),[50,50]) || ...
       ~isequal(size(UU),[50,50,10]) || ...
       ~isequal(size(tt),[10,1])

        error([ ...
            'Expected Xe,Ye,U = 50x50, UU = 50x50x10, ', ...
            'and tt = 10x1.']);
    end

    expected_tt = (1:10).'/10;

    if max(abs(tt-expected_tt)) > 1e-12
        error('Reference times must be 0.1,0.2,...,1.0.');
    end

    U_last = UU(:,:,end);

    if max(abs(U(:)-U_last(:))) > 1e-12
        error('Reference U must equal UU(:,:,end).');
    end

    if any(~isfinite(Xe(:))) || ...
       any(~isfinite(Ye(:))) || ...
       any(~isfinite(UU(:)))
        error('Reference data contain NaN or Inf.');
    end

    ref = struct();

    ref.Xe = Xe;
    ref.Ye = Ye;
    ref.U = U;
    ref.UU = UU;
    ref.tt = tt;

    ref.points = [Xe(:),Ye(:)];
    ref.num_snapshots = numel(tt);
    ref.file = data_file;

    if isfield(S,'verification')
        ref.verification = S.verification;
    else
        ref.verification = [];
    end

    if isfield(S,'meta')
        ref.meta = S.meta;
    else
        ref.meta = [];
    end

    fprintf('Reference file: %s\n',data_file);
    fprintf('Reference size: 50 x 50 x 10\n');

    if isstruct(ref.verification) && ...
       isfield(ref.verification,'passed')

        fprintf('Reference self-convergence passed: %d\n', ...
            logical(ref.verification.passed));
    end
end
