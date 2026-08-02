function dataset = load_black_scholes_dataset(cfg)
%LOAD_BLACK_SCHOLES_DATASET Load and validate fixed Monte-Carlo data.

    if exist(cfg.data_file,'file') ~= 2
        error('Data file not found: %s',cfg.data_file);
    end

    S = load(cfg.data_file,'dataset');

    if ~isfield(S,'dataset')
        error('Variable dataset not found in %s.',cfg.data_file);
    end

    dataset = S.dataset;

    required_meta = { ...
        'dimension','x_lower','x_upper','t_domain','mu','sigma', ...
        'payoff_strike','num_mc_samples'};

    for k = 1:numel(required_meta)
        if ~isfield(dataset.metadata,required_meta{k})
            error('Dataset metadata is missing %s.',required_meta{k});
        end
    end

    if dataset.metadata.dimension ~= cfg.dimension
        error('Dataset dimension does not match config.');
    end

    if abs(dataset.metadata.payoff_strike-cfg.payoff_strike) > 1e-12
        error([ ...
            'Dataset strike K=%.16g differs from config K=%.16g. ', ...
            'Regenerate the data file.'], ...
            dataset.metadata.payoff_strike,cfg.payoff_strike);
    end

    if dataset.metadata.num_mc_samples ~= cfg.data.num_mc_samples
        error('Dataset Monte-Carlo sample count differs from config.');
    end

    if max(abs(dataset.metadata.sigma(:)-cfg.sigma(:))) > 1e-14 || ...
            abs(dataset.metadata.mu-cfg.mu) > 1e-14
        error('Dataset Black-Scholes coefficients differ from config.');
    end

    validate_split(dataset.train,cfg,'train');
    validate_split(dataset.solve,cfg,'solve');

    if ~isfield(dataset,'test') || ...
            ~isfield(dataset.test,'xt') || ...
            ~isfield(dataset.test,'reference_values')
        error('Dataset test split is incomplete.');
    end
end


function validate_split(split,cfg,name)

    fields = { ...
        'interior_xt','boundary_xt','boundary_values', ...
        'initial_xt','initial_values'};

    for k = 1:numel(fields)
        if ~isfield(split,fields{k})
            error('Dataset %s split is missing %s.',name,fields{k});
        end
    end

    D = cfg.dimension+1;

    if size(split.interior_xt,2) ~= D || ...
            size(split.boundary_xt,2) ~= D || ...
            size(split.initial_xt,2) ~= D
        error('Dataset %s points have the wrong number of columns.',name);
    end

    if size(split.boundary_xt,1) ~= numel(split.boundary_values) || ...
            size(split.initial_xt,1) ~= numel(split.initial_values)
        error('Dataset %s values do not match point counts.',name);
    end
end
