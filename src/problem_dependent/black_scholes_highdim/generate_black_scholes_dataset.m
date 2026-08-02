function dataset = generate_black_scholes_dataset(cfg)
%GENERATE_BLACK_SCHOLES_DATASET Generate fixed train/solve/test data.

    d = cfg.dimension;

    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('GENERATE BLACK-SCHOLES DATA\n');
    fprintf('====================================================================\n');
    fprintf('d                         = %d\n',d);
    fprintf('strike K                  = %.8g\n',cfg.payoff_strike);
    fprintf('Monte-Carlo samples       = %d\n',cfg.data.num_mc_samples);
    fprintf('train int/bnd/init        = %d / %d / %d\n', ...
        cfg.data.train.num_interior, ...
        cfg.data.train.num_boundary, ...
        cfg.data.train.num_initial);
    fprintf('solve int/bnd/init        = %d / %d / %d\n', ...
        cfg.data.solve.num_interior, ...
        cfg.data.solve.num_boundary, ...
        cfg.data.solve.num_initial);
    fprintf('test points               = %d\n',cfg.data.num_test);
    fprintf('====================================================================\n\n');

    stream = RandStream('mt19937ar','Seed',cfg.data.sampling_seed);

    dataset = struct();

    dataset.train = make_split( ...
        cfg,cfg.data.train,stream,cfg.data.mc_seed_train,'train');

    dataset.solve = make_split( ...
        cfg,cfg.data.solve,stream,cfg.data.mc_seed_solve,'solve');

    fprintf('\nGenerating independent test reference values...\n');
    dataset.test.xt = single(sample_black_scholes_points( ...
        cfg,'test',cfg.data.num_test,stream));

    dataset.test.reference_values = ...
        monte_carlo_black_scholes_reference( ...
            double(dataset.test.xt),cfg,cfg.data.mc_seed_test,'test');

    dataset.metadata = struct();
    dataset.metadata.dimension = d;
    dataset.metadata.x_lower = cfg.x_lower;
    dataset.metadata.x_upper = cfg.x_upper;
    dataset.metadata.t_domain = cfg.t_domain;
    dataset.metadata.mu = cfg.mu;
    dataset.metadata.sigma = cfg.sigma;
    dataset.metadata.payoff_strike = cfg.payoff_strike;
    dataset.metadata.num_mc_samples = cfg.data.num_mc_samples;
    dataset.metadata.sampling_seed = cfg.data.sampling_seed;
    dataset.metadata.mc_seed_train = cfg.data.mc_seed_train;
    dataset.metadata.mc_seed_solve = cfg.data.mc_seed_solve;
    dataset.metadata.mc_seed_test = cfg.data.mc_seed_test;
    dataset.metadata.generated_at = char(datetime('now'));
end


function split = make_split(cfg,count_cfg,stream,mc_seed,label)

    fprintf('Generating %s points...\n',label);

    split.interior_xt = single(sample_black_scholes_points( ...
        cfg,'interior',count_cfg.num_interior,stream));

    split.boundary_xt = single(sample_black_scholes_points( ...
        cfg,'boundary',count_cfg.num_boundary,stream));

    split.initial_xt = single(sample_black_scholes_points( ...
        cfg,'initial',count_cfg.num_initial,stream));

    split.initial_values = black_scholes_payoff( ...
        double(split.initial_xt(:,1:cfg.dimension)), ...
        cfg.payoff_strike);

    fprintf('Generating %s boundary Monte-Carlo values...\n',label);

    split.boundary_values = monte_carlo_black_scholes_reference( ...
        double(split.boundary_xt),cfg,mc_seed,[label ' boundary']);
end
