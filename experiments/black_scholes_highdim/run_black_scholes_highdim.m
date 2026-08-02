function results = run_black_scholes_highdim(d)
%RUN_BLACK_SCHOLES_HIGHDIM Run one high-dimensional Black-Scholes case.

    if nargin < 1 || isempty(d)
        d = 50;
    end

    [cfg,~] = config_black_scholes_highdim(d);

    if exist(cfg.data_file,'file') ~= 2
        if cfg.data.auto_generate_if_missing
            generate_black_scholes_data(d);
        else
            error([ ...
                'Pre-sampled data file is missing:\n%s\n\n', ...
                'Run generate_black_scholes_data(%d) first.'], ...
                cfg.data_file,d);
        end
    end

    results = black_scholes_highdim_study(cfg);
end
