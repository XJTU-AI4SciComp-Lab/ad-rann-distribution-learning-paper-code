function data_file = generate_black_scholes_data(d)
%GENERATE_BLACK_SCHOLES_DATA Pre-sample one dimension and save a MAT file.
%
% Monte-Carlo generation is deliberately separated from solver timing.

    if nargin < 1 || isempty(d)
        d = 100;
    end

    [cfg,~] = config_black_scholes_highdim(d);

    if exist(cfg.data_dir,'dir') ~= 7
        mkdir(cfg.data_dir);
    end

    if exist(cfg.data_file,'file') == 2 && ~cfg.data.overwrite
        fprintf('Data file already exists and overwrite=false:\n%s\n', ...
            cfg.data_file);
        data_file = cfg.data_file;
        return
    end

    dataset = generate_black_scholes_dataset(cfg);

    save(cfg.data_file,'dataset','-v7.3');

    data_file = cfg.data_file;
    fprintf('\nSaved Black-Scholes data:\n%s\n',data_file);
end
