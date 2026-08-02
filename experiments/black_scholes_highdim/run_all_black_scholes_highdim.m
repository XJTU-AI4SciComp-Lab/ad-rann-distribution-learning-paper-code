function Summary = run_all_black_scholes_highdim()
%RUN_ALL_BLACK_SCHOLES_HIGHDIM Run d=20,50,100 and write one summary CSV.

    dimensions = [20,50,100];
    n = numel(dimensions);

    Dim = zeros(n,1);
    NumFeatures = zeros(n,1);
    Rs = nan(n,1);
    Rt = nan(n,1);
    TotalTimeSec = nan(n,1);
    RelativeL2 = nan(n,1);

    for j = 1:n
        d = dimensions(j);
        R = run_black_scholes_highdim(d);

        Dim(j) = d;
        NumFeatures(j) = R.cfg.num_features;
        Rs(j) = R.p_opt(1);
        Rt(j) = R.p_opt(2);
        TotalTimeSec(j) = R.timing.total;
        RelativeL2(j) = R.relative_l2;
    end

    Summary = table( ...
        Dim,NumFeatures,Rs,Rt,TotalTimeSec,RelativeL2);

    [~,project_root] = config_black_scholes_highdim(dimensions(1));
    out_dir = fullfile(project_root,'results','black_scholes_highdim');

    if exist(out_dir,'dir') ~= 7
        mkdir(out_dir);
    end

    csv_file = fullfile(out_dir,'black_scholes_summary.csv');
    writetable(Summary,csv_file);

    disp(Summary);
    fprintf('Summary saved to:\n%s\n',csv_file);
end
