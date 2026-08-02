clear;
clc;
close all;

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);


% The configuration function locates the project root and adds src/.
[cfg,project_root] = config_burgers_2d_dt('DDAD');

Nt_values = [50;100;200;400];
methods = {'PDAD','DDAD'};
growth_values = [false,true];

records = struct( ...
    'Method',{}, ...
    'Growth',{}, ...
    'Nt',{}, ...
    'TimeSec',{}, ...
    'SnapshotRelL2',{}, ...
    'FinalTimeRelL2',{}, ...
    'OrderSnapshot',{}, ...
    'OrderFinal',{}, ...
    'NumUpdateAttempts',{}, ...
    'NumEffectiveUpdates',{}, ...
    'NumGrowthRefreshes',{});

for im = 1:numel(methods)

    for ig = 1:numel(growth_values)

        previous_snapshot_error = NaN;
        previous_final_error = NaN;
        previous_Nt = NaN;

        for in = 1:numel(Nt_values)

            cfg = config_burgers_2d_dt(methods{im});
            cfg.num_time_steps = Nt_values(in);
            cfg.growth.enabled = growth_values(ig);
            cfg.verbose = false;

            result = burgers_2d_dt_study(cfg,project_root);

            if isfinite(previous_snapshot_error)

                order_snapshot = ...
                    log(previous_snapshot_error/ ...
                        result.snapshot_relative_l2) / ...
                    log(result.cfg.num_time_steps/previous_Nt);

                order_final = ...
                    log(previous_final_error/ ...
                        result.final_time_relative_l2) / ...
                    log(result.cfg.num_time_steps/previous_Nt);

            else

                order_snapshot = NaN;
                order_final = NaN;
            end

            idx = numel(records)+1;

            records(idx).Method = methods{im};

            if growth_values(ig)
                records(idx).Growth = 'LG';
            else
                records(idx).Growth = 'NoLG';
            end

            records(idx).Nt = result.cfg.num_time_steps;
            records(idx).TimeSec = result.total_time;
            records(idx).SnapshotRelL2 = ...
                result.snapshot_relative_l2;
            records(idx).FinalTimeRelL2 = ...
                result.final_time_relative_l2;
            records(idx).OrderSnapshot = order_snapshot;
            records(idx).OrderFinal = order_final;
            records(idx).NumUpdateAttempts = ...
                result.num_update_attempts;
            records(idx).NumEffectiveUpdates = ...
                result.num_effective_updates;
            records(idx).NumGrowthRefreshes = ...
                result.num_growth_refreshes;

            previous_snapshot_error = ...
                result.snapshot_relative_l2;
            previous_final_error = ...
                result.final_time_relative_l2;
            previous_Nt = result.cfg.num_time_steps;
        end
    end
end

Method = {records.Method}.';
Growth = {records.Growth}.';
Nt = reshape([records.Nt],[],1);
TimeSec = reshape([records.TimeSec],[],1);
SnapshotRelL2 = reshape([records.SnapshotRelL2],[],1);
FinalTimeRelL2 = reshape([records.FinalTimeRelL2],[],1);
OrderSnapshot = reshape([records.OrderSnapshot],[],1);
OrderFinal = reshape([records.OrderFinal],[],1);
NumUpdateAttempts = reshape([records.NumUpdateAttempts],[],1);
NumEffectiveUpdates = reshape([records.NumEffectiveUpdates],[],1);
NumGrowthRefreshes = reshape([records.NumGrowthRefreshes],[],1);

convergence = table( ...
    Method,Growth,Nt,TimeSec, ...
    SnapshotRelL2,FinalTimeRelL2, ...
    OrderSnapshot,OrderFinal, ...
    NumUpdateAttempts,NumEffectiveUpdates,NumGrowthRefreshes);

disp(convergence);

out_dir = fullfile( ...
    project_root,'results','burgers_2d_dt','convergence');

if exist(out_dir,'dir') ~= 7
    mkdir(out_dir);
end

writetable( ...
    convergence, ...
    fullfile(out_dir,'BURGERS2D_convergence.csv'));

save( ...
    fullfile(out_dir,'BURGERS2D_convergence.mat'), ...
    'convergence','records','-v7.3');
