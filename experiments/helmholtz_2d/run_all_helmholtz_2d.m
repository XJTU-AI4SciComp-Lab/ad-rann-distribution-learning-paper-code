clear;
clear functions;
clc;
close all;

case_list = {'6_6','1_20'};
method_list = {'FRE','PDAD'};

all_tables = cell(numel(case_list)*numel(method_list),1);
row_id = 0;
project_root = '';

for i = 1:numel(case_list)
    for j = 1:numel(method_list)

        [cfg,project_root] = ...
            config_helmholtz_2d(case_list{i},method_list{j});

        % Do not open many windows in the batch run.
        cfg.plot.enabled = false;

        % PDAD uses 1000 features for p training and 3000 for final solve.
        cfg.training_reduction.enabled = true;
        cfg.training_reduction.num_features = 1000;

        results = helmholtz_2d_study(cfg,project_root);

        row_id = row_id+1;
        all_tables{row_id} = results.summary;
    end
end

HelmholtzSummary = vertcat(all_tables{:});

disp(HelmholtzSummary);

output_dir = fullfile(project_root,'results','helmholtz_2d');

if ~isfolder(output_dir)
    mkdir(output_dir);
end

csv_file = fullfile(output_dir,'helmholtz_2d_all_cases_summary.csv');
mat_file = fullfile(output_dir,'helmholtz_2d_all_cases_summary.mat');

writetable(HelmholtzSummary,csv_file);
save(mat_file,'HelmholtzSummary');

fprintf('\nCombined summary saved to:\n%s\n',csv_file);
