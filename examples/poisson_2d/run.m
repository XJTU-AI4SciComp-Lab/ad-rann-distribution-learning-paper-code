clear; clc;

this_file = mfilename('fullpath');
example_dir = fileparts(this_file);
root = fileparts(fileparts(example_dir));

addpath(genpath(fullfile(root,'src')));
addpath(example_dir);

cfg = config();

total_timer = tic;

activation = normalize_activation_name(cfg.activation);

%% ========================================================================
%  Distribution initialization
% =========================================================================
initialization_timer = tic;

switch lower(cfg.initialization.method)

    case 'frequency'

        if strcmp(activation,'gaussian')
            % Exact legacy Gaussian initialization path.
            [p0,frequency_info] = frequency_initialization(cfg);
        else
            [p0,frequency_info] = ...
                frequency_initialization_activation(cfg,activation);
        end

        fprintf('\n');
        fprintf('Frequency-based initialization\n');
        fprintf('initial p = [%.6f, %.6f]\n',p0(1),p0(2));

    case 'manual'

        p0 = cfg.initialization.manual_p(:);
        frequency_info = struct();

        fprintf('\n');
        fprintf('Manual initialization\n');
        fprintf('initial p = [%.6f, %.6f]\n',p0(1),p0(2));

    otherwise

        error('Unknown initialization method: %s', ...
            cfg.initialization.method);
end

initialization_time = toc(initialization_timer);

%% ========================================================================
%  Fixed basis and PDE data
% =========================================================================
setup_timer = tic;

% Keep the original 2-D random draw unchanged for this Poisson example.
basis = build_random_weights( ...
    cfg.num_features,cfg.domain,cfg.seed);

problem.domain = cfg.domain;
problem.boundary_penalty = cfg.boundary_penalty;

problem.Xi = tensor_grid( ...
    cfg.domain,cfg.interior_grid,1e-6);

problem.fi = rhs(problem.Xi);

nB = cfg.boundary_points_per_side;

x = linspace(cfg.domain(1,1),cfg.domain(1,2),nB)';
y = linspace(cfg.domain(2,1),cfg.domain(2,2),nB)';

problem.Xb = [ ...
    cfg.domain(1,1)*ones(nB,1), y; ...
    cfg.domain(1,2)*ones(nB,1), y; ...
    x, cfg.domain(2,1)*ones(nB,1); ...
    x, cfg.domain(2,2)*ones(nB,1)];

problem.gb = exact_solution(problem.Xb);

% y is independent of p, so construct it once.
problem.y = [ ...
    problem.fi; ...
    cfg.boundary_penalty*problem.gb];

ls_opts = cfg.linear_solver;
ls_opts.compute_spectrum = cfg.compute_spectrum;

if cfg.use_fast_evaluator

    cache = prepare_poisson_cache(problem,basis);

    switch activation

        case 'gaussian'
            % Exact legacy Gaussian fast evaluator. The inner optimization
            % path is unchanged.
            objective_fun = @(p) ...
                evaluate_poisson_reduced_fast( ...
                    p,cache,cfg.lambda,ls_opts);

        case 'tanh'
            objective_fun = @(p) ...
                evaluate_poisson_reduced_fast_tanh( ...
                    p,cache,cfg.lambda,ls_opts);

        case 'sin'
            objective_fun = @(p) ...
                evaluate_poisson_reduced_fast_sin( ...
                    p,cache,cfg.lambda,ls_opts);
    end

else

    if strcmp(activation,'gaussian')
        % Exact legacy Gaussian assembly.
        system_builder = @(p) ...
            build_system(p,problem,basis);
    else
        system_builder = @(p) ...
            build_system_activation( ...
                p,problem,basis,activation);
    end

    objective_fun = @(p) ...
        evaluate_pde_reduced_generic( ...
            p,system_builder,cfg.lambda,ls_opts);
end

setup_time = toc(setup_timer);

%% ========================================================================
%  Distribution training
% =========================================================================
training_timer = tic;

[p_opt,history] = ...
    optimize_distribution_adam( ...
        p0,objective_fun,cfg.optimizer);

training_time = toc(training_timer);

%% ========================================================================
%  Final unregularized least-squares refit
% =========================================================================
final_assembly_timer = tic;

if strcmp(activation,'gaussian')
    % Exact legacy Gaussian final assembly.
    [M,y_rhs,~] = ...
        build_system(p_opt,problem,basis);
else
    [M,y_rhs,~] = ...
        build_system_activation( ...
            p_opt,problem,basis,activation);
end

final_assembly_time = toc(final_assembly_timer);

[coef,final_ls_info] = ...
    solve_least_squares( ...
        M,y_rhs,cfg.linear_solver);

%% ========================================================================
%  Independent error evaluation
% =========================================================================
evaluation_timer = tic;

Xtest = tensor_grid( ...
    cfg.domain,cfg.test_grid,0);

switch activation

    case 'gaussian'
        % Exact legacy Gaussian feature evaluation.
        Phi_test = gaussian_features( ...
            Xtest,p_opt,basis);

    case 'tanh'
        Phi_test = tanh_features( ...
            Xtest,p_opt,basis);

    case 'sin'
        Phi_test = sin_features( ...
            Xtest,p_opt,basis);
end

pred = Phi_test*coef;
ref = exact_solution(Xtest);

err_l2 = relative_l2(pred,ref);
err_linf = relative_linf(pred,ref);

evaluation_time = toc(evaluation_timer);

total_time = toc(total_timer);

other_time = ...
    total_time ...
    -initialization_time ...
    -setup_time ...
    -training_time ...
    -final_assembly_time ...
    -final_ls_info.total_time ...
    -evaluation_time;

%% ========================================================================
%  Output
% =========================================================================
fprintf('\n');
fprintf('========================================\n');
fprintf('             Final result\n');
fprintf('========================================\n');

fprintf('seed                = %d\n',cfg.seed);
fprintf('num features        = %d\n',cfg.num_features);
fprintf('activation          = %s\n',activation);
fprintf('lambda              = %.3e\n',cfg.lambda);
fprintf('initialization      = %s\n',cfg.initialization.method);

fprintf('initial p           = [%.8f, %.8f]\n', ...
    p0(1),p0(2));

fprintf('selected p          = [%.8f, %.8f]\n', ...
    p_opt(1),p_opt(2));

fprintf('best training MSE   = %.6e\n', ...
    history.best_selection_value);

fprintf('relative L2 error   = %.6e\n',err_l2);
fprintf('relative Linf error = %.6e\n',err_linf);

fprintf('\n');
fprintf('---------------- Timing ----------------\n');
fprintf('initialization time = %.6f s\n',initialization_time);
fprintf('setup time          = %.6f s\n',setup_time);
fprintf('training time       = %.6f s\n',training_time);
fprintf('final assembly time = %.6f s\n',final_assembly_time);
fprintf('final LS solver     = %s\n',final_ls_info.method);
fprintf('final LS time       = %.6f s\n',final_ls_info.total_time);
fprintf('evaluation time     = %.6f s\n',evaluation_time);
fprintf('other time          = %.6f s\n',other_time);
fprintf('total time          = %.6f s\n',total_time);
fprintf('========================================\n');

%% ========================================================================
%  Workspace result
% =========================================================================
result.cfg = cfg;
result.activation = activation;
result.initial_p = p0;
result.p_opt = p_opt;
result.history = history;

result.relative_l2 = err_l2;
result.relative_linf = err_linf;

result.frequency_info = frequency_info;

result.initialization_time = initialization_time;
result.setup_time = setup_time;
result.training_time = training_time;
result.final_assembly_time = final_assembly_time;
result.final_ls_info = final_ls_info;
result.evaluation_time = evaluation_time;
result.other_time = other_time;
result.total_time = total_time;


function activation = normalize_activation_name(activation)

    activation = lower(strtrim(char(activation)));

    switch activation
        case {'gaussian','gauss'}
            activation = 'gaussian';
        case 'tanh'
            activation = 'tanh';
        case {'sin','sine'}
            activation = 'sin';
        otherwise
            error(['Unknown activation "%s". Supported activations are ', ...
                   'gaussian, tanh, and sin.'],activation);
    end
end
