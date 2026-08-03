clear;
clc;
close all;

%% ========================================================================
%  Load PDAD result
% ========================================================================

mat_file = fullfile( ...
    'results_PDAD', ...
    'PDAD_sharp_layer_newton_results.mat');

S = load(mat_file);

if ~isfield(S,'results')
    error('The MAT file does not contain variable "results".');
end

results = S.results;

%% ========================================================================
%  Extract data
% ========================================================================

X = results.problem.Xtest;

u_exact  = results.problem.utest(:);
u_pdad   = results.base.pred_test(:);
u_growth = results.growth.pred_test(:);

% Pointwise absolute errors
err_pdad   = abs(u_pdad   - u_exact);
err_growth = abs(u_growth - u_exact);

%% ========================================================================
%  Recover structured grid
% ========================================================================

x_unique = unique(X(:,1));
y_unique = unique(X(:,2));

nx = numel(x_unique);
ny = numel(y_unique);

if nx*ny ~= size(X,1)
    error('Xtest is not a complete tensor-product grid.');
end

Xgrid = reshape(X(:,1),ny,nx);
Ygrid = reshape(X(:,2),ny,nx);

E_pdad   = reshape(err_pdad,ny,nx);
E_growth = reshape(err_growth,ny,nx);

%% ========================================================================
%  Output directory
% ========================================================================

out_dir = fullfile('results_PDAD','figures');

if exist(out_dir,'dir') ~= 7
    mkdir(out_dir);
end

%% ========================================================================
%  Figure 1: PDAD pointwise error
% ========================================================================

fig1 = figure( ...
    'Color','w', ...
    'Position',[100,100,600,500]);

contourf( ...
    Xgrid,Ygrid,E_pdad, ...
    40, ...
    'LineColor','none');

axis equal tight;
box on;
grid off;

xlabel('$x$', ...
    'Interpreter','latex', ...
    'FontSize',14);

ylabel('$y$', ...
    'Interpreter','latex', ...
    'FontSize',14);

title('PDAD', ...
    'Interpreter','latex', ...
    'FontSize',12);

set(gca, ...
    'FontSize',12, ...
    'LineWidth',1);


clim('auto');


colorbar;

exportgraphics( ...
    fig1, ...
    fullfile(out_dir,'PDAD_pointwise_error.pdf'), ...
    'ContentType','vector');

exportgraphics( ...
    fig1, ...
    fullfile(out_dir,'PDAD_pointwise_error.png'), ...
    'Resolution',300);

%% ========================================================================
%  Figure 2: PDAD + layer growth pointwise error
% ========================================================================

fig2 = figure( ...
    'Color','w', ...
    'Position',[750,100,600,500]);

contourf( ...
    Xgrid,Ygrid,E_growth, ...
    40, ...
    'LineColor','none');

axis equal tight;
box on;
grid off;

xlabel('$x$', ...
    'Interpreter','latex', ...
    'FontSize',14);

ylabel('$y$', ...
    'Interpreter','latex', ...
    'FontSize',14);

title('PDAD + layer growth', ...
    'Interpreter','latex', ...
    'FontSize',12);

set(gca, ...
    'FontSize',12, ...
    'LineWidth',1);


clim('auto');


colorbar;

exportgraphics( ...
    fig2, ...
    fullfile(out_dir,'PDAD_layer_growth_pointwise_error.pdf'), ...
    'ContentType','vector');

exportgraphics( ...
    fig2, ...
    fullfile(out_dir,'PDAD_layer_growth_pointwise_error.png'), ...
    'Resolution',300);

%% ========================================================================
%  Print information
% ========================================================================

fprintf('\n');
fprintf('PDAD max pointwise error        = %.6e\n',max(err_pdad));
fprintf('PDAD+LG max pointwise error     = %.6e\n',max(err_growth));

fprintf('\nFigures saved to:\n');
fprintf('  %s\n',fullfile(out_dir,'PDAD_pointwise_error.pdf'));
fprintf('  %s\n',fullfile(out_dir,'PDAD_layer_growth_pointwise_error.pdf'));
