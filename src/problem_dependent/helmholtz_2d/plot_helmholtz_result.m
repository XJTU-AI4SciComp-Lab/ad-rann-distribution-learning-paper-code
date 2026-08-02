function plot_helmholtz_result(results)
%PLOT_HELMHOLTZ_RESULT Plot the numerical solution and absolute error.

    cfg = results.cfg;

    nx = cfg.test_grid(1);
    ny = cfg.test_grid(2);

    X = results.test.X;

    x = unique(X(:,1));
    y = unique(X(:,2));

    if numel(x) ~= nx || numel(y) ~= ny
        error('Stored test grid does not match cfg.test_grid.');
    end

    U_pred = reshape(results.test.u_pred,ny,nx);
    AbsError = reshape(results.test.abs_error,ny,nx);

    method_title = strrep(results.method,'_',' ');

    %% Numerical solution

    fig1 = figure( ...
        'Color','w', ...
        'Name',[method_title,' Helmholtz solution'], ...
        'NumberTitle','off');

    surf(x,y,zeros(ny,nx),U_pred, ...
        'EdgeColor','none','FaceColor','interp');

    view(2);
    axis equal tight;
    box on;
    colorbar;
    colormap(gca,jet(256));

    xlabel('$x$','Interpreter','latex','FontSize',14);
    ylabel('$y$','Interpreter','latex','FontSize',14);
    title(sprintf('%s solution: $(a_1,a_2)=(%g,%g)$', ...
        method_title,cfg.a1,cfg.a2), ...
        'Interpreter','latex','FontSize',12);

    set(gca,'FontSize',12,'LineWidth',1.0,'Layer','top');

    %% Absolute error

    fig2 = figure( ...
        'Color','w', ...
        'Name',[method_title,' Helmholtz absolute error'], ...
        'NumberTitle','off');

    surf(x,y,zeros(ny,nx),AbsError, ...
        'EdgeColor','none','FaceColor','interp');

    view(2);
    axis equal tight;
    box on;
    colorbar;
    colormap(gca,turbo(256));

    xlabel('$x$','Interpreter','latex','FontSize',14);
    ylabel('$y$','Interpreter','latex','FontSize',14);
    title(sprintf('%s absolute error: $(a_1,a_2)=(%g,%g)$', ...
        method_title,cfg.a1,cfg.a2), ...
        'Interpreter','latex','FontSize',12);

    set(gca,'FontSize',12,'LineWidth',1.0,'Layer','top');

    %% Save

    if cfg.plot.save

        figure_dir = fullfile(cfg.output_root,'figures');

        if ~isfolder(figure_dir)
            mkdir(figure_dir);
        end

        tag = sprintf('helmholtz_%s_case_%s', ...
            lower(results.method),cfg.case_name);

        exportgraphics(fig1, ...
            fullfile(figure_dir,[tag,'_solution.png']), ...
            'Resolution',300);

        exportgraphics(fig2, ...
            fullfile(figure_dir,[tag,'_absolute_error.png']), ...
            'Resolution',300);
    end
end
