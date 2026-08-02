function files = plot_burgers_deeponet_result(results)
%PLOT_BURGERS_DEEPONET_RESULT Three separate paper-ready figure windows.

    cfg = results.cfg;
    test = results.test;

    if ischar(cfg.plot.sample) || isstring(cfg.plot.sample)
        if strcmpi(cfg.plot.sample,'worst')
            sample_index = test.worst_index;
        else
            error('cfg.plot.sample must be ''worst'' or a test index.');
        end
    else
        sample_index = cfg.plot.sample;
    end

    if ~isscalar(sample_index) || sample_index < 1 || ...
            sample_index > size(test.exact,1)
        error('Requested plot sample is outside the test set.');
    end

    shape = test.grid_shape;
    exact = reshape(test.exact(sample_index,:),shape);
    prediction = reshape(test.prediction(sample_index,:),shape);
    absolute_error = reshape(test.absolute_error(sample_index,:),shape);

    solution_limits = [ ...
        min([exact(:);prediction(:)]), ...
        max([exact(:);prediction(:)])];

    if solution_limits(1) == solution_limits(2)
        solution_limits = solution_limits+[-1,1];
    end

    error_limit = max(absolute_error(:));

    if error_limit == 0
        error_limit = eps;
    end

    figure_dir = fullfile(results.output_dir,'figures');

    if exist(figure_dir,'dir') ~= 7
        mkdir(figure_dir);
    end

    [fig_exact,files.exact] = make_field_figure( ...
        test.t,test.x,exact,'Exact solution', ...
        solution_limits,'exact_solution',figure_dir,cfg.plot);

    [fig_prediction,files.prediction] = make_field_figure( ...
        test.t,test.x,prediction,'Numerical solution', ...
        solution_limits,'numerical_solution',figure_dir,cfg.plot);

    [fig_error,files.absolute_error] = make_field_figure( ...
        test.t,test.x,absolute_error,'Absolute error', ...
        [0,error_limit],'absolute_error',figure_dir,cfg.plot);

    files.figure_handles = [fig_exact,fig_prediction,fig_error];
    files.sample_index = sample_index;
    files.relative_l2 = test.relative_l2(sample_index);

    fprintf('Plotted test sample %d (relative L2 %.6e).\n', ...
        sample_index,files.relative_l2);
    fprintf('Figure files saved to: %s\n',figure_dir);
end


function [fig,paths] = make_field_figure( ...
    t,x,Z,title_text,color_limits,file_stem,figure_dir,plot_cfg)

    fig = figure( ...
        'Name',title_text, ...
        'Color','w', ...
        'Units','pixels', ...
        'Position',plot_cfg.window_position);

    layout = tiledlayout(fig,1,1, ...
        'Padding','compact','TileSpacing','compact');
    ax = nexttile(layout);

    contourf(ax,t,x,Z,plot_cfg.num_levels,'LineStyle','none');
    axis(ax,'xy');
    axis(ax,'square');
    xlim(ax,[0,1]);
    ylim(ax,[0,1]);
    clim(ax,color_limits);
    colormap(ax,jet(256));
    colorbar(ax);

    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$x$','Interpreter','latex');
    title(ax,title_text,'Interpreter','latex','FontWeight','normal');

    set(ax, ...
        'FontName','Times New Roman', ...
        'FontSize',plot_cfg.font_size, ...
        'LineWidth',0.8, ...
        'Box','on', ...
        'Layer','top');

    paths = struct();
    paths.png = fullfile(figure_dir,[file_stem,'.png']);
    paths.pdf = fullfile(figure_dir,[file_stem,'.pdf']);

    exportgraphics(fig,paths.png, ...
        'Resolution',plot_cfg.png_resolution, ...
        'BackgroundColor','white');
    exportgraphics(fig,paths.pdf, ...
        'ContentType','vector', ...
        'BackgroundColor','white');
end
