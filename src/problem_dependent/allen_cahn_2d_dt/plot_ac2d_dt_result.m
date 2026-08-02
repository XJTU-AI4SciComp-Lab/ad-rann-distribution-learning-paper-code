function plot_ac2d_dt_result(results)
%PLOT_AC2D_DT_RESULT Plot final solution, error, p(t), rho(t), and centers.

    ref = results.reference;
    pred = results.pred_snapshots(:,:,end);
    exact = ref.U;

    err = abs(pred-exact);

    method_label = sprintf( ...
        '%s-DT %s', ...
        results.method,results.growth_tag);

    % =====================================================================
    % Final-time fields
    % =====================================================================
    figure('Color','w');

    tiledlayout(1,3, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    nexttile;
    surf(ref.Xe,ref.Ye,exact,'EdgeColor','none');
    view(2);
    axis equal tight;
    colorbar;
    xlabel('$x$','Interpreter','latex','FontSize',14);
    ylabel('$y$','Interpreter','latex','FontSize',14);
    title('Reference at $t=1$', ...
        'Interpreter','latex','FontSize',12);
    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;

    nexttile;
    surf(ref.Xe,ref.Ye,pred,'EdgeColor','none');
    view(2);
    axis equal tight;
    colorbar;
    xlabel('$x$','Interpreter','latex','FontSize',14);
    ylabel('$y$','Interpreter','latex','FontSize',14);
    title([method_label ' at $t=1$'], ...
        'Interpreter','latex','FontSize',12);
    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;

    nexttile;
    surf(ref.Xe,ref.Ye,err,'EdgeColor','none');
    view(2);
    axis equal tight;
    colorbar;
    xlabel('$x$','Interpreter','latex','FontSize',14);
    ylabel('$y$','Interpreter','latex','FontSize',14);
    title('Absolute error', ...
        'Interpreter','latex','FontSize',12);
    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;

    % =====================================================================
    % First-layer p(t)
    % =====================================================================
    Nt = results.cfg.num_time_steps;

    t = linspace( ...
        results.cfg.t_domain(1), ...
        results.cfg.t_domain(2), ...
        Nt+1).';

    figure('Color','w');

    plot(t,results.p_history(:,1), ...
        'LineWidth',1.5);
    hold on;
    plot(t,results.p_history(:,2), ...
        '--','LineWidth',1.5);
    hold off;

    xlabel('$t$','Interpreter','latex','FontSize',14);
    ylabel('$p$','Interpreter','latex','FontSize',14);

    title([method_label ': first-layer parameters'], ...
        'Interpreter','latex','FontSize',12);

    legend({'$p_x$','$p_y$'}, ...
        'Interpreter','latex', ...
        'Location','best');

    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;

    % =====================================================================
    % Second-layer rho(t)
    % =====================================================================
    if any(isfinite(results.rho_history))

        figure('Color','w');

        stairs(t,results.rho_history, ...
            'LineWidth',1.5);

        xlabel('$t$','Interpreter','latex','FontSize',14);
        ylabel('$\rho$','Interpreter','latex','FontSize',14);

        title([method_label ': second-layer parameter'], ...
            'Interpreter','latex','FontSize',12);

        set(gca,'FontSize',12,'LineWidth',1);
        box on;
        grid off;
    end

    % =====================================================================
    % Final residual centers
    % =====================================================================
    if isstruct(results.final_growth) && ...
       isfield(results.final_growth,'centers')

        figure('Color','w');

        surf(ref.Xe,ref.Ye,pred,'EdgeColor','none');
        view(2);
        axis equal tight;
        colorbar;
        hold on;

        centers = results.final_growth.centers;

        plot( ...
            centers(:,1),centers(:,2), ...
            '.','MarkerSize',8);

        hold off;

        xlabel('$x$','Interpreter','latex','FontSize',14);
        ylabel('$y$','Interpreter','latex','FontSize',14);

        title([method_label ': final growth centers'], ...
            'Interpreter','latex','FontSize',12);

        set(gca,'FontSize',12,'LineWidth',1);
        box on;
        grid off;
    end
end
