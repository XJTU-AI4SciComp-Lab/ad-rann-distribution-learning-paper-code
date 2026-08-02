function plot_ac_1d_dt_result(results)
%PLOT_AC_1D_DT_RESULT Plot final solution and p(t).

    xx = results.reference.xx;
    exact = results.reference.uu;
    pred = results.pred_snapshots;

    method = results.method;

    % =====================================================================
    % Final-time solution
    % =====================================================================
    figure('Color','w');

    plot(xx,exact(:,end),'k-','LineWidth',1.6);
    hold on;
    plot(xx,pred(:,end),'--','LineWidth',1.6);
    hold off;

    xlabel('$x$','Interpreter','latex','FontSize',14);
    ylabel('$u(x,T)$','Interpreter','latex','FontSize',14);

    title([method '-DT Allen--Cahn: final-time solution'], ...
        'Interpreter','latex','FontSize',12);

    legend({'Reference','Prediction'}, ...
        'Location','best','Interpreter','none');

    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;

    % =====================================================================
    % p(t)
    % =====================================================================
    Nt = results.cfg.num_time_steps;
    t = linspace( ...
        results.cfg.t_domain(1), ...
        results.cfg.t_domain(2), ...
        Nt+1).';

    figure('Color','w');

    stairs(t,results.p_history,'LineWidth',1.4);

    xlabel('$t$','Interpreter','latex','FontSize',14);
    ylabel('$r_x$','Interpreter','latex','FontSize',14);

    title([method '-DT Allen--Cahn: evolution of $r_x$'], ...
        'Interpreter','latex','FontSize',12);

    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;
end
