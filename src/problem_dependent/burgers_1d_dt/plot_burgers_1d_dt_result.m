function plot_burgers_1d_dt_result(results)
%PLOT_BURGERS_1D_DT_RESULT Plot final solution and p(t) history.

    xx = results.reference.xx;
    exact = results.reference.exact;
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

    title([method '-DT: final-time solution'], ...
        'Interpreter','none','FontSize',12);

    legend({'Reference','Prediction'}, ...
        'Location','best','Interpreter','none');

    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;

    % =====================================================================
    % Distribution-parameter history
    % =====================================================================
    Nt = results.cfg.num_time_steps;
    t0 = results.cfg.t_domain(1);
    tf = results.cfg.t_domain(2);

    t = linspace(t0,tf,Nt+1).';

    figure('Color','w');

    plot(t,results.p_history,'LineWidth',1.5);

    xlabel('$t$','Interpreter','latex','FontSize',14);
    ylabel('$p$','Interpreter','latex','FontSize',14);

    title([method '-DT: distribution parameter'], ...
        'Interpreter','none','FontSize',12);

    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;
end
