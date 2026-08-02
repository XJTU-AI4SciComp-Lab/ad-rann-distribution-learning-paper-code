function plot_sharp_layer_result(results,cfg,method_name)
%PLOT_SHARP_LAYER_RESULT

    nx = cfg.test_grid(1);
    ny = cfg.test_grid(2);

    x = linspace(cfg.domain(1,1),cfg.domain(1,2),nx);
    y = linspace(cfg.domain(2,1),cfg.domain(2,2),ny);

    Ue = reshape(results.problem.utest,ny,nx);
    Ub = reshape(results.base.pred_test,ny,nx);
    Ug = reshape(results.growth.pred_test,ny,nx);

    figure('Color','w');

    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

    nexttile;
    contourf(x,y,Ue,40,'LineColor','none');
    shading interp;
    colorbar;
    xlabel('$x$','Interpreter','latex','FontSize',14);
    ylabel('$y$','Interpreter','latex','FontSize',14);
    title('Reference solution','Interpreter','latex','FontSize',12);
    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;

    nexttile;
    contourf(x,y,abs(Ue-Ub),40,'LineColor','none');
    shading interp;
    colorbar;
    xlabel('$x$','Interpreter','latex','FontSize',14);
    ylabel('$y$','Interpreter','latex','FontSize',14);
    title([method_name ' absolute error'], ...
        'Interpreter','none','FontSize',12);
    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;

    nexttile;
    contourf(x,y,abs(Ue-Ug),40,'LineColor','none');
    shading interp;
    colorbar;
    xlabel('$x$','Interpreter','latex','FontSize',14);
    ylabel('$y$','Interpreter','latex','FontSize',14);
    title([method_name ' + DDAD growth error'], ...
        'Interpreter','none','FontSize',12);
    set(gca,'FontSize',12,'LineWidth',1);
    box on;
    grid off;
end
