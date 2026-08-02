function plot_diffusion_reaction_deeponet_result(results)
%PLOT_DIFFUSION_REACTION_DEEPONET_RESULT First test function comparison.

    M = results.test_metrics;

    figure('Color','w');

    subplot(1,3,1);
    imagesc(M.first_true);
    axis image;
    colorbar;
    title('Reference');

    subplot(1,3,2);
    imagesc(M.first_prediction);
    axis image;
    colorbar;
    title(strrep(results.mode,'_','\_'));

    subplot(1,3,3);
    imagesc(M.first_absolute_error);
    axis image;
    colorbar;
    title('Absolute error');

    sgtitle(sprintf('Overall relative L2 = %.4e', ...
        M.overall_relative_l2));
end
