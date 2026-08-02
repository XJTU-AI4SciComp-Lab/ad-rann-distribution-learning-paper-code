function [cfg,project_root] = config_poisson_lshape_mixture()
%CONFIG_POISSON_LSHAPE_MIXTURE Seed-42 distribution-flexibility study.

    [cfg,project_root] = config_poisson_lshape_2d('ADAM');

    cfg.method = 'MIXTURE_STUDY';
    cfg.study.seeds = 42;
    cfg.study.representative_seed_index = 1;

    % Equal final feature budget in both methods.
    cfg.global_num_features = 1100;
    cfg.mixture.num_near = 550;
    cfg.mixture.num_far = 550;
    cfg.mixture.radius_split = 0.1;

    % [r_x^near;r_y^near;r_x^far;r_y^far].
    cfg.mixture.p0 = [1;1;1;1];

    % Both methods use the same Adam settings, bounds, collocation points,
    % ridge value during distribution learning, and unregularized refit.
    % Inherited unchanged from the original non-decomposed L-shape run:
    % learning_rate=1, maxit=25, beta1=0.9, beta2=0.999.
    cfg.optimizer.verbose = true;

    cfg.plot.num_levels = 40;
    cfg.plot.font_size = 12;
    cfg.plot.window_position_solution = [80,100,520,440];
    cfg.plot.window_position_error = [640,100,520,440];
    cfg.plot.png_resolution = 600;

    cfg.output_root = fullfile( ...
        project_root,'results','poisson_lshape_mixture_2d');
end
