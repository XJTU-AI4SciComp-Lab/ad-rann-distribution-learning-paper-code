function test_data_driven_nd()
%TEST_DATA_DRIVEN_ND Generic DDAD fast/generic agreement in 3 dimensions.

    rng(1234,'twister');

    domain = [-1 1;0 2;-2 2];
    basis = build_random_weights_nd(25,domain,91);

    X = [ ...
        -1 + 2*rand(60,1), ...
         2*rand(60,1), ...
        -2 + 4*rand(60,1)];

    y = ...
        sin(1.3*X(:,1)) + ...
        0.2*X(:,2).^2 - ...
        0.1*cos(2.1*X(:,3));

    cache = prepare_data_cache(X,y,basis);

    p = [1.4;2.1;0.9];
    lambda = 1e-4;

    ls_opts.method = 'linsolve';
    ls_opts.use_gpu = false;
    ls_opts.compute_spectrum = false;

    activations = {'gaussian','tanh','sin'};

    for j = 1:numel(activations)

        activation = activations{j};

        [obj_g,grad_g,~] = ...
            evaluate_data_reduced_generic( ...
                p,cache,lambda,ls_opts,activation);

        [obj_f,grad_f,~] = ...
            evaluate_data_reduced_fast( ...
                p,cache,lambda,ls_opts,activation);

        rel_obj = abs(obj_f-obj_g)/max(1,abs(obj_g));
        rel_grad = norm(grad_f-grad_g)/max(1,norm(grad_g));

        fprintf( ...
            'DDAD 3-D %s: rel obj=%.3e, rel grad=%.3e\n', ...
            activation,rel_obj,rel_grad);

        assert(rel_obj < 1e-11);
        assert(rel_grad < 1e-10);

        h = 1e-6;
        grad_fd = zeros(size(p));

        for k = 1:numel(p)
            pp = p;
            pm = p;
            pp(k) = pp(k)+h;
            pm(k) = pm(k)-h;

            fp = evaluate_data_reduced_fast( ...
                pp,cache,lambda,ls_opts,activation);

            fm = evaluate_data_reduced_fast( ...
                pm,cache,lambda,ls_opts,activation);

            grad_fd(k) = (fp-fm)/(2*h);
        end

        rel_fd = norm(grad_f-grad_fd)/max(1,norm(grad_fd));

        fprintf('DDAD 3-D %s finite-difference grad diff=%.3e\n', ...
            activation,rel_fd);

        assert(rel_fd < 1e-5);
    end
end
