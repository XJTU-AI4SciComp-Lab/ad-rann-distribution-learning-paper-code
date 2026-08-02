function basis = build_random_weights(num_features, domain, seed)
%BUILD_RANDOM_WEIGHTS Generate one frozen 2D random-feature realization.
% domain = [xmin xmax; ymin ymax].
% Z ~ U(-1,1), centers C are uniform in the physical domain.
%
% IMPORTANT:
% The random draw shape/order is intentionally kept identical to the
% reference implementation. With a fixed seed, changing a 2-by-m draw into
% two separate 1-by-m draws changes the actual realization.

    if numel(domain) == 4 && ~isequal(size(domain),[2,2])
        domain = [domain(1:2); domain(3:4)];
    end

    if ~isequal(size(domain),[2,2])
        error('domain must be [xmin xmax; ymin ymax].');
    end

    stream = RandStream('mt19937ar','Seed',seed);

    % Keep these two matrix draws exactly in this order.
    Z = 2*rand(stream,2,num_features)-1;
    U = rand(stream,2,num_features);

    C = zeros(2,num_features);

    C(1,:) = ...
        domain(1,1) + ...
        (domain(1,2)-domain(1,1))*U(1,:);

    C(2,:) = ...
        domain(2,1) + ...
        (domain(2,2)-domain(2,1))*U(2,:);

    basis.Z = Z;
    basis.C = C;
    basis.num_features = num_features;
    basis.seed = seed;
    basis.domain = domain;
end
