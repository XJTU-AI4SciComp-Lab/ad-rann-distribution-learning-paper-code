function basis = build_burgers_deeponet_basis( ...
    num_sensors,num_branch,num_trunk,seed)
%BUILD_BURGERS_DEEPONET_BASIS Frozen paper-aligned random features.

    branch_domain = repmat([-1,1],num_sensors,1);

    % Trunk input is [cos(2*pi*x), sin(2*pi*x), t].
    trunk_domain = [-1,1;-1,1;0,1];

    basis = struct();
    basis.branch = build_random_weights_nd( ...
        num_branch,branch_domain,seed);
    basis.trunk = build_random_weights_nd( ...
        num_trunk,trunk_domain,seed+1);
    basis.seed = seed;
end
