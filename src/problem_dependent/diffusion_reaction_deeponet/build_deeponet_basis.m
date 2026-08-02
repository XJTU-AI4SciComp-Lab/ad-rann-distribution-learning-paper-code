function basis = build_deeponet_basis(num_sensors,num_branch,num_trunk,seed)
%BUILD_DEEPONET_BASIS Frozen branch/trunk random-feature realizations.

    if ~isscalar(num_sensors) || num_sensors < 1 || ...
            num_sensors ~= floor(num_sensors)
        error('num_sensors must be a positive integer.');
    end

    branch_domain = repmat([-1,1],num_sensors,1);
    trunk_domain = [0,1;0,1];

    basis = struct();
    basis.branch = build_random_weights_nd( ...
        num_branch,branch_domain,seed);
    basis.trunk = build_random_weights_nd( ...
        num_trunk,trunk_domain,seed+1);
    basis.num_sensors = num_sensors;
    basis.num_branch = num_branch;
    basis.num_trunk = num_trunk;
    basis.seed = seed;
end
