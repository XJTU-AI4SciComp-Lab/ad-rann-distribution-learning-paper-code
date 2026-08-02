function basis = build_poisson_lshape_basis(num_features,domain,seed,tol)
%BUILD_POISSON_LSHAPE_BASIS Frozen random directions with centers in Omega.
%
% build_random_weights_nd is used for the common basis structure and the
% frozen random directions. The rectangular centers are replaced by points
% sampled uniformly from the three unit squares composing the L-shape.

    if nargin < 4 || isempty(tol)
        tol = 1e-12;
    end

    basis = build_random_weights_nd(num_features,domain,seed);

    stream = RandStream('mt19937ar','Seed',seed+7919);

    block = floor(3*rand(stream,1,num_features))+1;
    U = rand(stream,2,num_features);

    C = zeros(2,num_features);

    % Bottom-left square: [-1,0] x [-1,0].
    idx = block == 1;
    C(1,idx) = -1 + U(1,idx);
    C(2,idx) = -1 + U(2,idx);

    % Top-left square: [-1,0] x [0,1].
    idx = block == 2;
    C(1,idx) = -1 + U(1,idx);
    C(2,idx) = U(2,idx);

    % Top-right square: [0,1] x [0,1].
    idx = block == 3;
    C(1,idx) = U(1,idx);
    C(2,idx) = U(2,idx);

    if ~all(poisson_lshape_is_inside(C.',false,tol))
        error('The L-shaped basis sampler generated an invalid center.');
    end

    basis.C = C;
    basis.center_geometry = 'uniform_L_shape';
end
