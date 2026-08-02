function basis = build_poisson_lshape_mixture_basis( ...
        num_near,num_far,domain,seed,radius_split,tol)
%BUILD_POISSON_LSHAPE_MIXTURE_BASIS Two center-conditioned Gaussian blocks.
%
% The first block has centers strictly inside r<radius_split and the
% second block strictly inside r>radius_split.  Both sets remain in the
% true L-shaped domain.  Random directions match a same-seed global basis
% with num_near+num_far features, which makes the comparison controlled.

    if nargin < 6 || isempty(tol)
        tol = 1e-12;
    end

    validateattributes(num_near,{'numeric'},{'scalar','integer','positive'});
    validateattributes(num_far,{'numeric'},{'scalar','integer','positive'});
    validateattributes(radius_split,{'numeric'},{'scalar','positive'});

    total = num_near+num_far;
    frozen = build_random_weights_nd(total,domain,seed);

    near = frozen;
    near.Z = frozen.Z(:,1:num_near);
    near.C = sample_conditioned_centers( ...
        num_near,domain,seed+7919,radius_split,'near',tol);
    near.num_features = num_near;
    near.center_geometry = 'L_shape_r_less_than_split';

    far = frozen;
    far.Z = frozen.Z(:,num_near+1:end);
    far.C = sample_conditioned_centers( ...
        num_far,domain,seed+15838,radius_split,'far',tol);
    far.num_features = num_far;
    far.center_geometry = 'L_shape_r_greater_than_split';

    assert(all(poisson_lshape_is_inside(near.C.',false,tol)));
    assert(all(poisson_lshape_is_inside(far.C.',false,tol)));
    assert(all(vecnorm(near.C,2,1) < radius_split));
    assert(all(vecnorm(far.C,2,1) > radius_split));

    basis = struct();
    basis.near = near;
    basis.far = far;
    basis.num_near = num_near;
    basis.num_far = num_far;
    basis.num_features = total;
    basis.radius_split = radius_split;
    basis.seed = seed;
    basis.parameter_order = { ...
        'r_x_near','r_y_near','r_x_far','r_y_far'};
end


function C = sample_conditioned_centers( ...
        count,domain,seed,radius_split,region,tol)

    stream = RandStream('mt19937ar','Seed',seed);
    C = zeros(2,count);
    filled = 0;

    while filled < count
        remaining = count-filled;
        batch_size = max(4096,ceil(remaining*300));
        U = rand(stream,batch_size,2);
        X = domain(:,1).' + U.*(domain(:,2)-domain(:,1)).';

        inside = poisson_lshape_is_inside(X,false,tol);
        radius = vecnorm(X,2,2);

        switch region
            case 'near'
                accept = inside & radius < radius_split;
            case 'far'
                accept = inside & radius > radius_split;
            otherwise
                error('Unknown center region: %s',region);
        end

        accepted = X(accept,:);
        take = min(size(accepted,1),remaining);

        if take > 0
            C(:,filled+(1:take)) = accepted(1:take,:).';
            filled = filled+take;
        end
    end
end
