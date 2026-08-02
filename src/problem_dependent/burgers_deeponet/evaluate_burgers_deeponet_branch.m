function B = evaluate_burgers_deeponet_branch(F,rb,basis,activation)
%EVALUATE_BURGERS_DEEPONET_BRANCH Evaluate frozen branch random features.

    if size(F,2) ~= size(basis.Z,1)
        error('Branch input width does not match the frozen basis.');
    end

    B = activation_features( ...
        F,rb*ones(size(F,2),1),basis,activation);
end
