function u = predict_deeponet_rows(F,Y,model,activation)
%PREDICT_DEEPONET_ROWS Evaluate paired (input function, coordinate) rows.

    if nargin < 4 || isempty(activation)
        activation = 'tanh';
    end

    if size(F,1) ~= size(Y,1)
        error('F and Y must contain the same number of paired rows.');
    end

    B = evaluate_deeponet_branch( ...
        F,model.p(1),model.basis.branch,activation);
    T = evaluate_deeponet_trunk( ...
        Y,model.p(2:3),model.basis.trunk,activation);

    u = sum((B.value*model.W).*T.value,2);
end
