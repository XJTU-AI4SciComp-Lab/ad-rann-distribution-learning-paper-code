function T = evaluate_burgers_deeponet_trunk(Y,rt,basis,activation)
%EVALUATE_BURGERS_DEEPONET_TRUNK Periodic hard-constraint trunk features.

    if size(Y,2) ~= 2
        error('Y must be N-by-2 with columns [x,t].');
    end

    rt = rt(:);

    if numel(rt) ~= 3 || any(~isfinite(rt)) || any(rt <= 0)
        error('rt=[r_cos;r_sin;r_t] must contain three positive values.');
    end

    x = Y(:,1);
    t = Y(:,2);
    periodic_input = [cos(2*pi*x),sin(2*pi*x),t];

    T = activation_features( ...
        periodic_input,rt,basis,activation);
end
