function [theta_new, m_new, v_new, step] = adam_step(theta, grad, m, v, t, opts)
%ADAM_STEP One standard Adam update.
%
% opts.learning_rate may be scalar or have one entry per parameter.

    lr = get_opt(opts,'learning_rate',1e-3);
    b1 = get_opt(opts,'beta1',0.9);
    b2 = get_opt(opts,'beta2',0.999);
    eps0 = get_opt(opts,'epsilon',1e-8);

    sz = size(theta);

    theta = theta(:);
    grad = grad(:);
    m = m(:);
    v = v(:);

    if ~isscalar(lr)
        lr = lr(:);

        if numel(lr) ~= numel(theta)
            error('learning_rate must be scalar or match theta.');
        end
    end

    m_new = b1*m + (1-b1)*grad;
    v_new = b2*v + (1-b2)*(grad.^2);

    m_hat = m_new/(1-b1^t);
    v_hat = v_new/(1-b2^t);

    step = lr.*m_hat./(sqrt(max(v_hat,0))+eps0);

    theta_new = theta-step;

    theta_new = reshape(theta_new,sz);
    m_new = reshape(m_new,sz);
    v_new = reshape(v_new,sz);
    step = reshape(step,sz);
end


function value = get_opt(opts,name,default_value)

    if isfield(opts,name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default_value;
    end
end
