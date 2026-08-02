function value = monte_carlo_black_scholes_reference(xt,cfg,seed,label)
%MONTE_CARLO_BLACK_SCHOLES_REFERENCE Fixed-seed Feynman-Kac estimator.
%
% One common Ns-by-d standard-normal matrix is reused for every queried
% point.  This is a valid common-random-number estimator and makes saved
% data exactly reproducible.

    if nargin < 4 || isempty(label)
        label = 'reference';
    end

    d = cfg.dimension;

    if size(xt,2) ~= d+1
        error('xt must have d+1 columns.');
    end

    x = double(xt(:,1:d));
    t = double(xt(:,end));

    Ns = cfg.data.num_mc_samples;
    batch_size = cfg.data.mc_point_batch_size;

    mu = cfg.mu;
    sigma = cfg.sigma(:);
    strike = cfg.payoff_strike;

    stream = RandStream('mt19937ar','Seed',seed);
    Z = randn(stream,Ns,d);

    N = size(x,1);
    value = zeros(N,1);

    num_batches = ceil(N/batch_size);

    for batch = 1:num_batches
        first = (batch-1)*batch_size+1;
        rows = first:min(first+batch_size-1,N);

        xb = x(rows,:);
        tb = max(t(rows),0);
        nb = numel(rows);

        max_terminal = -inf(nb,Ns);
        sqrt_t = sqrt(tb);

        for i = 1:d
            deterministic = (mu-0.5*sigma(i)^2)*tb;
            stochastic = (sigma(i)*sqrt_t)*Z(:,i).';
            terminal_i = xb(:,i).*exp(deterministic+stochastic);
            max_terminal = max(max_terminal,terminal_i);
        end

        payoff = max(max_terminal-strike,0);
        value(rows) = mean(payoff,2);

        if batch == 1 || batch == num_batches || mod(batch,100) == 0
            fprintf('[%s] batch %d/%d, points %d/%d\n', ...
                label,batch,num_batches,rows(end),N);
        end
    end
end
