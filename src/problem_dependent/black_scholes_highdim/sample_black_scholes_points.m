function xt = sample_black_scholes_points(cfg,kind,n,stream)
%SAMPLE_BLACK_SCHOLES_POINTS Sample exact requested row counts.

    if nargin < 4 || isempty(stream)
        stream = RandStream.getGlobalStream();
    end

    d = cfg.dimension;
    lo = cfg.x_lower;
    hi = cfg.x_upper;
    t0 = cfg.t_domain(1);
    t1 = cfg.t_domain(2);

    if ~isscalar(n) || n < 1 || n ~= floor(n)
        error('n must be a positive integer.');
    end

    x = lo + (hi-lo)*rand(stream,n,d);

    switch lower(strtrim(char(kind)))

        case 'interior'
            t = t0 + (t1-t0)*rand(stream,n,1);

        case 'test'
            t = t0 + (t1-t0)*rand(stream,n,1);

        case 'initial'
            t = t0*ones(n,1);

        case 'boundary'
            t = t0 + (t1-t0)*rand(stream,n,1);
            face_dim = randi(stream,d,n,1);
            face_side = randi(stream,2,n,1);

            for j = 1:n
                if face_side(j) == 1
                    x(j,face_dim(j)) = lo;
                else
                    x(j,face_dim(j)) = hi;
                end
            end

        otherwise
            error('Unknown point kind: %s',kind);
    end

    xt = [x,t];
end
