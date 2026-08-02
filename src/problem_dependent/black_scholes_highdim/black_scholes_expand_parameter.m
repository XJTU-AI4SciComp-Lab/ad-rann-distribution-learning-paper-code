function p_full = black_scholes_expand_parameter(p,d)
%BLACK_SCHOLES_EXPAND_PARAMETER [r_s;r_t] -> [r_s,...,r_s,r_t].

    p = p(:);

    if numel(p) ~= 2
        error('p must equal [r_s;r_t].');
    end

    p_full = [p(1)*ones(d,1);p(2)];
end
