function value = black_scholes_payoff(x,strike)
%BLACK_SCHOLES_PAYOFF psi(x)=max(max_i x_i-strike,0).

    if isempty(x) || ~ismatrix(x)
        error('x must be a nonempty N-by-d matrix.');
    end

    value = max(max(x,[],2)-strike,0);
    value = value(:);
end
