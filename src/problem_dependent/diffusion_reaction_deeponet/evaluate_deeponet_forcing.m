function rhs = evaluate_deeponet_forcing(F,x)
%EVALUATE_DEEPONET_FORCING Linearly interpolate each input function at x.
%
% Sensors are assumed uniformly distributed on [0,1].

    x = x(:);

    if size(F,1) ~= numel(x)
        error('F and x must have the same row count.');
    end

    m = size(F,2);

    if m < 2
        error('At least two branch sensors are required.');
    end

    q = min(max(x,0),1)*(m-1)+1;
    j0 = floor(q);
    j1 = min(j0+1,m);
    a = q-j0;

    rows = (1:size(F,1)).';
    i0 = sub2ind(size(F),rows,j0);
    i1 = sub2ind(size(F),rows,j1);

    rhs = (1-a).*F(i0)+a.*F(i1);
end
