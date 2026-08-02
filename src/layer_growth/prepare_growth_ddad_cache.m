function cache = prepare_growth_ddad_cache( ...
    Xdata,ydata,base_features,base_solution,model)
%PREPARE_GROWTH_DDAD_CACHE Cache for data-driven growth-scale optimization.
%
% The data-driven reduced model is
%
%   [Phi_base, Psi(rho)] * w ~= y,
%
% where the base feature matrix is fixed and
%
%   Psi(rho) = exp(-rho^2 S).
%
% The expensive geometry S is assembled once here and reused for every Adam
% evaluation of rho.

    if nargin < 5
        error(['prepare_growth_ddad_cache requires Xdata, ydata, ', ...
               'base_features, base_solution, and model.']);
    end

    N = size(Xdata,1);

    ydata = ydata(:);
    base_solution = base_solution(:);

    if numel(ydata) ~= N || numel(base_solution) ~= N
        error('ydata and base_solution must have size N-by-1.');
    end

    if size(base_features,1) ~= N
        error('base_features must have N rows.');
    end

    if size(Xdata,2) ~= model.dim
        error('Xdata dimension does not match growth model.');
    end

    R = model.num_features;

    % Solution-space localization.
    du = base_solution-model.center_values.';
    S = du.^2 .* model.direction_norm_sq.';

    % Coordinate-space localization.
    for k = 1:model.dim
        dx = Xdata(:,k)-model.centers(:,k).';
        S = S + 0.5 * dx.^2 .* (model.directions(k,:).^2);
    end

    % Roundoff can only create tiny negative values.
    S = max(S,0);

    cache = struct();

    cache.base_features = base_features;
    cache.S = S;
    cache.y = ydata;

    cache.num_rows = N;
    cache.num_base_features = size(base_features,2);
    cache.num_growth_features = R;
end
