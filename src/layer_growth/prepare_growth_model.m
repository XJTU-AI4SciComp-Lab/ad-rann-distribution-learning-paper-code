function model = ...
    prepare_growth_model(centers,directions,center_values)
%PREPARE_GROWTH_MODEL Store the geometry of one localized growth block.
%
% The growth features are
%
%   psi_j(x;rho) = exp(-rho^2 S_j(x)),
%
% with
%
%   S_j(x)
%     = ||beta_j||^2 (u_0(x)-u_0(c_j))^2
%       + 1/2 sum_k beta_{k,j}^2 (x_k-c_{j,k})^2.
%
% centers       : R-by-d
% directions    : d-by-R
% center_values : R-by-1, frozen base solution u_0(c_j)

    if nargin < 3
        error('prepare_growth_model requires centers, directions, center_values.');
    end

    [R,d] = size(centers);

    if ~isequal(size(directions),[d,R])
        error('directions must have size d-by-R.');
    end

    center_values = center_values(:);

    if numel(center_values) ~= R
        error('center_values must contain one value per center.');
    end

    if any(~isfinite(centers(:))) || ...
            any(~isfinite(directions(:))) || ...
            any(~isfinite(center_values))
        error('Growth model inputs must be finite.');
    end

    model = struct();

    model.centers = centers;
    model.directions = directions;
    model.direction_norm_sq = sum(directions.^2,1).';
    model.center_values = center_values;

    model.dim = d;
    model.num_features = R;
end
