function X = tensor_grid_nd(domain,n,inset)
%TENSOR_GRID_ND Tensor-product grid on a d-dimensional box.
%
%   X = tensor_grid_nd(domain,n)
%   X = tensor_grid_nd(domain,n,inset)
%
% Inputs
% ------
% domain : d-by-2 matrix
%
%          domain(k,1) = lower bound of dimension k
%          domain(k,2) = upper bound of dimension k
%
% n      : scalar or d-vector
%
%          scalar:
%              same number of points in every dimension
%
%          vector:
%              n(k) points in dimension k
%
% inset  : scalar or d-vector, optional
%
%          The grid in dimension k is generated on
%
%              [domain(k,1)+inset(k),
%               domain(k,2)-inset(k)].
%
% Output
% ------
% X      : N-by-d matrix, where
%
%              N = prod(n).
%
% Examples
% --------
%
% 1-D:
%
%   domain = [0 1];
%   X = tensor_grid_nd(domain,101,0);
%
%
% 2-D:
%
%   domain = [-1 1;
%             -1 1];
%
%   X = tensor_grid_nd(domain,[30 80],1e-6);
%
%
% Space-time:
%
%   domain = [-1 1;
%              0 1];
%
%   X = tensor_grid_nd(domain,[101 51],0);
%
%
% 3-D space-time:
%
%   domain = [-1 1;
%             -1 1;
%              0 1];
%
%   X = tensor_grid_nd(domain,[41 41 51],0);
%
% This routine is dimension independent.

    %% Defaults

    if nargin < 3 || isempty(inset)
        inset = 0;
    end


    %% Check domain

    if ~isnumeric(domain) || ...
            ndims(domain) ~= 2 || ...
            size(domain,2) ~= 2 || ...
            isempty(domain)

        error('domain must be a nonempty d-by-2 matrix.');

    end

    if any(~isfinite(domain(:)))
        error('domain contains NaN or Inf.');
    end

    if any(domain(:,1) >= domain(:,2))
        error('Each lower domain bound must be smaller than its upper bound.');
    end


    %% Dimension

    d = size(domain,1);


    %% Number of points

    if isscalar(n)

        n = repmat(n,d,1);

    else

        n = n(:);

    end

    if numel(n) ~= d
        error( ...
            'n must be scalar or have one entry per dimension.');
    end

    if any(~isfinite(n)) || ...
            any(n < 1) || ...
            any(n ~= round(n))

        error('All entries of n must be positive integers.');

    end


    %% Inset

    if isscalar(inset)

        inset = repmat(inset,d,1);

    else

        inset = inset(:);

    end

    if numel(inset) ~= d
        error( ...
            'inset must be scalar or have one entry per dimension.');
    end

    if any(~isfinite(inset)) || any(inset < 0)
        error('inset must contain finite nonnegative values.');
    end

    if any( ...
            domain(:,1)+inset >= ...
            domain(:,2)-inset)

        error('inset is too large for at least one dimension.');

    end


    %% Coordinate vectors

    coordinates = cell(d,1);

    for k = 1:d

        coordinates{k} = linspace( ...
            domain(k,1)+inset(k), ...
            domain(k,2)-inset(k), ...
            n(k));

    end


    %% Special case: 1-D

    if d == 1

        X = coordinates{1}(:);

        return;

    end


    %% Special case: 2-D
    %
    % Use meshgrid here so that the point ordering is exactly consistent
    % with the existing legacy tensor_grid.m implementation.

    if d == 2

        [X1,X2] = meshgrid( ...
            coordinates{1}, ...
            coordinates{2});

        X = [ ...
            X1(:), ...
            X2(:)];

        return;

    end


    %% General d-dimensional case
    %
    % For d >= 3, use ndgrid.

    G = cell(1,d);

    coordinate_row = coordinates.';

    [G{:}] = ndgrid(coordinate_row{:});


    %% Flatten

    N = prod(n);

    X = zeros(N,d);

    for k = 1:d

        X(:,k) = G{k}(:);

    end

end