function [M,y,parts] = assemble_poisson_lshape_mixture_matrix( ...
        p,problem,basis,activation)
%ASSEMBLE_POISSON_LSHAPE_MIXTURE_MATRIX Assemble the additive global model.
%
% No artificial interface rows are required: both feature blocks are
% evaluated over the whole L-shaped domain and their sum is one smooth
% global approximation.

    p = p(:);
    if numel(p) ~= 4
        error('Mixture parameter p must have four components.');
    end

    [Mnear,~,near_parts] = assemble_poisson_lshape_matrix( ...
        p(1:2),problem,basis.near,activation);
    [Mfar,y,far_parts] = assemble_poisson_lshape_matrix( ...
        p(3:4),problem,basis.far,activation);

    M = [Mnear,Mfar];

    if nargout >= 3
        parts.Mi = [near_parts.Mi,far_parts.Mi];
        parts.Mb = [near_parts.Mb,far_parts.Mb];
        parts.near = near_parts;
        parts.far = far_parts;
    end
end
