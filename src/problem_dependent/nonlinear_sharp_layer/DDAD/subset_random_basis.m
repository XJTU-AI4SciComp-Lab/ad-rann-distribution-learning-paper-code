function basis_train = subset_random_basis(basis,m_train)
%SUBSET_RANDOM_BASIS Use a deterministic subset of one full random basis.
%
% The first m_train columns are used.  This has two advantages:
%
%   1. reduced and full training use the SAME random realization;
%   2. switching reduced training off recovers the original full basis.
%
% prepare_data_cache and the Gaussian fast evaluators use basis.Z/basis.C.

    if ~isstruct(basis) || ...
            ~isfield(basis,'Z') || ...
            ~isfield(basis,'C')
        error('basis must contain Z and C.');
    end

    m_full = size(basis.Z,2);

    if ~isscalar(m_train) || ...
            m_train < 1 || ...
            m_train ~= floor(m_train) || ...
            m_train > m_full
        error('m_train must be an integer in [1,%d].',m_full);
    end

    basis_train = basis;

    basis_train.Z = basis.Z(:,1:m_train);
    basis_train.C = basis.C(:,1:m_train);
end
