% MIT License
% 
% Copyright (c) 2026 Shan Gai, Chenglong Lv, Chengjie Xie
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.

classdef QT
    %QT Quaternion Tensor Tools for vectorized operations.
    %   All inputs and outputs are numeric arrays where the last dimension 
    %   MUST be 4, representing the (a, b, c, d) components.
    
    methods (Static)
        %% --- Basic Construction & Separation ---
        
        function T = create(a, b, c, d)
            %CREATE Constructs a quaternion tensor from its four components.
            if isscalar(a), a = a * ones(size(b)); end
            if isscalar(c), c = c * ones(size(b)); end
            if isscalar(d), d = d * ones(size(b)); end
            dim = ndims(b) + 1;
            T = cat(dim, a, b, c, d);
        end
        
        function [a, b, c, d] = split(T)
            %SPLIT Extracts the four components from a quaternion tensor.
            if isempty(T), a=[]; b=[]; c=[]; d=[]; return; end
            if ~QT.isQuat(T), error('QT:InvalidInput', 'Last dimension must be 4.'); end
            dims = ndims(T); idx = repmat({':'}, 1, dims-1);
            a = T(idx{:}, 1); b = T(idx{:}, 2); c = T(idx{:}, 3); d = T(idx{:}, 4);
        end
        
        %% --- Algebra Operations ---
        
        function res = plus(T1, T2)
            %PLUS Adds two quaternion tensors or a tensor and a scalar.
            if isscalar(T2)
                res = T1; idx = repmat({':'}, 1, ndims(T1)-1); res(idx{:}, 1) = T1(idx{:}, 1) + T2;
            else
                if ~isequal(size(T1), size(T2)), error('QT:DimensionMismatch', 'Dims must match.'); end
                res = T1; idx = repmat({':'}, 1, ndims(T1)-1);
                for k = 1:4, res(idx{:}, k) = T1(idx{:}, k) + T2(idx{:}, k); end
            end
        end
        
        function res = minus(T1, T2)
            %MINUS Subtracts a quaternion tensor or scalar from another.
            if isscalar(T2)
                res = T1; idx = repmat({':'}, 1, ndims(T1)-1); res(idx{:}, 1) = T1(idx{:}, 1) - T2;
            else
                if ~isequal(size(T1), size(T2)), error('QT:DimensionMismatch', 'Dims must match.'); end
                res = T1; idx = repmat({':'}, 1, ndims(T1)-1);
                for k = 1:4, res(idx{:}, k) = T1(idx{:}, k) - T2(idx{:}, k); end
            end
        end
        
        function res = times(T, s)
            %TIMES Multiplies a quaternion tensor by a scalar.
            if ~isscalar(s), error('QT:InvalidInput', 'Multiplier must be scalar.'); end
            res = T; idx = repmat({':'}, 1, ndims(T)-1);
            for k = 1:4, res(idx{:}, k) = T(idx{:}, k) * s; end
        end
        
        function n = norm(T)
            %NORM Computes the element-wise magnitude across the last dimension.
            n = sqrt(sum(T.^2, ndims(T)));
        end
        
        function res = normalize(T)
            %NORMALIZE Normalizes the quaternion tensor along its last dimension.
            n = QT.norm(T); n(n == 0) = 1; idx = repmat({':'}, 1, ndims(T)-1);
            res = T; for k = 1:4, res(idx{:}, k) = T(idx{:}, k) ./ n; end
        end
        
        %% --- Low-Rank Matrix Operations ---
        
        function T_svt = svt(T_matrix, tau)
            %SVT Standard Singular Value Thresholding (Fixed Soft Thresholding).
            %   Implements: s_new = max(0, s - tau)
            [M, N, C] = size(T_matrix);
            if C ~= 4, error('QT:SVTInput', 'Input must be M-by-N-by-4.'); end
            
            [a, b, c, d] = QT.split(T_matrix);
            RealMat = [[a, -b, -c, -d]; [b,  a, -d,  c]; [c,  d,  a, -b]; [d, -c,  b,  a]];
            
            try
                [U, S, V] = svd(RealMat, 'econ');
            catch
                warning('QT:SVDConvergence', 'SVD did not converge. Returning zero matrix.');
                T_svt = zeros(size(T_matrix)); return;
            end
            
            s_val = diag(S);
            s_new = max(0, s_val - tau); 
            
            if sum(s_new) == 0, T_svt = zeros(size(T_matrix)); return; end
            
            RealMat_New = U * diag(s_new) * V';
            T_svt = cat(3, RealMat_New(1:M,1:N), RealMat_New(M+1:2*M,1:N), RealMat_New(2*M+1:3*M,1:N), RealMat_New(3*M+1:4*M,1:N));
        end
        
        function T_svt = adaptive_svt(T_matrix, keep_energy)
            %ADAPTIVE_SVT Adaptive Soft Thresholding based on energy retention.
            %   1. Finds the rank r that retains 'keep_energy' of the total energy.
            %   2. Sets an adaptive threshold tau based on the r-th singular value.
            %   3. Performs standard soft thresholding: s_new = max(0, s - tau).
            %   This strictly follows the QSVT formulation in the paper while 
            %   dynamically adapting the threshold to the data.
            
            [M, N, C] = size(T_matrix);
            if C ~= 4, error('QT:AdaptiveSVTInput', 'Input must be M-by-N-by-4.'); end
            
            [a, b, c, d] = QT.split(T_matrix);
            RealMat = [[a, -b, -c, -d]; [b,  a, -d,  c]; [c,  d,  a, -b]; [d, -c,  b,  a]];
            
            try
                [U, S, V] = svd(RealMat, 'econ');
            catch
                warning('QT:SVDConvergence', 'SVD did not converge. Returning zero matrix.');
                T_svt = zeros(size(T_matrix)); return;
            end
            
            s = diag(S);
            
            % 1. Determine the target rank r based on energy retention
            sorted_s = sort(s, 'descend');
            cum_energy = cumsum(sorted_s.^2) / sum(sorted_s.^2);
            r = find(cum_energy >= keep_energy, 1);
            if isempty(r), r = length(s); end
            r = min(r, length(s));
            
            % 2. Determine the adaptive soft threshold tau
            % Use the r-th singular value as the base and multiply by a factor (0.8)
            % to alleviate the over-shrinkage bias of soft thresholding.
            tau = sorted_s(r) * 0.8;
            
            % 3. Execute standard soft thresholding
            s_new = max(0, s - tau);
            
            if sum(s_new) == 0, T_svt = zeros(size(T_matrix)); return; end
            
            RealMat_New = U * diag(s_new) * V';
            T_svt = cat(3, RealMat_New(1:M,1:N), RealMat_New(M+1:2*M,1:N), RealMat_New(2*M+1:3*M,1:N), RealMat_New(3*M+1:4*M,1:N));
        end
        
        %% --- Utilities ---
        
        function T = clamp(T, min_v, max_v)
            %CLAMP Clips the values of a tensor to a specified range.
            T = max(min_v, min(max_v, T));
        end
        
        function tf = isQuat(T)
            %ISQUAT Checks if the last dimension of the input is 4.
            tf = (size(T, ndims(T)) == 4);
        end
    end
end