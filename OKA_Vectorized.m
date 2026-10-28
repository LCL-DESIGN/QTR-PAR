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

function [T_out, params] = OKA_Vectorized(M_in, target_order, verbose)
    %OKA_VECTORIZED Vectorized Quaternion Overlapping Ket Augmentation (Q-OKA).
    %   T_out = OKA_VECTORIZED(M_in, target_order) maps the input tensor M_in 
    %   to a high-order tensor by recursively performing multi-level adaptive 
    %   overlap extraction and orthogonal mode stacking.
    %
    %   Inputs:
    %       M_in         - Input tensor of size (H x W x ... x T).
    %       target_order - Desired number of dimensions.
    %       verbose      - (Optional) Logical flag to print execution details. Default is false.
    %
    %   Outputs:
    %       T_out        - High-order output tensor.
    %       params       - Struct containing the detailed block indices and dimension info.

    if nargin < 3
        verbose = false;
    end
    
    if ~isnumeric(M_in) || isempty(M_in)
        error('OKA:InvalidInput', 'M_in must be a non-empty numeric tensor.');
    end
    
    sz = size(M_in);
    dims = sz(1:end-1);
    current_order = length(dims);
    
    if current_order < 2
        error('OKA:InvalidInput', 'Input tensor must have at least 2 spatial dimensions.');
    end
    
    if target_order <= current_order
        warning('OKA:TargetOrder', 'Target order is <= current order. Returning input as is.');
        T_out = M_in;
        params = struct('steps', 0, 'final_order', current_order, 'history', []);
        return;
    end

    current_tensor = M_in;
    params = struct();
    params.history = struct('H', {}, 'W', {}, 'block_r', {}, 'block_c', {}, 'indices', {}, 'dim_idx', {});
    step = 0;
    
    if verbose
        fprintf('Starting vectorized OKA: %s -> Target %d-order\n', mat2str(dims), target_order);
    end
    
    while current_order < target_order
        step = step + 1;
        sz_curr = size(current_tensor);
        H_dim_idx = step; 
        W_dim_idx = step + 1;
        H = sz_curr(H_dim_idx);
        W = sz_curr(W_dim_idx);
        
        if H <= 4 || W <= 4
            % Rollback step because the spatial dimension is too small
            step = step - 1;
            if verbose
                fprintf('  Spatial dims (%dx%d) <= 4. Stopping recursion.\n', H, W);
            end
            break;
        end
        
        overlap_r = 2 + mod(H, 2); 
        overlap_c = 2 + mod(W, 2);
        block_r = ceil((H + overlap_r) / 2);
        block_c = ceil((W + overlap_c) / 2);
        
        r1_s = 1;               r1_e = block_r;
        r2_s = H - block_r + 1; r2_e = H;
        c1_s = 1;               c1_e = block_c;
        c2_s = W - block_c + 1; c2_e = W;
        
        % Store parameters for inverse mapping (adjoint operator)
        params.history(step).H = H; 
        params.history(step).W = W;
        params.history(step).block_r = block_r;
        params.history(step).block_c = block_c;
        params.history(step).indices = [r1_s, r1_e, c1_s, c1_e; 
                                        r2_s, r2_e, c1_s, c1_e;
                                        r1_s, r1_e, c2_s, c2_e;
                                        r2_s, r2_e, c2_s, c2_e];
        params.history(step).dim_idx = [H_dim_idx, W_dim_idx];
        
        idx_base = repmat({':'}, 1, ndims(current_tensor));
        
        idx = idx_base; idx{H_dim_idx} = r1_s:r1_e; idx{W_dim_idx} = c1_s:c1_e;
        B1 = current_tensor(idx{:});
        
        idx = idx_base; idx{H_dim_idx} = r2_s:r2_e; idx{W_dim_idx} = c1_s:c1_e;
        B2 = current_tensor(idx{:});
        
        idx = idx_base; idx{H_dim_idx} = r1_s:r1_e; idx{W_dim_idx} = c2_s:c2_e;
        B3 = current_tensor(idx{:});
        
        idx = idx_base; idx{H_dim_idx} = r2_s:r2_e; idx{W_dim_idx} = c2_s:c2_e;
        B4 = current_tensor(idx{:});
        
        % Stack into a new mode
        temp_cat = cat(ndims(current_tensor)+1, B1, B2, B3, B4);
        n_dims = ndims(temp_cat);
        
        % Move the new mode to the front to construct the '4 x 4 x ... x t' structure
        perm_order = [n_dims, 1:n_dims-2, n_dims-1];
        current_tensor = permute(temp_cat, perm_order);
        current_order = current_order + 1;
        
        if verbose
            fprintf('  Step %d: Tensor size updated to %s\n', step, mat2str(size(current_tensor)));
        end
    end
    
    T_out = current_tensor;
    params.final_order = current_order;
    params.steps = step;
    
    if verbose
        fprintf('OKA completed. Output size: %s\n', mat2str(size(T_out)));
    end
end