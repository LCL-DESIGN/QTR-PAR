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

classdef TubalQuaternionNSS < handle
    %TUBALQUATERNIONNSS Non-Local Self-Similarity (NSS) Denoiser for Quaternion Tensors.
    %   This class implements a V-BM3D style block matching and collaborative 
    %   filtering for multi-channel data represented as quaternions.
    %   It caches the block-matching trajectories to accelerate subsequent calls.
    
    properties
        blk_size    = [8, 8, 2];    
        step        = [4, 4, 1];    
        search_win  = 20;           
        n_similar   = 10;           
        lambda              = 0.1;
        gamma               = 0.5;
        tube_thresh_factor  = 1.5; 
        keep_energy         = 0.90; % Retain 90% of singular value energy
        cached_trajectories = {};   
        is_initialized      = false;
        verbose             = false;
    end
    
    methods
        function obj = TubalQuaternionNSS(varargin)
            %TUBALQUATERNIONNSS Constructor supporting struct input.
            if nargin > 0 && isstruct(varargin{1})
                params = varargin{1};
                f = fieldnames(params);
                for i = 1:length(f)
                    if isprop(obj, f{i}), obj.(f{i}) = params.(f{i}); end
                end
            end
        end
        
        function X_out = denoise(obj, X_in, Mask_in, do_search, lambda_in)
            %DENOISE Perform NSS-based denoising on quaternion tensor.
            if nargin >= 5, obj.lambda = lambda_in; end
            
            [H_orig, W_orig, F_orig, ~] = size(X_in);
            bs = obj.blk_size;
            
            pad_h = obj.search_win; 
            pad_w = obj.search_win; 
            pad_f = 0;
            X_pad = padarray(X_in, [pad_h, pad_w, pad_f, 0], 'symmetric', 'both');
            
            if isempty(Mask_in)
                Mask_tmp = ones(H_orig, W_orig, F_orig);
            else
                Mask_tmp = double(Mask_in);
            end
            Mask_pad = padarray(Mask_tmp, [pad_h, pad_w, pad_f], 0); 
            [H, W, F, ~] = size(X_pad);
            
            h_grid = 1 : obj.step(1) : H - bs(1) + 1;
            w_grid = 1 : obj.step(2) : W - bs(2) + 1;
            f_grid = 1 : obj.step(3) : F - bs(3) + 1;
            [HH, WW, FF] = ndgrid(h_grid, w_grid, f_grid);
            ref_blocks = [HH(:), WW(:), FF(:)];
            num_ref = size(ref_blocks, 1);
            
            % ---------- Block Matching (with caching) ----------
            if do_search || ~obj.is_initialized
                if obj.verbose
                    fprintf('    [NSS] Executing block matching (blocks: %d) ...\n', num_ref);
                end
                
                par_params.bs = bs;
                par_params.search_win = obj.search_win;
                par_params.n_similar = obj.n_similar;
                par_params.size_X = [H, W, F];
                
                % Check if Parallel Computing Toolbox is available
                has_parallel = license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
                
                temp_traj = cell(num_ref, 1);
                if has_parallel
                    if isempty(gcp('nocreate')), parpool; end
                    parfor k = 1:num_ref
                        temp_traj{k} = TubalQuaternionNSS.block_matching_hybrid_fast(ref_blocks(k,:), X_pad, Mask_pad, par_params);
                    end
                else
                    % Graceful degradation to sequential loop
                    for k = 1:num_ref
                        temp_traj{k} = TubalQuaternionNSS.block_matching_hybrid_fast(ref_blocks(k,:), X_pad, Mask_pad, par_params);
                    end
                end
                obj.cached_trajectories = temp_traj;
                obj.is_initialized = true;
            else
                if obj.verbose
                    fprintf('    [NSS] Reusing cached trajectories (skipping search)\n');
                end
            end
            
            % ---------- Collaborative Filtering & Aggregation ----------
            X_out_pad = obj.collaborative_filtering_aggregation(X_pad, obj.cached_trajectories, H, W, F, Mask_pad);
            X_out = X_out_pad(pad_h+1:pad_h+H_orig, pad_w+1:pad_w+W_orig, :, :);
        end
        
        function X_rec = collaborative_filtering_aggregation(obj, X_pad, trajectories, H, W, F, Mask_pad)
            %COLLABORATIVE_FILTERING_AGGREGATION Aggregates denoised blocks via weighted averaging.
            Numerator = zeros(size(X_pad), 'like', X_pad);   
            Denominator = zeros(size(X_pad), 'like', X_pad); 
            bs = obj.blk_size;
            
            for k = 1:length(trajectories)
                locs = trajectories{k};
                if isempty(locs), continue; end
                n_sim = length(locs);
                
                Group = zeros([bs(1), bs(2), bs(3), n_sim, 4], 'like', X_pad);
                GrpMask = zeros([bs(1), bs(2), bs(3), n_sim]);
                
                for i = 1:n_sim
                    [rh, rw, rf] = ind2sub([H, W, F], locs(i));
                    Group(:,:,:,i,:) = X_pad(rh:rh+bs(1)-1, rw:rw+bs(2)-1, rf:rf+bs(3)-1, :);
                    GrpMask(:,:,:,i) = Mask_pad(rh:rh+bs(1)-1, rw:rw+bs(2)-1, rf:rf+bs(3)-1);
                end
                
                valid_sum = sum(GrpMask, 4);
                valid_sum(valid_sum == 0) = 1; 
                
                for c = 1:4
                    TubeData = Group(:,:,:,:,c); 
                    weighted_sum = sum(TubeData .* GrpMask, 4);
                    local_mean = weighted_sum ./ valid_sum;
                    local_mean_rep = repmat(local_mean, [1, 1, 1, n_sim]);
                    TubeData(GrpMask == 0) = local_mean_rep(GrpMask == 0);
                    Group(:,:,:,:,c) = TubeData;
                end
                
                [d1, d2, d3, ~, ~] = size(Group);
                CleanGroup = Group;
                weight = 1.0; 
                
                for c = 1:4
                    Mat = reshape(Group(:,:,:,:,c), [d1*d2*d3, n_sim]);
                    [U, S, V] = svd(Mat, 'econ');
                    s_val = diag(S);
                    
                    total_energy = sum(s_val.^2);
                    if total_energy == 0
                        CleanGroup(:,:,:,:,c) = 0;
                        continue;
                    end
                    sq_s = s_val.^2;
                    energy_ratio = cumsum(sq_s) / total_energy;
                    n_keep = find(energy_ratio >= obj.keep_energy, 1);
                    if isempty(n_keep), n_keep = 1; end
                    n_keep = max(1, n_keep); 
                    
                    s_new = zeros(size(s_val));
                    s_new(1:n_keep) = s_val(1:n_keep);
                    Mat_New = U * diag(s_new) * V';
                    CleanGroup(:,:,:,:,c) = reshape(Mat_New, [d1, d2, d3, n_sim]);
                end
                
                for i = 1:n_sim
                    [rh, rw, rf] = ind2sub([H, W, F], locs(i));
                    patch = reshape(CleanGroup(:,:,:,i,:), [bs(1), bs(2), bs(3), 4]);
                    rr = rh:rh+bs(1)-1; 
                    cc = rw:rw+bs(2)-1; 
                    ff = rf:rf+bs(3)-1;
                    Numerator(rr, cc, ff, :) = Numerator(rr, cc, ff, :) + patch * weight;
                    Denominator(rr, cc, ff, :) = Denominator(rr, cc, ff, :) + weight;
                end
            end
            
            Uncovered = (Denominator == 0);
            Denominator(Uncovered) = 1;
            X_rec = Numerator ./ Denominator;
            X_rec(Uncovered) = X_pad(Uncovered);
        end
    end
    
    methods(Static)
        function best_locs = block_matching_hybrid_fast(ref_coord, X_pad, Mask_pad, p)
            %BLOCK_MATCHING_HYBRID_FAST Optimized block matching with pre-allocation.
            ch = ref_coord(1); cw = ref_coord(2); cf = ref_coord(3);
            bs = p.bs;
            ref_patch = X_pad(ch:ch+bs(1)-1, cw:cw+bs(2)-1, cf:cf+bs(3)-1, :);
            ref_mask  = Mask_pad(ch:ch+bs(1)-1, cw:cw+bs(2)-1, cf:cf+bs(3)-1);
            ref_vec = reshape(ref_patch, [], 4);
            ref_mask_vec = ref_mask(:);
            
            [H, W, F, ~] = size(X_pad);
            h_min = max(1, ch - p.search_win); h_max = min(H-bs(1)+1, ch + p.search_win);
            w_min = max(1, cw - p.search_win); w_max = min(W-bs(2)+1, cw + p.search_win);
            f_min = max(1, cf - 2);            f_max = min(F-bs(3)+1, cf + 2); 
            
            % Pre-allocate memory to avoid dynamic resizing (critical for performance)
            max_candidates = length(h_min:3:h_max) * length(w_min:3:w_max) * length(f_min:f_max);
            dists = zeros(max_candidates, 1);
            locs = zeros(max_candidates, 1);
            count = 0;
            
            for hh = h_min : 3 : h_max
                for ww = w_min : 3 : w_max
                    for ff = f_min : f_max
                        target_mask = Mask_pad(hh:hh+bs(1)-1, ww:ww+bs(2)-1, ff:ff+bs(3)-1);
                        target_mask_vec = target_mask(:);
                        target_patch = X_pad(hh:hh+bs(1)-1, ww:ww+bs(2)-1, ff:ff+bs(3)-1, :);
                        target_vec = reshape(target_patch, [], 4);
                        
                        diff = ref_vec - target_vec;
                        sq_diff = sum(diff.^2, 2); 
                        mask_prod = ref_mask_vec .* target_mask_vec;
                        w_vec = 0.5 + 0.5 * mask_prod;
                        
                        d = sum(sq_diff .* w_vec) / (sum(w_vec) + 1e-6);
                        
                        count = count + 1;
                        dists(count) = d;
                        locs(count) = sub2ind([H,W,F], hh, ww, ff);
                    end
                end
            end
            
            % Truncate pre-allocated arrays if count < max_candidates
            dists = dists(1:count);
            locs = locs(1:count);
            
            if isempty(dists)
                best_locs = sub2ind([H,W,F], ch, cw, cf);
            else
                [~, idx] = sort(dists, 'ascend');
                n_keep = min(length(idx), p.n_similar);
                best_locs = locs(idx(1:n_keep));
            end
        end
    end
end