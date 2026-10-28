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

classdef QuaternionTensorCompleter_Vectorized < handle
    %QUATERNIONTENSORCOMPLETER_VECTORIZED Hybrid solver for QTR-PAR.
    %   Implements the ADMM-based optimization algorithm for Quaternion 
    %   Tensor Ring with Periodic Anchoring Regularization (QTR-PAR). 
    %
    %   Key Features:
    %   - SVT Modes: 'adaptive' (default, energy-based soft thresholding) 
    %     and 'fixed' (standard soft thresholding with a constant tau).
    %   - Periodic NSS (PAR): NSS weight is fixed at 0.5 for the first 50% of
    %     iterations, linearly decays to 0 between 50%~70%, and is disabled 
    %     after 70%. The NSS frequency doubles in the 50%~70% phase.

    properties
        params
        nss_module
    end

    methods
        function obj = QuaternionTensorCompleter_Vectorized(varargin)
            p = inputParser;
            p.addParameter('rho', 0.05, @isscalar);
            p.addParameter('max_iter', 200, @isscalar);
            p.addParameter('tol', 1e-4, @isscalar);
            p.addParameter('target_order', 6, @isscalar);
            p.addParameter('unfolding_shift', 3, @isscalar);
            p.addParameter('use_nss', true, @islogical);
            p.addParameter('nss_lambda', 0.1, @isscalar);
            p.addParameter('nss_gamma', 0.5, @isscalar);
            p.addParameter('nss_blk_size', [8,8,2], @isnumeric);
            p.addParameter('nss_step', [4,4,1], @isnumeric);
            p.addParameter('nss_start_iter', 6, @isscalar);
            p.addParameter('nss_freq', 5, @isscalar);
            p.addParameter('nss_search_freq', 1, @isscalar);
            p.addParameter('n_similar', 15, @isscalar);
            p.addParameter('search_win', 20, @isscalar);
            p.addParameter('tube_thresh_factor', 1.5, @isscalar);
            p.addParameter('true_data', [], @isnumeric);
            p.addParameter('eval_freq', 1, @isscalar);
            p.addParameter('log_file', '', @ischar);
            p.addParameter('verbose', false, @islogical);
            
            % === SVT Configuration ===
            p.addParameter('svt_mode', 'adaptive', @(x) ismember(x, {'fixed', 'adaptive'}));
            p.addParameter('svt_tau', 0.05, @isscalar);    % Used if svt_mode = 'fixed'
            p.addParameter('svt_energy', 0.9, @isscalar);  % Used if svt_mode = 'adaptive'

            parse(p, varargin{:});
            obj.params = p.Results;

            if obj.params.use_nss
                conf.gamma = obj.params.nss_gamma;
                conf.blk_size = obj.params.nss_blk_size;
                conf.step = obj.params.nss_step;
                conf.n_similar = obj.params.n_similar;
                conf.search_win = obj.params.search_win;
                conf.tube_thresh_factor = obj.params.tube_thresh_factor;
                conf.verbose = obj.params.verbose;
                obj.nss_module = TubalQuaternionNSS(conf);
            end
        end

        function [X_final, Info] = solve(obj, ObservedData, Mask)
            %SOLVE Execute the QTR-PAR algorithm.
            
            % 1. Q-OKA mapping to high-order representation
            [T_high, oka_params] = OKA_Vectorized(ObservedData, obj.params.target_order, obj.params.verbose);
            full_dims = size(T_high);
            K = length(full_dims) - 1;

            Mask_Double = double(Mask);
            [T_Mask_High, ~] = OKA_Vectorized(repmat(Mask_Double, [1 1 1 4]), obj.params.target_order, false);
            idx_m = repmat({':'}, 1, ndims(T_high)-1);
            Omega = T_Mask_High(idx_m{:}, 1) > 0.5;

            % 2. Initialize X with mean of observed pixels
            X = T_high;
            for c = 1:4
                chan = T_high(idx_m{:}, c);
                vals = chan(Omega);
                m_val = isempty(vals) * 0.5 + ~isempty(vals) * mean(vals);
                chan(~Omega) = m_val;
                X(idx_m{:}, c) = chan;
            end

            % 3. ADMM variable initialization
            M = cell(K, 1);  Y = cell(K, 1);
            s = obj.params.unfolding_shift;
            for k = 1:K
                unfolded = obj.circular_unfold(X, k, s);
                M{k} = unfolded;
                Y{k} = zeros(size(unfolded));
            end

            rho = obj.params.rho;
            max_iter = obj.params.max_iter;
            start_iter = obj.params.nss_start_iter;
            base_freq = obj.params.nss_freq;
            nss_call_count = 0;
            
            % SVT Params
            svt_mode = obj.params.svt_mode;
            svt_tau = obj.params.svt_tau;
            svt_energy = obj.params.svt_energy;

            % Best snapshot tracking
            best_psnr = -inf;
            best_X_video = [];
            true_video = obj.params.true_data;
            eval_freq = obj.params.eval_freq;

            % Logging setup
            log_file = obj.params.log_file;
            fid = -1;
            if ~isempty(log_file)
                fid = fopen(log_file, 'w');
                fprintf(fid, 'Iter\tRelativeError\tChangeRatio\tPSNR(dB)\n');
            end

            if ~isempty(true_video)
                true_norm = norm(true_video(:));
            else
                true_norm = [];
            end

            if obj.params.verbose
                fprintf('Starting QTR-PAR (Order=%d, rho=%.3f)\n', obj.params.target_order, rho);
                if strcmp(svt_mode, 'adaptive')
                    fprintf('  SVT Mode: ADAPTIVE SOFT (Energy = %.2f)\n', svt_energy);
                else
                    fprintf('  SVT Mode: FIXED SOFT (Tau = %.4f)\n', svt_tau);
                end
            end

            change_ratio = inf;
            iter = 0;

            for iter = 1:max_iter
                X_old = X;

                % ===== Part 1: TR Low-Rank Update =====
                for k = 1:K
                    Unfold_X = obj.circular_unfold(X, k, s);
                    B = Unfold_X + Y{k} ./ rho;
                    
                    if strcmp(svt_mode, 'adaptive')
                        % Adaptive soft thresholding (strictly follows the paper)
                        M{k} = QT.adaptive_svt(B, svt_energy);
                    else
                        % Fixed soft thresholding
                        M{k} = QT.svt(B, svt_tau);
                    end
                end

                % Fold back to X_TR
                Accumulator = zeros(size(X));
                for k = 1:K
                    Temp = M{k} - Y{k} ./ rho;
                    Accumulator = Accumulator + obj.circular_fold(Temp, full_dims, k, s);
                end
                X_TR = Accumulator ./ K;

                % ===== Part 2: NSS Local Enhancement (PAR mechanism) =====
                X_new = X_TR;
                if obj.params.use_nss
                    do_nss = false;
                    if iter >= start_iter
                        progress = iter / max_iter;
                        if progress <= 0.5
                            current_freq = base_freq;
                        elseif progress <= 0.7
                            current_freq = base_freq * 2;
                        else
                            current_freq = inf;
                        end
                        
                        if current_freq < inf && mod(iter - start_iter, current_freq) == 0
                            do_nss = true;
                        end
                    end

                    if do_nss
                        nss_call_count = nss_call_count + 1;
                        if obj.params.nss_search_freq == 1
                            do_search = true;
                        else
                            do_search = (mod(nss_call_count-1, obj.params.nss_search_freq) == 0);
                        end

                        X_video_pilot = InverseOKA_Vectorized(X_TR, oka_params, false);
                        X_video_pilot = QT.clamp(X_video_pilot, 0, 1);
                        X_video_nss = obj.nss_module.denoise(X_video_pilot, Mask, do_search, obj.params.nss_lambda);
                        
                        % NSS only affects missing regions
                        Mask_4D_nss = repmat(logical(Mask), [1, 1, 1, 4]);
                        X_video_nss(Mask_4D_nss) = X_video_pilot(Mask_4D_nss);
                        
                        [X_nss_high, ~] = OKA_Vectorized(X_video_nss, obj.params.target_order, false);

                        progress = iter / max_iter;
                        if progress <= 0.5
                            w_nss = 0.5;
                        elseif progress <= 0.7
                            w_nss = 0.5 * (1 - (progress - 0.5) / 0.2);
                        else
                            w_nss = 0;
                        end
                        
                        X_new = (1 - w_nss) * X_TR + w_nss * X_nss_high;
                    end
                end

                % ===== Part 3: Data Fidelity =====
                for c = 1:4
                    chan = X_new(idx_m{:}, c);
                    chan_t = T_high(idx_m{:}, c);
                    chan(Omega) = chan_t(Omega);
                    X_new(idx_m{:}, c) = chan;
                end
                X = X_new;

                % ===== Part 4: Dual Update =====
                for k = 1:K
                    Unfold_X = obj.circular_unfold(X, k, s);
                    Y{k} = Y{k} + rho * (Unfold_X - M{k});
                end

                % Convergence check
                diff = QT.norm(X - X_old);
                normX = QT.norm(X);
                change_ratio = norm(diff(:)) / (norm(normX(:)) + eps);

                % ===== Logging & Best Snapshot =====
                if fid > 0 && ~isempty(true_video)
                    X_curr_video = InverseOKA_Vectorized(X, oka_params, false);
                    X_curr_video = QT.clamp(X_curr_video, 0, 1);
                    rel_err = norm(X_curr_video(:) - true_video(:)) / true_norm;
                    gt_rgb = true_video(:,:,:,2:4);
                    rec_rgb = X_curr_video(:,:,:,2:4);
                    mse = mean((gt_rgb(:) - rec_rgb(:)).^2);
                    psnr_val = 10 * log10(1 / max(mse, eps));
                    fprintf(fid, '%d\t%.8f\t%.8f\t%.4f\n', iter, rel_err, change_ratio, psnr_val);
                end

                if ~isempty(true_video) && mod(iter, eval_freq) == 0
                    X_video_curr = InverseOKA_Vectorized(X, oka_params, false);
                    X_video_curr = QT.clamp(X_video_curr, 0, 1);
                    gt_rgb = true_video(:,:,:,2:4);
                    rec_rgb = X_video_curr(:,:,:,2:4);
                    mse_curr = mean((gt_rgb(:) - rec_rgb(:)).^2);
                    psnr_curr = 10 * log10(1 / max(mse_curr, eps));
                    if psnr_curr > best_psnr
                        best_psnr = psnr_curr;
                        best_X_video = X_video_curr;
                    end
                end

                if obj.params.verbose && mod(iter, 5) == 0
                    fprintf('  Iter %d: Change Ratio = %.6f\n', iter, change_ratio);
                end

                if change_ratio < obj.params.tol
                    break;
                end
            end

            if fid > 0
                fclose(fid);
            end

            if ~isempty(best_X_video)
                X_final = best_X_video;
            else
                X_final = InverseOKA_Vectorized(X, oka_params, false);
                X_final = QT.clamp(X_final, 0, 1);
            end

            Info.err = change_ratio;
            Info.iter = iter;
            Info.best_psnr = best_psnr;
        end

        function Mat = circular_unfold(~, T, k, s)
            %CIRCULAR_UNFOLD Circular mode-(k,s) unfolding for quaternion tensors.
            sz = size(T); K = length(sz)-1;
            row_dims = mod((k:k+s-1)-1, K)+1;
            col_dims = setdiff(1:K, row_dims);
            perm_order = [row_dims, col_dims, K+1];
            Mat = reshape(permute(T, perm_order), [prod(sz(row_dims)), prod(sz(col_dims)), 4]);
        end

        function T = circular_fold(~, Mat, full_dims, k, s)
            %CIRCULAR_FOLD Inverse operation of circular_unfold.
            sz = full_dims; K = length(sz)-1;
            row_dims = mod((k:k+s-1)-1, K)+1;
            col_dims = setdiff(1:K, row_dims);
            perm_order = [row_dims, col_dims, K+1];
            T = ipermute(reshape(Mat, full_dims(perm_order)), perm_order);
        end
    end
end