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

%% Demo_Video_TC.m
% This script demonstrates the tensor completion process for video data.
% It treats all images in the 'data' folder as sequential video frames,
% constructs a third-order quaternion tensor, and uses the QTR-PAR solver
% for holistic completion. Results and QA tables are saved to the output folder.
%
% Note: This script will automatically use the external 'Img_QA' function 
% if it is available in the path; otherwise, it silently falls back to the 
% built-in FSIM implementation provided at the end of this file.

clear; clc; close all;

% Add dependencies (Ensure required folders exist)
if isfolder('lib')
    addpath(genpath('lib'));
end

%% Configuration Parameters
data_folder = 'data';
result_root = 'results/Video_TC';
missing_rates = [0.7];   % Missing rate (e.g., 70% missing)
method_name = 'Ours';

% Solver parameters (tuned for video tensor)
solver_params = struct(...
    'rho', 0.001, ...
    'max_iter', 100, ...
    'tol', 1e-4, ...
    'target_order', 8, ...
    'unfolding_shift', 4, ...
    'use_nss', true, ...
    'nss_lambda', 0.5, ...
    'nss_gamma', 0.6, ...
    'nss_start_iter', 1, ...
    'nss_freq', 5, ...
    'n_similar', 15, ...
    'search_win', 20, ...
    'nss_blk_size', [8, 8, 1], ...   % Intra-frame block matching (3rd dim = 1)
    'nss_step', [4, 4, 1], ...
    'nss_search_freq', 3, ...
    'eval_freq', 1, ...
    'true_data', [], ...             % Set dynamically in the loop
    'verbose', true, ...
    'svt_mode', 'adaptive', ...      % Use energy-adaptive soft thresholding
    'svt_energy', 0.9 ...
);

%% Read all images from 'data' folder (as video frames)
if ~isfolder(data_folder)
    error('Data folder not found: %s', data_folder);
end

files = dir(fullfile(data_folder, '*.png'));
if isempty(files)
    files = dir(fullfile(data_folder, '*.jpg'));
end

% Sort files by the numeric part in their filenames to ensure correct order
num_str = regexp({files.name}, '\d+', 'match', 'once');
file_nums = zeros(size(files));
for i = 1:length(files)
    if ~isempty(num_str{i})
        file_nums(i) = str2double(num_str{i});
    else
        file_nums(i) = 0;
    end
end
[~, sort_idx] = sort(file_nums);
files = files(sort_idx);
num_frames = length(files);

if num_frames == 0
    error('No images (.png / .jpg) found in the data folder.');
end

% Read the first frame to determine dimensions
img0 = imread(fullfile(data_folder, files(1).name));
if max(img0(:)) > 1
    img0 = double(img0) / 255;
else
    img0 = double(img0);
end

[H, W, C] = size(img0);
if C == 1
    % Expand grayscale to RGB
    need_expand = true;
    C = 3;
else
    need_expand = false;
end
fprintf('Video size: %d x %d, Frames: %d\n', H, W, num_frames);

% Preload all frames and construct Ground Truth Quaternion Tensor
Observed_Data = zeros(H, W, num_frames, 4, 'single');
for k = 1:num_frames
    img = imread(fullfile(data_folder, files(k).name));
    if max(img(:)) > 1
        img = double(img) / 255;
    else
        img = double(img);
    end
    if need_expand
        img = repmat(img, [1, 1, 3]);
    end
    Observed_Data(:,:,k,2) = single(img(:,:,1));
    Observed_Data(:,:,k,3) = single(img(:,:,2));
    Observed_Data(:,:,k,4) = single(img(:,:,3));
end

% Retain frame names
frame_names = cell(num_frames, 1);
for k = 1:num_frames
    [~, name, ~] = fileparts(files(k).name);
    frame_names{k} = name;
end

%% Create result directory
if ~exist(result_root, 'dir'), mkdir(result_root); end

%% Outer loop: For each missing rate
for mr_idx = 1:length(missing_rates)
    missing = missing_rates(mr_idx);
    fprintf('\n====== Missing Rate: %.1f ======\n', missing);
    
    missing_str = ['missing', strrep(num2str(missing), '.', '')];
    res_root_mr = fullfile(result_root, missing_str);
    if ~exist(res_root_mr, 'dir'), mkdir(res_root_mr); end
    
    % Generate random mask (shared across all frames)
    Mask = rand(H, W, num_frames) > missing;
    Observed_Input = Observed_Data;
    Mask_4D = repmat(Mask, [1, 1, 1, 4]);
    Observed_Input(~Mask_4D) = 0;
    
    % Set true_data for solver-internal PSNR evaluation
    solver_params.true_data = Observed_Data;
    
    % Run solver
    solver = QuaternionTensorCompleter_Vectorized(solver_params);
    t_start = tic;
    [Recovered, Info] = solver.solve(Observed_Input, Mask);
    elapsed_total = toc(t_start);
    
    % Initialize summary metrics
    summary_images = {};
    summary_psnr = [];
    summary_ssim = [];
    summary_fsim = [];
    summary_time = [];
    
    %% Frame-by-frame post-processing, evaluation, and saving
    for k = 1:num_frames
        base_name = frame_names{k};
        
        % Extract recovered RGB (convert to double for consistency)
        Rec_RGB = double(squeeze(Recovered(:,:,k,2:4)));
        Rec_RGB = max(0, min(1, Rec_RGB));
        GT_RGB  = double(squeeze(Observed_Data(:,:,k,2:4)));
        
        % PSNR
        mse = mean((GT_RGB(:) - Rec_RGB(:)).^2);
        psnr_val = 10 * log10(1 / mse);
        
        % SSIM (per-channel average)
        ssim_val = 0;
        for c = 1:3
            ssim_val = ssim_val + ssim(Rec_RGB(:,:,c), GT_RGB(:,:,c));
        end
        ssim_val = ssim_val / 3;
        
        % FSIM (Silent fallback to built-in implementation)
        if exist('Img_QA', 'file')
            [~, ~, fsim_val] = Img_QA(GT_RGB, Rec_RGB);
        else
            fsim_val = my_fsim(GT_RGB, Rec_RGB);
        end
        
        fprintf('  Frame %s: PSNR=%.2f dB, SSIM=%.4f, FSIM=%.4f\n', ...
            base_name, psnr_val, ssim_val, fsim_val);
        
        % Save recovered images
        save_dir = fullfile(res_root_mr, base_name, 'Recovered_Image');
        if ~exist(save_dir, 'dir'), mkdir(save_dir); end
        imwrite(GT_RGB, fullfile(save_dir, 'Original.jpg'));
        imwrite(Rec_RGB, fullfile(save_dir, [method_name, '.jpg']));
        
        % Save single-frame Excel
        T_single = table({method_name}, psnr_val, ssim_val, fsim_val, elapsed_total, ...
            'VariableNames', {'Method', 'PSNR', 'SSIM', 'FSIM', 'Time'});
        writetable(T_single, fullfile(res_root_mr, base_name, 'QA_Results.xlsx'));
        
        % Append to summary data
        summary_images{end+1} = base_name;
        summary_psnr(end+1) = psnr_val;
        summary_ssim(end+1) = ssim_val;
        summary_fsim(end+1) = fsim_val;
        summary_time(end+1) = elapsed_total;
    end
    
    %% Generate summary Excel
    if ~isempty(summary_images)
        % Sort summary by frame number again just in case
        num_str = regexp(summary_images, '\d+', 'match', 'once');
        sort_nums = zeros(size(summary_images));
        for i = 1:length(summary_images)
            if ~isempty(num_str{i})
                sort_nums(i) = str2double(num_str{i});
            else
                sort_nums(i) = 0;
            end
        end
        [~, order] = sort(sort_nums);
        summary_images = summary_images(order);
        summary_psnr = summary_psnr(order);
        summary_ssim = summary_ssim(order);
        summary_fsim = summary_fsim(order);
        summary_time = summary_time(order);
        
        Image = summary_images(:);
        Method = repmat({method_name}, length(summary_images), 1);
        PSNR = summary_psnr(:);
        SSIM = summary_ssim(:);
        FSIM = summary_fsim(:);
        Time = summary_time(:);
        T_all = table(Image, Method, PSNR, SSIM, FSIM, Time);
        
        % Average row
        avg_psnr = mean(PSNR);
        avg_ssim = mean(SSIM);
        avg_fsim = mean(FSIM);
        avg_time = mean(Time);
        avg_row = table({'Average'}, {method_name}, avg_psnr, avg_ssim, avg_fsim, avg_time, ...
            'VariableNames', {'Image', 'Method', 'PSNR', 'SSIM', 'FSIM', 'Time'});
        T_final = [T_all; avg_row];
        
        writetable(T_final, fullfile(res_root_mr, 'All_QA_Results.xlsx'));
        fprintf('Summary table for missing rate %.1f saved to %s\n', missing, fullfile(res_root_mr, 'All_QA_Results.xlsx'));
    end
end

fprintf('\nAll missing rates processed successfully!\n');

%% ========== Built-in FSIM Implementation (Fixed) ==========
% The following functions provide a fallback FSIM calculation if the 
% external 'Img_QA' toolbox is not available in the user's environment.

function fsim = my_fsim(ref, img)
    %MY_FSIM Stable FSIM calculation for double RGB images in [0,1].
    if max(ref(:)) <= 1.0
        ref = ref * 255;
        img = img * 255;
    end
    
    % Convert to grayscale
    if size(ref,3) == 3
        ref_gray = 0.2989 * ref(:,:,1) + 0.5870 * ref(:,:,2) + 0.1140 * ref(:,:,3);
        img_gray = 0.2989 * img(:,:,1) + 0.5870 * img(:,:,2) + 0.1140 * img(:,:,3);
    else
        ref_gray = ref;
        img_gray = img;
    end
    
    % Phase Congruency
    PC1 = phasecong2_fixed(ref_gray);
    PC2 = phasecong2_fixed(img_gray);
    
    % Gradient Magnitude (Sobel approx via gradient, 0~255 range)
    [Gx, Gy] = gradient(double(ref_gray));
    GM1 = sqrt(Gx.^2 + Gy.^2);
    [Gx, Gy] = gradient(double(img_gray));
    GM2 = sqrt(Gx.^2 + Gy.^2);
    
    % Similarity calculation
    T1 = 0.85; T2 = 160;
    S_PC = (2*PC1.*PC2 + T1) ./ (PC1.^2 + PC2.^2 + T1);
    S_GM = (2*GM1.*GM2 + T2) ./ (GM1.^2 + GM2.^2 + T2);
    S_L = S_PC .* S_GM;
    
    PC_max = max(PC1, PC2);
    sumPC = sum(PC_max(:));
    if sumPC < 1e-6
        fsim = 0;
    else
        fsim = sum(S_L(:) .* PC_max(:)) / sumPC;
    end
end

function PC = phasecong2_fixed(im)
    %PHASECONG2_FIXED Robust phase congruency based on Kovesi's method.
    %   Input: im is a 0~255 grayscale image.
    nscale = 4; minWavelength = 6; mult = 2.1; sigmaOnf = 0.55;
    norient = 6;
    [rows, cols] = size(im);
    radius = min(rows, cols) / 2;
    
    % Frequency domain coordinates
    [u, v] = meshgrid(-cols/2:cols/2-1, -rows/2:rows/2-1);
    u = u / radius; v = v / radius;
    R = sqrt(u.^2 + v.^2);
    R(rows/2+1, cols/2+1) = 1e-6;  % Avoid log(0)
    theta = atan2(v, u);
    
    fftim = fft2(im);
    sumAn = zeros(rows, cols);
    Energy = zeros(rows, cols, norient);
    
    for o = 1:norient
        angl = (o-1)*pi/norient;
        angular = exp(-(theta - angl).^2 / (2*(pi/6)^2));
        Eo = zeros(rows, cols);
        Ao = zeros(rows, cols);
        
        for s = 1:nscale
            wavelength = minWavelength * mult^(s-1);
            fo = 1.0 / wavelength;
            radial = exp(-(log(R/fo).^2) / (2*log(sigmaOnf)^2));
            radial(rows/2+1, cols/2+1) = 0;  % DC suppression
            
            gr = radial .* angular;
            gi = -1i * sign(u) .* gr;
            
            c = real(ifft2(fftim .* gr));
            s_part = real(ifft2(fftim .* gi));
            an = sqrt(c.^2 + s_part.^2);
            Eo = Eo + an;
            Ao = Ao + an;
        end
        
        Energy(:,:,o) = Eo;
        sumAn = sumAn + Ao;
    end
    
    PC = max(Energy ./ (repmat(sumAn, [1 1 norient]) + eps), [], 3);
end