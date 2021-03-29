clear all
close all
clc

%%
addpath([pwd, filesep, 'utilities'])
addpath([pwd, filesep, 'raw_images'])
addpath([pwd, filesep, './'])

%%
do_export = true; % export restored images
scale = 1/4;    % define the down-sampling scale (default is 1/4)
model_type = 'Gaussian';   % PSF model: 'Gaussian' or 'Laplacian'

%%
image_name = 'random_64.png';
image_scan_original = imread(image_name);
image_scan_original = im2double(image_scan_original);
[N_1, N_2, N_3] = size(image_scan_original);
n_iters=10;
times = [];
for k = 1:n_iters+1
    %%
    tic;
    [h_psf, c1_estimate, c2_estimate, alpha_estimate, amplitude_estimate] = ...
        blur_kernel_estimation(image_scan_original, model_type, scale);

    %%
    [deblurring_kernel] = deblurring_kernel_estimation(h_psf, model_type);

    %%
    significany = 0.5; % edge significany control (optional) default is 0.5
    
    [deblurred_image] = OneShotMaxPol(image_scan_original, deblurring_kernel, ...
        model_type, alpha_estimate, c1_estimate, h_psf, significany);
    elapsed_time=toc;
    times = [times; elapsed_time];
end
disp(median(times));