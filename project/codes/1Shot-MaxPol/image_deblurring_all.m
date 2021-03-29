clear all
close all
clc

directories = ["nirscene1/NIR_Images/oldbuilding_nir","nirscene1/NIR_Images/indoor_nir","nirscene1/NIR_Images/street_nir","nirscene1/NIR_Images/mountain_nir","nirscene1/NIR_Images/field_nir","nirscene1/NIR_Images/forest_nir","nirscene1/NIR_Images/urban_nir","nirscene1/NIR_Images/country_nir","nirscene1/NIR_Images/water_nir"];
addpath([pwd, filesep, 'utilities'])
do_export = true; % export restored images
scale = 1/4;    % define the down-sampling scale (default is 1/4)
model_type = 'Gaussian';   % PSF model: 'Gaussian' or 'Laplacian'


for i = 1:length(directories)
    image_directory = convertStringsToChars(directories(i));
    addpath([pwd, filesep, image_directory])

    %%
    save_dir = append(image_directory,'_restored_',model_type);
    if ~exist(save_dir, 'dir')
        mkdir([pwd, filesep, save_dir]);
    else
        which_dir = save_dir;
        dinfo = dir(which_dir);
        dinfo([dinfo.isdir]) = [];   %skip directories
        filenames = fullfile(which_dir, {dinfo.name});
        delete( filenames{:} )
    end

    %% Get all files
    dinfo = dir(image_directory);
    dinfo(ismember( {dinfo.name}, {'.', '..'})) = [];
    shape = size(dinfo);
    n_files = shape(1);


    for i = 1:n_files
        image_name = dinfo(i).name;
        image_scan_original = imread(image_name);
        image_scan_original = im2double(image_scan_original);
        [N_1, N_2, N_3] = size(image_scan_original);
        %%
        [h_psf, c1_estimate, c2_estimate, alpha_estimate, amplitude_estimate] = ...
            blur_kernel_estimation(image_scan_original, model_type, scale);

        %%
        [deblurring_kernel] = deblurring_kernel_estimation(h_psf, model_type);

        %%
        significany = 0.5; % edge significany control (optional) default is 0.5
        tic;
        [deblurred_image] = OneShotMaxPol(image_scan_original, deblurring_kernel, ...
            model_type, alpha_estimate, c1_estimate, h_psf, significany);
        elapsed_time=toc;

    %     %%
    %     figure('rend','painters','pos', [50 , 300, 1500, 600]);
    %     subplot(1,2,1)
    %     imshow(image_scan_original, 'border', 'tight')
    %     title('Natural Blurred Image')
    %     subplot(1,2,2)
    %     imshow(deblurred_image, 'border', 'tight')
    %     title('1Shot-MaxPol Deblurring')
    %     %%

        if do_export
            imwrite(deblurred_image, [pwd, filesep, save_dir, filesep, ...
            image_name], 'TIFF', 'compression', 'none')
        end
    end

    to_print = ['Done with %s', image_directory];
    disp(to_print);

end
%%

model_type = 'Laplacian';   % PSF model: 'Gaussian' or 'Laplacian'


for i = 1:length(directories)
    image_directory = convertStringsToChars(directories(i));
    addpath([pwd, filesep, image_directory])

    %%
    save_dir = append(image_directory,'_restored_',model_type);
    if ~exist(save_dir, 'dir')
        mkdir([pwd, filesep, save_dir]);
    else
        which_dir = save_dir;
        dinfo = dir(which_dir);
        dinfo([dinfo.isdir]) = [];   %skip directories
        filenames = fullfile(which_dir, {dinfo.name});
        delete( filenames{:} )
    end

    %% Get all files
    dinfo = dir(image_directory);
    dinfo(ismember( {dinfo.name}, {'.', '..'})) = [];
    shape = size(dinfo);
    n_files = shape(1);


    for i = 1:n_files
        image_name = dinfo(i).name;
        image_scan_original = imread(image_name);
        image_scan_original = im2double(image_scan_original);
        [N_1, N_2, N_3] = size(image_scan_original);
        %%
        [h_psf, c1_estimate, c2_estimate, alpha_estimate, amplitude_estimate] = ...
            blur_kernel_estimation(image_scan_original, model_type, scale);

        %%
        [deblurring_kernel] = deblurring_kernel_estimation(h_psf, model_type);

        %%
        significany = 0.5; % edge significany control (optional) default is 0.5
        tic;
        [deblurred_image] = OneShotMaxPol(image_scan_original, deblurring_kernel, ...
            model_type, alpha_estimate, c1_estimate, h_psf, significany);
        elapsed_time=toc;

    %     %%
    %     figure('rend','painters','pos', [50 , 300, 1500, 600]);
    %     subplot(1,2,1)
    %     imshow(image_scan_original, 'border', 'tight')
    %     title('Natural Blurred Image')
    %     subplot(1,2,2)
    %     imshow(deblurred_image, 'border', 'tight')
    %     title('1Shot-MaxPol Deblurring')
    %     %%

        if do_export
            imwrite(deblurred_image, [pwd, filesep, save_dir, filesep, ...
            image_name], 'TIFF', 'compression', 'none')
        end
    end

    to_print = ['Done with %s', image_directory];
    disp(to_print);

end
%%
