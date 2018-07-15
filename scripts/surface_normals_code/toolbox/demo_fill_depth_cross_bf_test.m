% Demo's the in-painting function fill_depth_cross_bf.m

DATASET_PATH = '/nfs/nas-3-39/aayushb/threeD/dataset/nyu_depth_v2_labeled.mat';

load(DATASET_PATH, 'images', 'rawDepths');

%%
imageInd = 1;

imgRgb = images(:,:,:,imageInd);
imgDepthAbs = rawDepths(:,:,imageInd);

% Crop the images to include the areas where we have depth information.
imgRgb = crop_image(imgRgb);
imgDepthAbs = crop_image(imgDepthAbs);

imgDepthFilled = fill_depth_cross_bf(imgRgb, double(imgDepthAbs));

keyboard;

figure(1);
subplot(1,3,1); imagesc(imgRgb);
subplot(1,3,2); imagesc(imgDepthAbs);
subplot(1,3,3); imagesc(imgDepthFilled);
