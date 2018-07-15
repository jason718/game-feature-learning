% Demos the use of project_depth_map.m

% The location of the RAW dataset.
CLIPS_DIR = '/nfs/nas-3-39/aayushb/threeD/dataset/nyu_depth_v2_raw/';

% The path to the labeled dataset.
LABELED_DATASET_PATH = '/nfs/nas-3-39/aayushb/threeD/dataset/nyu_depth_v2_labeled.mat';

load(LABELED_DATASET_PATH, 'rawDepthFilenames', 'rawRgbFilenames');

%% Load a pair of frames and align them.
imgRgb = imread(sprintf('%s/%s', CLIPS_DIR, rawRgbFilenames{1}));

imgDepth = imread(sprintf('%s/%s', CLIPS_DIR, rawDepthFilenames{1}));
imgDepth = swapbytes(imgDepth);

[imgDepth2, imgRgb2] = project_depth_map(imgDepth, imgRgb);

%% Now visualize the pair before and after alignment.
imgDepthAbsBefore = depth_rel2depth_abs(double(imgDepth));
imgOverlayBefore = get_rgb_depth_overlay(imgRgb, imgDepthAbsBefore);

imgOverlayAfter = get_rgb_depth_overlay(imgRgb2, imgDepth2);

keyboard;

figure;
subplot(1,2,1);
imagesc(crop_image(imgOverlayBefore));
title('Before projection');

subplot(1,2,2);
imagesc(crop_image(imgOverlayAfter));
title('After projection');
