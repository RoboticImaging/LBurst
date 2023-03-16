% Visualising the RoBLo feature matches
% To read computed RoBLo features, we use read NPY function (https://github.com/kwikteam/npy-matlab) in MATLAB
% For descriptor matching, we use vl_ubcmatch (https://www.vlfeat.org/matlab/vl_ubcmatch.html)

img1 = im2double(imread('/image/1.png'));
image1 = imnoise(img1, 'gaussian', 0, 0.03);

img2 = im2double(imread('/image/2.png'));
image2 = imnoise(img2, 'gaussian', 0, 0.03);

kp1  = readNPY('/image/1.png/keypoints.npy');
kp2 = readNPY('/image/2.png/keypoints.npy');

d1  = readNPY('/image/2.png/descriptors.npy');
d2  = readNPY('/image/2.png/descriptors.npy');

[matches, scores] = vl_ubcmatch(d1, d2);

%---Display---
fprintf('Displaying matches...\n');
figure(1);
clf
Thumb = cat(2,image1,image2);
imshow(Thumb);

NumMatches = size(matches,2);
which=ceil(linspace(1,NumMatches)); % show 20 matches
x1 = kp1(1,matches(1,which));
x2 = kp2(1,matches(2,which)) + size(image1,2);
y1 = kp1(2,matches(1,which));
y2 = kp2(2,matches(2,which));

hold on;
plot([x1; x2], [y1; y2], 'color', 'yellow' ,'LineWidth', 2);
title(sprintf('Aachen Training epochs 25 -- %d matches', size(matches,2)))
fprintf('Done.\n');
