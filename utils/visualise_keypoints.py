# Visualize RoBLo feature locations and well-defined scales

import numpy as np
import matplotlib.pyplot as plt

descriptors = np.load('descriptors.npy')
keypoints = np.load('keypoints.npy')

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
img = plt.imread("/hpatches_sequences/hpatches-sequences-release/i-miniature/1.ppm")
fig, ax = plt.subplots()
ax.set_axis_off()
im = ax.imshow(img, cmap='gray')
plt.scatter(keypoints[:,0],keypoints[:,1],keypoints[:,2], facecolors='none', edgecolors='y')
plt.savefig('/hpatches_sequences/hpatches-sequences-release/i-miniature/img1_keypoints.png')
plt.show()
