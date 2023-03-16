#!/bin/bash
# Extraction of features - Salient keypoint locations, well-defined scales and descriptors

# This script extract R2D2 features on single images and noise-limited single images.
# It also extracts RoBLo features in a robotic burst of images corresponding to the common middle image.
burst_size=5
crop_size=1000
noise_var=10

# We generate a robotic burst of images to extract RoBLo features and extract R2D2 features on the common input image
# Note, if images are not '.png' format modify image format directly in extract.py
# Please refer to R2D2 repository for other R2D2 variant models.

run_tag=test
r2d2_model="models/R2D2.pt"
burst_model="models/RoBLo_N16_B5.pt"
image_list="imgs/image_list.txt"

# We use GPU to accelerate feature extraction
gpu=0

# Uncomment the following for a shorter list of test hpatches images
#image_list="imgs/image_list_hpatches_short.txt"

# R2D2 on gold standard image
python extract.py --model $r2d2_model --images $image_list --gpu ${gpu} \
--burst-size 1 --tag r2d2_$run_tag --crop-size $crop_size --noise-var 0

# R2D2 on noise-limited image with increasing amount of noise
#for n in 10
#do
#  python extract.py --model $r2d2_model --images $image_list --gpu ${gpu} \
#  --burst-size 1 --tag r2d2_noise_${n}_${run_tag} --crop-size $crop_size --noise-var ${n}
#done

# RoBLo on noise-limited burst
python extract.py --model $burst_model --images $image_list --gpu ${gpu} \
--burst-size $burst_size --tag burst_$run_tag --crop-size $crop_size --noise-var $noise_var
