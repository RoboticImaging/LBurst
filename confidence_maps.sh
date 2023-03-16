#!/bin/bash
# Visualisation of detection and description confidence maps for RoBLo features

# The script extracts features as defines with the following key variables and visualises the corresponding condifence scores. The output shows corresponding keypoints as a product of detection score (green corresponds to top-scored feature locations) and descriptor score (green corresponds to top-scored descriptors for features).
burst_size=5
crop_size=1000
noise_var=10

# We pass the common image of the robotic burst to overlay R2D2 and RoBLo burst features on the image. 
# Please refer to R2D2 repository for other R2D2 variant models. We call r2d2 model and our model to visualise confidence maps.
image_list="imgs/image_list.txt"
r2d2_model="models/R2D2.pt"
burst_model="models/RoBLo_N16_B5.pt"
save_path="outputs/"
suffix="heatmaps"

# We use GPU to accelerate feature extraction
gpu=0

for img in `cat ${image_list}`
do
  echo ${img}
  
  # R2D2 on a single gold-standard image
  
  python viz_heatmaps.py --checkpoint ${r2d2_model} --gpu ${gpu} --img ${img} --save-path ${save_path}  \
    --crop-size $crop_size --burst-size 0 --tag ${suffix}_r2d2
  
  # R2D2 on a noise-limited image
  
  python viz_heatmaps.py --checkpoint ${r2d2_model} --gpu ${gpu} --img ${img} --save-path ${save_path} \
    --crop-size $crop_size --noise-var $noise_var --burst-size 0 --tag ${suffix}_r2d2_noise
  
  # RoBLo features on noise-limited burst  
  
  python viz_heatmaps.py --checkpoint ${burst_model} --burst-size $burst_size --gpu ${gpu} --img ${img} --save-path ${save_path} \
    --out ${img} --crop-size $crop_size --tag ${suffix}_burst --noise-var $noise_var

done
