# LBurst
Official implementation of LBurst: Learning-based Robotic Burst Feature Extraction for 3D Reconstruction in Low-Light

We introduce **a learning architecture for a robotic burst** to jointly detect and describe blob features with well-defined scales. We demonstrate **3D reconstruction in millilux conditions** using captured **drone imagery** as burst sequences on the fly.

- LBurst: Robotic Burst for Low Light 3D Reconstruction
- submitted for oral presentation at WACV 2024
- Authors: [Ahalya Ravendran](ahalyaravendran.com/), [Mitch Bryson](https://scholar.google.com.au/citations?user=yIFgUxwAAAAJ&hl=en/)\, and [Donald G Dansereau](https://roboticimaging.org/)
- website: [roboticimaging.org/LBurst](https://roboticimaging.org/Projects/LBurst/) with dataset details, digestible contents and visualizations

![image](https://github.com/RoboticImaging/LBurst/blob/main/assets/architecture.png)

## Getting Started
Use ```conda``` to create the environment equipped with Python 3.6+ with standard scientific packages and PyTorch.

```bash
conda create -n LBurst python=3.9
conda activate LBurst
conda install tqdm pillow numpy matplotlib scipy
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

#### Clone the Git repository.  
```bash
git clone https://github.com/RoboticImaging/LBurst.git
cd LBurst
```

## Overview
This repository contains the following sub-modules<br>
- [assets](https://github.com/RoboticImaging/LBurst/blob/main/assets/) - Digestible visual content for repository.<br>
- [datasets](https://github.com/RoboticImaging/LBurst/blob/main/datasets/) - Dataset preparation for training.<br>
- [imgs](https://github.com/RoboticImaging/LBurst/blob/main/imgs/) - Images and image list for testing.<br>
- [models](https://github.com/RoboticImaging/LBurst/blob/main/models/) - Pretrained models with varying patch size and burst size.<br>
- [nets](https://github.com/RoboticImaging/LBurst/blob/main/nets/) - Netwrok architecture and loss functions.<br>
- [tools](https://github.com/RoboticImaging/LBurst/blob/main/tools/) - Tools associated with training including burst dataloader and trainer.<br>
- [utils](https://github.com/RoboticImaging/LBurst/blob/main/utils/) - Burst prepearation for visualisation.<br>

## Pretrained Models
We provide eight pre-trained models in the `models/` folder. Five of the pretrained models are trained with different burst sizes and the rest have different patch sizes for a burst of 5 images during training as follows.
| model name | description |
|:------------------:|:------------------:|
|`LBurst_N4_B5.pt` | Trained model with a patch size of 4 and burst size of 5 |
|`LBurst_N8_B5.pt` | Trained model with a patch size of 8 and burst size of 5 |
|`LBurst_N16_B5.pt` | Trained model with a patch size of 16 and burst size of 5 |
|`LBurst_N32_B5.pt` | Trained model with a patch size of 32 and burst size of 5 |
|`LBurst_N64_B5.pt` | Trained model with a patch size of 64 and burst size of 5 |
|`LBurst_N16_B3.pt` | Trained model with a patch size of 16 and burst size of 3 |
|`LBurst_N16_B7.pt` | Trained model with a patch size of 16 and burst size of 7 |
|`LBurst_N16_B9.pt` | Trained model with a patch size of 16 and burst size of 9 |

## Robotic Burst Feature Extraction
We provide a shell script to extract burst features for a given robotic burst.
```bash
./extract_burst.sh
```

The script saves the `top-k` keypoints as a feature file in numpy format with `LBurst` as the file suffix, and saves the file in the same path as the images in the burst.

The feature file includes the feature locations and well-defined scale of the common image of the burst as an array of size `N x 3` in `keypoints`; L2-normalized descriptors, as `N x 128` in `descriptors`; and corresponding confidence scores for each keypoint as `scores`.

The script allows for flexibility in modifying various feature parameters during the burst feature extraction process, which is explained in detail within the `extract_burst.sh` script. By default, the scale factor is set to `2^0.25`, similar to state-of-the-art scale invariant feature extractors.

For visualisation of features with corresponding detection and descriptor confidence,
```bash
./confidence_maps.sh
```

## Evaluation on the HPatches Bursts
The evaluation of the HPatches dataset is based on the [code](https://github.com/mihaidusmanu/d2-net) from [D2-Net](https://dsmn.ml/publications/d2-net.html) as,
```bash
git clone https://github.com/mihaidusmanu/d2-net.git
cd d2-net/hpatches_sequences/
bash download.sh
bash download_cache.sh
cd ../..
ln -s d2-net/hpatches_sequences #soft-link for the HPatches dataset
```

We synthetically generate noisy bursts for each image in the HPatches dataset to create our `HPatches bursts`. To evaluate our methods, we compare using all images within a noisy burst to using `r2d2` on a common image of a noisy burst, with the original HPatches images serving as gold standard images. We extract features from the gold standard images, noisy images, and noisy bursts. For more details, check `python extract.py --help` 

We evaluate the matching performance using iPython notebook, `d2-net/hpatches_sequences/HPatches-Sequences-Matching-Benchmark.ipynb`.

The following demonstrates the average matching performance of `LBurst` against `r2d2` in strong noise.
<p align="center">
  <img src="https://github.com/RoboticImaging/LBurst/blob/main/assets/matching_performance.png" width="1000" title="BuFF_architecture">
</p>

## Evaluation on the Drone Burst Imagery
We evaluate reconstruction performance and camera pose estimation for noise-limited burst datasets captured with 1D and 2D apparent motion and reconstruction performance using drone burst imagery captured in millilux conditions. For more details on captured burst, refer to Dataset section below.

## Training Details
Create a folder in a location where you have sufficient disk space (8 GB required) to host all the data as,
```bash
DATA_ROOT=/path/to/data
mkdir -p $DATA_ROOT
ln -fs $DATA_ROOT data 
mkdir $DATA_ROOT/aachen
```

Download the Aachen dataset manually from [here](https://drive.google.com/drive/folders/1fvb5gwqHCV4cr4QPVIEMTWkIhCpwei7n), and save it as `$DATA_ROOT/aachen/database_and_query_images.zip`. Complete the installation and download the remaining training data as,
```bash 
./download_training_data.sh
```

The training datasets are as follows,
| model name | disk space | number of images | instances |
|:------------------:|:------------------:|:------------------:|:------------------:|
| Aachen DB images | 2.7 GB | 4479 | `auto_pairs(aachen_db_images)` |
| Random Web images | 1.5 GB | 3190 |  `auto_pairs(web_images)` |

We introduce burst functions to create a robotic burst for each image in the dataset. To visualize the content of a robotic burst,
```bash
python -m tools.burst_dataloader "PairLoader(aachen_flow_pairs)"
```

For training,
```bash
python train_burst.py --save-path /path/to/LBurst_model.pt 
```

Training bursts of 5 images with a patch size of 16 on the NVIDIA GeForce RTX 3080 takes approximately 4 minutes per epoch and 37 minutes to complete 25 epochs. This is because we leverage the  `faster r2d2 ` backbone architecture for our approach. Note, the training time will increase as the patch size decreases and the number of images within the robotic bursts increases. For more information on all parameters that can be configured during the training, refer to `python train.py --help`.

## Dataset
We evaluate our feature extractor on a burst dataset collected in a light-constrained environment using the UR5e robotic arm and using multiple DJI drones.
To download the complete dataset and an example separately refer to the following links:
| Images        | Dataset |
| ------------- | ----- |
| Dataset description | [Read me](https://docs.google.com/document/d/1Ht5q7aVqLPeEca0Paon0wND1FC2mDWcwRyw0BCs2ztc/edit?usp=sharing) |
| Example burst | [robotic  burst](https://drive.google.com/file/d/11PDClfjjMdVFbSDDxLRm28E7soqPg8FV/view?usp=sharing) of images captured using a robotic arm (2.1GB) |
| Dataset with 1D apparent motion | [dataset](https://drive.google.com/file/d/19dqyBatFqHk1Yjy4QGMwWPU1Azftk9az/view?usp=sharing) including ground truth and noisy images (40.3GB) |
| Dataset with 2D apparent motion | [dataset](https://drive.google.com/file/d/1PZJmaDR7NONibRbJoyAxIZ2VrnEh9QKC/view?usp=sharing) including ground truth and noisy images (40.3GB) |
| Drone dataset description | [Read me](https://docs.google.com/document/d/1FguBX3V8Xab8H6nB9H50-dzhQJd-SuclAj2KfzspXoE/edit?usp=sharing) |
| Example drone burst | [robotic burst](https://drive.google.com/file/d/1ZoJPNvfSudslJoXgkZUX-9Vww5rGszvs/view?usp=share_link) of images captured using a drone in millilux conditions (2.1GB) |
| Drone dataset captured using DJI Mini Pro | [dataset](https://drive.google.com/file/d/1qzSutIh_3T27zfJcRmRJ7Xlg8I-As-zq/view?usp=share_link) of 5 scenes captured in millilux conditions (3.48 GB) |
| Drone dataset captured using DJI Phantom Pro | [dataset](https://drive.google.com/file/d/1_siPLHWNl7N5ES7V6iMbQfd-teLk_leW/view?usp=share_link)  of 5 scenes captured in millilux conditions (7.34 GB) |

**Preparation:** Download the dataset from above and unpack the zip folder.
For datasets captured using a robotic arm, select the directory in which images are stored and perform bias correction for accurate results.

## BibTex Citation
Please consider citing our paper if you use any of the ideas presented in the paper or code from this repository:
```
@inproceedings{ravendran2023burst,
  author    = {Ahalya Ravendran, Mitch Bryson and Donald G Dansereau},
  title     = {{LBurst: Robotic Burst for Low Light 3D Reconstruction}},
  booktitle = {arXiv},
  year      = {2023},
}
```
## Acknowledgement
We use some functions directly from [LFToolbox](https://github.com/doda42/LFToolbox) for visualisation and extend the work of [R2D2](https://github.com/naver/r2d2) for a robotic burst captured in millilux conditions. We compare reconstruction performance evaluation for state-of-the-art feature extractors similar to [the hierarchical localization toolbox](https://github.com/mihaidusmanu/Hierarchical-Localization), [Image matching benchmark](https://github.com/mihaidusmanu/image-matching-benchmark-baselines) and [evaluation benchmark](https://github.com/ahojnnes/local-feature-evaluation). We use [evo](https://github.com/MichaelGrupp/evo) for pose evaluation and compare against other robotic burst methods as described in [Burst with merge](https://github.com/RoboticImaging/LightConstrainedSfM) and [BuFF](https://github.com/RoboticImaging/BuFF). Please refer to individual repositories for more details on license.
