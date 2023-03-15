# RoBLo
Official implementation of Robotic Burst for Low Light 3D Reconstruction

We introduce **a learning architecture for a robotic burst** to jointly detect and describe blob features with well-defined scales. We demonstrate **3D reconstruction in millilux conditions** using captured **drone imagery** as burst sequences on the fly.

- RoBLo: Robotic Burst for Low Light 3D Reconstruction
- submitted for oral presentation at WACV 2024
- Authors: [Ahalya Ravendran](ahalyaravendran.com/), [Mitch Bryson](https://scholar.google.com.au/citations?user=yIFgUxwAAAAJ&hl=en/)\, and [Donald G Dansereau](https://roboticimaging.org/)
- website: [roboticimaging.org/RoBLo](https://roboticimaging.org/Projects/RoBLo/) with dataset details, digestible contents and visualizations
![image](https://github.com/RoboticImaging/RoBLo/blob/main/assets/architecture.png)

## Getting Started
Use ```conda``` to create the environment equipped with Python 3.6+ with standard scientific packages and PyTorch.

```bash
conda create -n roblo python=3.9
conda activate roblo
conda install tqdm pillow numpy matplotlib scipy
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

#### Clone the Git repository.  
```bash
git clone https://github.com/RoboticImaging/RoBLo.git
cd RoBLo
```
## Pretrained Models
We provide eight pre-trained models in the `models/` folder. Five of the pretrained models are trained with different burst sizes and the rest have different patch sizes for a burst of 5 images during training as follows.
| model name | description |
|:------------------:|:------------------:|
|`RoBLo_N4_B5.pt` | Trained model with a patch size of 4 and burst size of 5 |
|`RoBLo_N8_B5.pt` | Trained model with a patch size of 8 and burst size of 5 |
|`RoBLo_N16_B5.pt` | Trained model with a patch size of 16 and burst size of 5 |
|`RoBLo_N32_B5.pt` | Trained model with a patch size of 32 and burst size of 5 |
|`RoBLo_N64_B5.pt` | Trained model with a patch size of 64 and burst size of 5 |
|`RoBLo_N16_B3.pt` | Trained model with a patch size of 16 and burst size of 3 |
|`RoBLo_N16_B7.pt` | Trained model with a patch size of 16 and burst size of 7 |
|`RoBLo_N16_B9.pt` | Trained model with a patch size of 16 and burst size of 9 |

## Robotic Burst Feature Extraction
We provide a shell script to extract burst features for a given robotic burst.
```bash
./extract_burst.sh
```

The script saves the `top-k` keypoints as a feature file in numpy format with `roblo` as the file suffix, and saves the file in the same path as the images in the burst.

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

We synthetically generate noisy bursts for each image in the HPatches dataset to create our `HPatches bursts`. To evaluate our methods, we compare using all images within a noisy burst to using `r2d2` on a common image of a noisy burst, with the original HPatches images serving as gold standard images. We extract features from the gold standard images, noisy images, and noisy bursts using the following command,
```bash
./extract_hpatches.sh
```
We evaluate the matching performance using iPython notebook, `d2-net/hpatches_sequences/HPatches-Sequences-Matching-Benchmark.ipynb`.

The following demonstrates the average matching performance of `RoBLo` against `r2d2` in strong noise.
<p align="center">
  <img src="https://github.com/RoboticImaging/RoBLo/blob/main/assets/matching_performance.png" width="1000" title="BuFF_architecture">
</p>

## Evaluation on the Drone Burst Imagery
We evaluate reconstruction performance and camera pose estimation for noise-limited burst datasets captured with 1D and 2D apparent motion and reconstruction performance using drone burst imagery captured in millilux conditions. For more details on captured burst, refer to Dataset section below.

## Training Details

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
  title     = {{RoBLo: Robotic Burst for Low Light 3D Reconstruction}},
  booktitle = {arXiv},
  year      = {2023},
}
```
## Acknowledgement
We use some functions directly from [LFToolbox](https://github.com/doda42/LFToolbox) for visualisation and extend the work of [R2D2](https://github.com/naver/r2d2) for a robotic burst captured in millilux conditions. We compare reconstruction performance evaluation for state-of-the-art feature extractors similar to [the hierarchical localization toolbox](https://github.com/mihaidusmanu/Hierarchical-Localization), [Image matching benchmark](https://github.com/mihaidusmanu/image-matching-benchmark-baselines) and [evaluation benchmark](https://github.com/ahojnnes/local-feature-evaluation). We use [evo](https://github.com/MichaelGrupp/evo) for pose evaluation and compare against other robotic burst methods as described in [Burst with merge](https://github.com/RoboticImaging/LightConstrainedSfM) and [BuFF](https://github.com/RoboticImaging/BuFF). Please refer to individual repositories for more details on license.
