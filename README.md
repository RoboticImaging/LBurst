# RoBLo
Robotic Burst for Low Light 3D Reconstruction

We introduce **a learning architecture for a robotic burst** to jointly detect and describe blob features with well-defined scales. We demonstrate **3D reconstruction in millilux conditions** using captured **drone imagery** as burst sequences on the fly.

- [RoBLo: Robotic Burst for Low Light 3D Reconstruction]
- submitted for oral presentation at WACV 2024
- Authors: [Ahalya Ravendran](ahalyaravendran.com/), [Mitch Bryson](https://scholar.google.com.au/citations?user=yIFgUxwAAAAJ&hl=en/)\, and [Donald G Dansereau](https://roboticimaging.org/)
- website: [roboticimaging.org/RoBLo](https://roboticimaging.org/Projects/RoBLo/) with dataset details, digestible results and visualizations

<p align="center">
  <img src="https://github.com/RoboticImaging/BuFF/blob/main/assets/architecture.png" width="350" title="BuFF_architecture">
</p>

## Getting Started
## Pretrained Models
## Robotic Burst Feature Extraction
## Evaluation on the HPatches
## Evaluation on the Drone Imagery
## Training Details

## Dataset
We evaluate our feature extractor on a burst dataset collected in a light-constrained environment using the UR5e robotic arm and using multiple DJI drones.
To download the complete dataset and an example separately refer to the following links:
| Images        | Dataset |
| ------------- | ----- |
| Dataset description | [Read me](https://docs.google.com/document/d/1Ht5q7aVqLPeEca0Paon0wND1FC2mDWcwRyw0BCs2ztc/edit?usp=sharing) |
| Example burst | a burst of noisy images and corresponding ground truth with 1D and 2D apparent motion [here](https://drive.google.com/file/d/11PDClfjjMdVFbSDDxLRm28E7soqPg8FV/view?usp=sharing) (2.1GB) |
| Dataset with 1D <br> apparent motion | dataset including ground truth and noisy images [here](https://drive.google.com/file/d/19dqyBatFqHk1Yjy4QGMwWPU1Azftk9az/view?usp=sharing) (40.3GB) |
| Dataset with 2D <br> apparent motion | dataset including ground truth and noisy images [here](https://drive.google.com/file/d/1PZJmaDR7NONibRbJoyAxIZ2VrnEh9QKC/view?usp=sharing) (40.3GB) |
| Example drone burst | a burst of images captured in millilux conditions [here](https://drive.google.com/file/d/1ZoJPNvfSudslJoXgkZUX-9Vww5rGszvs/view?usp=share_link) (2.1GB) |
| Drone dataset captured using DJI Mini Pro <br> apparent motion | [here](
https://drive.google.com/file/d/1qzSutIh_3T27zfJcRmRJ7Xlg8I-As-zq/view?usp=share_link) (3.48 GB) |
| Drone dataset captured using DJI Phantom Pro <br> | [here](https://drive.google.com/file/d/1_siPLHWNl7N5ES7V6iMbQfd-teLk_leW/view?usp=share_link) (7.34 GB) |

**Preparation:** Download the dataset from above and unpack the zip folder.
Select the directory in which images are stored and perform bias correction for accurate results for datasets captured using the robotic arm

## BibTex Citation
Please consider citing our paper if you use any of the ideas presented in the paper or code from this repository:
```
@inproceedings{ravendran2023burst,
  author    = {Ahalya Ravendran and
               Mitch Bryson and
               Donald G Dansereau},
  title     = {{RoBLo: Robotic Burst for Low Light 3D Reconstruction}},
  booktitle = {arXiv},
  year      = {2023},
}
```
## Acknowledgement
We use some functions directly from [LFToolbox](https://github.com/doda42/LFToolbox) for visualisation and extend the work of [R2D2](https://github.com/naver/r2d2) for a robotic burst captured in millilux conditions.
compare reconstruction performance evaluation for state-of-the-art feature extractors similar to [the hierarchical localization toolbox](https://github.com/mihaidusmanu/Hierarchical-Localization), [Image matching benchmark](https://github.com/mihaidusmanu/image-matching-benchmark-baselines) and [Comparative Evaluation of Hand-Crafted and Learned Local Features](https://github.com/ahojnnes/local-feature-evaluation). We use [evo](https://github.com/MichaelGrupp/evo) for pose evaluation and compare against other robotic burst methods as described in [Burst with merge](https://github.com/RoboticImaging/LightConstrainedSfM) and [BuFF](https://github.com/RoboticImaging/BuFF). Please refer to individual repositories for license details.
