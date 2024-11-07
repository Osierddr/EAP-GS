# EAP-GS: Efficient Augmentation of Pointcloud for 3D Gaussian Splatting in Few-shot Scene Reconstruction

This is the official repository for the paper "EAP-GS: Efficient Augmentation of Pointcloud for 3D Gaussian Splatting in Few-shot Scene Reconstruction"
Our code is built upon the foundation of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [DepthRegGS](https://github.com/robot0321/DepthRegularizedGS), and we would like to thank the team for their valuable contributions.

## Installation and Setup
The repository contains submodules, clone this repository by:
```shell
git clone https://github.com/Osierddr/EAP-GS.git --recursive
```

Our default installation method is based on Conda package and environment management, allowing you to install the dependencies by:
```shell
conda env create --file environment.yml
conda activate EAP-GS
```