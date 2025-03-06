# EAP-GS

This is the official repository for the paper "EAP-GS: Efficient Augmentation of Pointcloud for 3D Gaussian Splatting in Few-shot Scene Reconstruction"
🚀**CVPR 2025 Nashville**🚀

[Project](https://osierddr.github.io/eapgs/)

![EAP-GS](assets/pipeline.png)


## Requirments
Before getting started, we outline the requirements for successfully running this code. The optimizer uses PyTorch and CUDA extensions in a Python environment.
- All of the requirements specified by [3DGS](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#optimizer).
- [COLMAP](https://github.com/colmap/colmap), which is a classical Structure-from-Motion (SfM) method.
- (optional) [DetectorfreeSfM](https://github.com/zju3dv/DetectorFreeSfM), which is another SfM method used in our work, leverages a detector-free matcher to enhances feature extraction in the texture-poor scenarios.

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

## Dataset Preparation and Initialization
In this work, we select the train/test set in LLFF and Mip-NeRF360 datasets. Of course, you can also use your own captured datasets. Following the conventions of sparse-view settings, Camera poses are assumed to be known based on full-view estimation or other methods, located in the directory corresponding to the dataset. Our experiments use community standards train-test split, i.e., select every eighth image as the testing view, and evenly sample sparse views from the remaining images for training.

### Step 1
Download and place the dataset in `<datadir>`, copy the corresponding scenario's `.json` file from the `/split` directory into `<datadir>`, and prepare as follows:
```
<datadir>
|---<scene1>
|   |---images
|   |   |---...
|   |---sparse
|   |   |---0
|       |   |---...
|   |---poses_bounds.npy
|   |---split_index.json
|---<scene2>
|...
```

### Step 2
Run `run_colmap.py`, here we use the parameters set by FSGS to execute feature extraction, feature matching, and triangulation under known camera poses:
```shell
python run_colmap.py --base_path <datadir>/ --views 3  # 3 for LLFF, 12 for MipNeRF360
```
The corresponding few-shot initialization results are in `3_views`, and `<points3D_3views.bin>` is also the few-shot 3D pointcloud,  your file path in `<datadir>` should be:
```
<datadir>
|---<scene1>
|   |---3_views
|   |   |---created
|   |   |---images
|   |   |---sparse
|   |   |---database.db
|   |---images
|   |   |---...
|   |---sparse
|   |   |---0
|       |   |---points3D_3views.bin
|       |   |---...
|   |---poses_bounds.npy
|   |---split_index.json
|---<scene2>
|...
```

### Step 3
Run `augmentation.py` to get attention regions, which are placed under `<datadir>/images` with the same naming convention as `images`:
```shell
python augmentation.py --source_path <datadir>/<scene> --pc_name points3D_3views
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for augmentation.py</span></summary>

  #### --source_path
  Path to the source directory containing images.
  #### --pc_name
  Name of the pointcloud generated by SfM.. (```points3D``` by default).
  #### --eps
  The maximum distance between two samples for one to be considered as in the neighborhood of the other. (```15``` by default).
  #### --min_samples
  The number of samples in a neighborhood for a point to be considered as a core point. (```10``` by default).
  #### --radius
  The radius of region around attention point. (```10``` by default).

</details>

### Step 4
Combine images and attention regions with same poses, running `run_colmap.py` again to generate a fine pointcloud, which is alse located in `<datadir>/<scene>/sparse/0`:
```shell
python run_colmap.py --base_path <datadir>/ --augment --views 3  # 3 for LLFF, 12 for MipNeRF360
```
(optional) using `DetectorfreeSfM` can yields a more numerous and balanced pointcloud, which is adopted in our work.

Replace the previously generated coarse pointcloud with a fine pointcloud for subsequent reconstruction.

## Running
For training, we optimize for 5000 iterations, simply use:
```shell
python train.py -s <datadir>/<scene> --model_path <datadir>/<scene>/output -r 4 --seed 3 --pc_name points3D_3views_aug   # Train without test set
python train.py -s <datadir>/<scene> --model_path <datadir>/<scene>/output -r 4 --seed 3 --pc_name points3D_3views_aug --eval    # Train with train/test set
```

Additionally, we could introduce depth regularization by adding `--depth` and `--usedepthReg`, which are the same as [DRGS](https://github.com/robot0321/DepthRegularizedGS):
```shell
python train.py -s <datadir>/<scene> --model_path <datadir>/<scene>/output -r 4 --seed 3 --pc_name points3D_3views_aug --eval --depth --usedepthReg
```

## Acknowledgement
Our code is built upon the foundation of following works, and we would like to thank the team for their valuable contributions!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [DRGS](https://github.com/robot0321/DepthRegularizedGS)

## Citation

Consider citing as below if you find this repository helpful to your project:

```
coming soon!
```
