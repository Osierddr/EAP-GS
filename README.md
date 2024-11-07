# EAP-GS: Efficient Augmentation of Pointcloud for 3D Gaussian Splatting in Few-shot Scene Reconstruction

![EAP-GS](assets/pipeline.png)

This is the official repository for the paper "EAP-GS: Efficient Augmentation of Pointcloud for 3D Gaussian Splatting in Few-shot Scene Reconstruction"
Our code is built upon the foundation of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [DepthRegGS](https://github.com/robot0321/DepthRegularizedGS), and we would like to thank the team for their valuable contributions.

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
In this work, we select the train/test set in LLFF and Mip-NeRF360 datasets. Of course, you can also use your own captured datasets, all you need to provide is the images. It should be emphasized that the images should be forward-facing.

### Step 1
Place the dataset under `<datadir>`, and prepare as below:
```
<datadir>
|---images
|   |---00000.jpg
|   |---00001.jpg
|   |---...
```
Run `split.py` to generate the `split_index.json` file for dividing the train/test set:
```shell
python split.py --workspace_path ./<datadir> --image_path ./<datadir>/images
```

### Step 2
Run COLMAP, here we use `automatic_reconstructor` to obtain pointcloud and camera poses:
```shell
colmap automatic_reconstructor --workspace_path ./<datadir> --image_path ./<datadir>/images --camera_model SIMPLE_PINHOLE --single_camera 1 --dense 0 --num_threads -1 --quality extreme
```
And now, your file path under `<datadir>` should be:
```
<datadir>
|---images
|   |---00000.jpg
|   |---00001.jpg
|   |---...
|---sparse
|   |---0
|       |---cameras.bin
|       |---images.bin
|       |---points3D.bin
|       |---incremental_model_refiner_output.txt
|---database.db
|---split_index.json
```

### Step 3
Run `augmentation.py` to get attention regions, which are placed under `<datadir>/images` with the same naming convention as `images`:
```shell
python augmentation.py --source_path ./<datadir>
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for augmentation.py</span></summary>

  #### --source_path
  Path to the source directory containing images.
  #### --eps
  The maximum distance between two samples for one to be considered as in the neighborhood of the other. (```15``` by default).
  #### --min_samples
  The number of samples in a neighborhood for a point to be considered as a core point. (```10``` by default).
  #### --radius
  The radius of region around attention point. (```10``` by default).

</details>

### Step 4
Combine images and attention regions, using `automatic_reconstructor` to generate a fine pointcloud:
```shell
colmap automatic_reconstructor --workspace_path ./<datadir> --image_path ./<datadir>/images --camera_model SIMPLE_PINHOLE --single_camera 1 --dense 0 --num_threads -1 --quality extreme
```
(optional) using `DetectorfreeSfM` can yields a more numerous and balanced pointcloud, which is adopted in our work.

Replace the previously generated coarse pointcloud with a fine pointcloud for subsequent reconstruction.

## Running
For training, we optimize for 5000 iterations, simply use:
```shell
python train.py -s <datadir> --model_path <datadir>/output --resolution 1 --seed 3 --iteration 5000    # Train without test set
python train.py -s <datadir> --model_path <datadir>/output --resolution 1 --seed 3 --iteration 5000 --eval    # Train with train/test set
```

Additionally, we could introduce depth regularization by adding `--depth` and `--usedepthReg`, which are the same as [DepthRegGS](https://github.com/robot0321/DepthRegularizedGS):
```shell
python train.py -s <datadir> --model_path <datadir>/output --resolution 1 --seed 3 --iteration 5000 --eval --depth --usedepthReg
```
