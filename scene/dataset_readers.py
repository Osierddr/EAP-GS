#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
from glob import glob
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import imageio
from datetime import datetime
from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array = None
    depthmap: np.array = None
    depth_weight: np.array = None
    mask: np.array = None
    depthloss: float=1e5


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, mask_folder, pcd=None, train_idx=None, test_idx=None, white_background=False, Depthoptim=True):
    cam_infos = []
    model_zoe = None

    for idx, key in enumerate(sorted(cam_extrinsics,key=lambda x:cam_extrinsics[x].name)):

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        K = np.array([[focal_length_x, 0, width/2],[0,focal_length_y,height/2],[0,0,1]])

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        if white_background:
            ############################## borrow from blender ##################################
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1,0]) if white_background else np.array([0, 0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:4] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
            #######################################################################################
        depth, depthmap, source_depth = None, None, None
        depth_weight, mask = None, None
        depthloss = 1e8
        # source_depth: monocluar depth estimation
        # depthmap: sparse point cloud
        # depth: final depth

        if Depthoptim and pcd is not None and idx in train_idx:
            depthmap, depth_weight = np.zeros((height,width)), np.zeros((height,width))
            cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1)) ### for coordinate definition, see getWorld2View2() function
            valid_idx = np.where(np.logical_and.reduce((cam_coord[2]>0, cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=width-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=height-1)))[0]
            pts_depths = cam_coord[-1:, valid_idx]
            cam_coord = cam_coord[:2, valid_idx]/cam_coord[-1:, valid_idx]
            depthmap[np.round(cam_coord[1]).astype(np.int32).clip(0,height-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width-1)] = pts_depths
            depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0,height-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width-1)] = 1/pcd.errors[valid_idx] if pcd.errors is not None else 1
            depth_weight = depth_weight/depth_weight.max()

            if model_zoe is None:
                model_zoe = torch.hub.load("./ZoeDepth", "ZoeD_NK", source="local", pretrained=True).to('cuda')
            source_depth = model_zoe.infer_pil(image.convert("RGB"))  

            depth, depthloss = optimize_depth(source=source_depth, target=depthmap, mask=depthmap>0.0, depth_weight=depth_weight)

        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name,
                              width=width, height=height, depth=depth, depthmap=depthmap, depth_weight=depth_weight, mask=mask, depthloss=depthloss)
        cam_infos.append(cam_info)
        torch.cuda.empty_cache()

    sys.stdout.write('\n')
    return cam_infos


def optimize_depth(source, target, mask, depth_weight, prune_ratio=0.05):
    """
    Arguments
    =========
    source: np.array(h,w)
    target: np.array(h,w)
    mask: np.array(h,w):
        array of [True if valid pointcloud is visible.]
    depth_weight: np.array(h,w):
        weight array at loss.
    Returns
    =======
    refined_source: np.array(h,w)
        literally "refined" source.
    loss: float
    """
    source = torch.from_numpy(source).cuda()
    target = torch.from_numpy(target).cuda()
    mask = torch.from_numpy(mask).cuda()
    depth_weight = torch.from_numpy(depth_weight).cuda()

    # Prune some depths considered "outlier"     
    with torch.no_grad():
        target_depth_sorted = target[target>1e-7].sort().values
        min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
        max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

        mask2 = target > min_prune_threshold
        mask3 = target < max_prune_threshold
        mask = torch.logical_and( torch.logical_and(mask, mask2), mask3)

    source_masked = source[mask]
    target_masked = target[mask]
    depth_weight_masked = depth_weight[mask]
    # tmin, tmax = target_masked.min(), target_masked.max()

    # # Normalize
    # target_masked = target_masked - tmin 
    # target_masked = target_masked / (tmax-tmin)

    scale = torch.ones(1).cuda().requires_grad_(True)
    shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

    optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
    loss = torch.ones(1).cuda() * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0

    while abs(loss_ema - loss_prev) > 1e-5:
        source_hat = scale*source_masked + shift
        loss = torch.mean(((target_masked - source_hat)**2)*depth_weight_masked)

        # penalize depths < 1
        loss_hinge1 = 0.0
        if (source_hat <= 0.0).any():
            loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()

        loss = loss + loss_hinge1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        iteration += 1
        if iteration % 1000 == 0:
            print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()       
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    loss = loss.item()
    print(f"loss ={loss:10.5f}")

    with torch.no_grad():
        refined_source = (scale*source + shift) 
    torch.cuda.empty_cache()
    return refined_source.cpu().numpy(), loss


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T               if 'x' in vertices else None
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0 if 'red' in vertices else None
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T              if 'nx' in vertices else None
    errors = vertices['xyzerr']/(np.min(vertices['xyzerr'] + 1e-8))                      if 'xyzerr' in vertices else None
    return BasicPointCloud(points=positions, colors=colors, normals=normals, errors=errors)


def storePly(path, xyz, rgb, xyzerr=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('xyzerr', 'f4')]

    normals = np.zeros_like(xyz)
    if xyzerr is None:
        xyzerr = np.ones((xyz.shape[0],1))

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, xyzerr), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def refineColmapWithIndex(path, train_index, pc_name):
    """ result
    'cam_extrinsics' and 'point3D.ply' contains the points observed in (at least 2) train-views 
    """

    bin_path = os.path.join(path, f"sparse/0/{pc_name}.bin")

    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    xyz, rgb, err = read_points3D_binary(bin_path)
    # save in ply format
    os.makedirs(os.path.join(path, 'plydummy'), exist_ok=True)
    ply_path = os.path.join(path, 'plydummy', f"points3D_{int(datetime.now().timestamp())}.ply") # k-shot train
    storePly(ply_path, xyz, rgb, xyzerr=err)

    return ply_path, cam_extrinsics, cam_intrinsics


def pick_idx_from_360(path, train_idx, kshot, center, num_trials=100_000):
    """
    [Taekkii]
    randomly pick ONE index from train_idx.
    The rest are decided by RANSAC-like brute search method to match criterion:
    - let vectors v: center to each camera positions.
    - maximize: prod(angle between two vectors)

    ARGUMENTS
    ---------
    path: colmap standard directory path.
    train_idx: list of train indice.
    kshot: # of shots (int)
    center: ndarray(3): center saved in the path.
    num_trials: number of RANSAC search trials.

    RETURNS
    -------
    indice: (list)selected train-indice. Not guaranteed to be optimal.
            NOTE: Same seed always results in same indice.
    """

    # guard.
    if kshot>=len(train_idx):
        return train_idx

    # Get camera positions. Kinda redundant code, but we read extrinsics again.
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

    cam_locs = []
    
    for idx, key in enumerate(sorted(cam_extrinsics, key=lambda x:cam_extrinsics[x].name)):

        if idx not in train_idx:
            continue

        cam_extrinsic = cam_extrinsics[key]
        R = np.transpose(qvec2rotmat(cam_extrinsic.qvec))
        T = np.array(cam_extrinsic.tvec)
        cam_locs.append(-T@R)

    cam_locs = np.stack(cam_locs)

    # fix pivot index.
    pivot = np.random.randint(len(train_idx))

    choice_indice_pull = np.array([e for e in range(len(train_idx)) if e != pivot])
    candidate_indice, candidate_criterion = None , 0.0

    pivot = np.array([pivot])

    # RANSAC-like random search.
    for _ in tqdm(range(num_trials), desc='Choosing best indice'):
        indice = np.random.choice(choice_indice_pull, kshot-1, replace=False)
        indice = np.concatenate([indice,pivot]) # Always include pivot.

        selected_camlocs = cam_locs[indice] # (kshot,3)
        vectors = selected_camlocs - center # (kshot,3)

        # Take unit vector (makes my life easier.)
        vectors = vectors / np.linalg.norm(vectors,axis=-1,keepdims=True)

        radians = np.arccos( (vectors * vectors[:,None,:]).sum(axis=-1).clip(-0.99999 , 0.99999) ) # (kshot,3) * (kshot,1,3) --> (kshot,kshot,3) --(sum)--> (kshot,kshot) 

        criterion = radians.prod() # Strictly speaking, sqrt of this criterion.

        if candidate_criterion < criterion:
            candidate_criterion = criterion
            candidate_indice = indice

    final_indice = (np.array(train_idx)[candidate_indice]).tolist()
    return final_indice


def readColmapSceneInfo(path, images, eval, kshot=1000, seed=0, white_background=False, pc_name='points3D', Depthoptim=True):
    ## load split_idx.json
    with open(os.path.join(path, "split_index.json"), "r") as jf:
        jsonf = json.load(jf)
        train_idx, test_idx = jsonf["train"], jsonf["test"]

    reading_dir = "images" if images == None else images
    mask_dir = os.path.join(path, "mask")

    scene_center_path = os.path.join(path, "center.npy")

    np.random.seed(seed)
    if os.path.exists(scene_center_path) and eval:
        train_idx = pick_idx_from_360(path, train_idx, kshot, center=np.load(scene_center_path))
    else:
        train_idx = sorted(np.random.choice(train_idx, size=min(kshot, len(train_idx)), replace=False)) if eval else np.arange(len(train_idx)).tolist()

    ### refineColmapWithIndex() remove the cameras and features except the train set
    ply_path, cam_extrinsics, cam_intrinsics = refineColmapWithIndex(path, train_idx, pc_name)

    ### making pcd with the features captured from train_cam
    pcd = fetchPly(ply_path)

    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, Depthoptim=Depthoptim, mask_folder=mask_dir,
                                  images_folder=os.path.join(path, reading_dir), pcd=pcd, train_idx=train_idx, test_idx=test_idx, white_background=white_background).copy()

    if eval:
        train_cam_infos = [cam_infos[i] for i in train_idx]
        test_cam_infos = [cam_infos[i] for i in test_idx]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []     

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def depth_minmax(depth_name):
    depths = np.stack(depths)
    batch, vx, vy = np.where(depths!=0)

    valid_depth = depths[batch, vx, vy]
    return valid_depth.min(), valid_depth.max()


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", use_depth=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]

        depth_namelist = sorted(glob(os.path.join(path, '/'.join(frames[0]["file_path"].split('/')[:-1])+ "/*depth*")))
        if len(depth_namelist)>0:
            depths = []
            for i in range(len(depth_namelist)):
                depths.append(np.load(depth_namelist[i])) # normalized [0,1]
            depths = np.stack(depths)
            batch, vx, vy = np.where(depths!=0)

            valid_depth = depths[batch, vx, vy]
            dmin, dmax = valid_depth.min(), valid_depth.max()

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if use_depth and os.path.exists(os.path.join(path, frame["file_path"].replace("/train/", "/depths_train/")+'.npy')):
                depth = np.load(os.path.join(path, frame["file_path"].replace("/train/", "/depths_train/")+'.npy'))
                if os.path.exists(os.path.join(path, frame["file_path"].replace("/train/", "/masks_train/")+'.png')):
                    mask = imageio.v3.imread(os.path.join(path, frame["file_path"].replace("/train/", "/masks_train/")+'.png'))[:,:,0]/255.
                else:
                    mask = np.ones_like(depth)
                final_depth = depth*mask
            else:
                final_depth = None

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=final_depth,
                                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], depthloss=None))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png", use_depth=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, use_depth=use_depth)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, use_depth=use_depth)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
