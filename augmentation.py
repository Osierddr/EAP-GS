import numpy as np
from sklearn.cluster import DBSCAN
from skimage.draw import disk
from PIL import Image
import os
import sys
import json
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
from utils.graphics_utils import focal2fov
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud

path = 'data'

with open(os.path.join(path, "split_index.json"), "r") as jf:
    jsonf = json.load(jf)
    train_idx, test_idx = jsonf["train"], jsonf["test"]


xyz, rgb, err = read_points3D_binary(os.path.join(path, "sparse/0/points3D.bin"))
xyzerr = err if err is not None else np.ones((xyz.shape[0], 1))

cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)


dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('xyzerr', 'f4')]
normals = np.zeros_like(xyz)
elements = np.empty(xyz.shape[0], dtype=dtype)
attributes = np.concatenate((xyz, normals, rgb, xyzerr), axis=1)
elements[:] = list(map(tuple, attributes))

vertex_element = PlyElement.describe(elements, 'vertex')
ply_data = PlyData([vertex_element])

vertices = ply_data['vertex']
positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T if 'x' in vertices else None
colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0 if 'red' in vertices else None
normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T if 'nx' in vertices else None
errors = vertices['xyzerr'] / (np.min(vertices['xyzerr'] + 1e-8)) if 'xyzerr' in vertices else None

pcd = BasicPointCloud(points=positions, colors=colors, normals=normals, errors=errors)

for idx, key in enumerate(sorted(cam_extrinsics, key=lambda x: cam_extrinsics[x].name)):
    if idx not in train_idx and idx not in test_idx:
        continue

    sys.stdout.write(f'\rReading camera {idx+1}/{len(cam_extrinsics)}')
    sys.stdout.flush()

    extr = cam_extrinsics[key]
    intr = cam_intrinsics[extr.camera_id]
    height, width = intr.height, intr.width

    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)

    focal_length_x = intr.params[0]
    focal_length_y = intr.params[0] if intr.model == "SIMPLE_PINHOLE" else intr.params[1]
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)

    K = np.array([[focal_length_x, 0, width / 2], [0, focal_length_y, height / 2], [0, 0, 1]])

    image_path = os.path.join(path, "images")
    image = Image.open(os.path.join(image_path, os.path.basename(extr.name)))

    if pcd and idx in train_idx:
        depthmap = np.zeros((height, width))
        cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3, 1))
        valid_idx = np.where(np.logical_and.reduce((cam_coord[2] > 0, cam_coord[0] / cam_coord[2] >= 0,
                                                     cam_coord[0] / cam_coord[2] <= width - 1, 
                                                     cam_coord[1] / cam_coord[2] >= 0, 
                                                     cam_coord[1] / cam_coord[2] <= height - 1)))   [0]
        pts_depths = cam_coord[-1:, valid_idx]
        cam_coord = cam_coord[:2, valid_idx] / cam_coord[-1:, valid_idx]
        depthmap[np.round(cam_coord[1]).astype(np.int32).clip(0, height - 1),
                 np.round(cam_coord[0]).astype(np.int32).clip(0, width - 1)] = pts_depths

        indices = np.array(np.nonzero(depthmap)).transpose()
        labels = DBSCAN(eps=15, min_samples=10).fit(indices).labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        mask = np.ones((height, width), dtype=np.uint8)
        for cluster_id in range(num_clusters):
            cluster_points = indices[labels == cluster_id]
            if cluster_points.size > 0:
                for point in cluster_points:
                    x, y = int(point[1]), int(point[0])
                    rr, cc = disk((y, x), 10, shape=mask.shape)
                    mask[rr, cc] = 0

        img = np.array(image) * mask[:, :, np.newaxis]
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(image_path, f"{(idx+len(train_idx)):05d}.jpg"))
