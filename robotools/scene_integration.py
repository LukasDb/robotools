import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import tensorflow as tf
import open3d.core as o3c
import simpose as sp
from scipy.spatial.transform import Rotation as R
from pathlib import Path


class SceneIntegration:
    def __init__(self, use_color: bool = True, voxel_size=2.5 / 1024, block_resolution=16):
        #self.device = o3c.Device("cuda:0")
        self.device = o3c.Device("cpu:0")

        self.USE_COLOR = use_color
        self.voxel_size = voxel_size
        self.block_resolution = block_resolution

        if use_color:
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"),
                attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                attr_channels=((1), (1), (3)),
                voxel_size=voxel_size,
                block_resolution=block_resolution,  # local voxel block resolution
                block_count=50000,
            )
        else:
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight"),
                attr_dtypes=(o3c.float32, o3c.float32),
                attr_channels=((1), (1)),
                voxel_size=voxel_size,
                block_resolution=block_resolution,
                block_count=50000,
            )

    def integrate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        """only in meters"""
        # first we locate blocks, that contain points from the rgbd image

        h, w = depth.shape[:2]
        depth = o3d.t.geometry.Image((depth * 1000).astype(np.uint16))
        color = o3d.t.geometry.Image(rgb)

        intrinsics = o3c.Tensor(intrinsics.astype(np.float64))

        r = extrinsics[:3, :3]
        t = extrinsics[:3, 3]

        # extrinsic = o3c.Tensor(extrinsic)  # , device=device)
        extrinsic_o3d = np.eye(4)
        extrinsic_o3d[:3, :3] = r.T
        extrinsic_o3d[:3, 3] = -t @ r  # .T.T
        extrinsic_o3d = o3c.Tensor(extrinsic_o3d)

        pcl = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth,
            intrinsics,
            depth_scale=1000.0,
            extrinsics=extrinsic_o3d,
            stride=1,
        )
        pcl, _ = pcl.remove_statistical_outliers(20, 1.0)

        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            pcl, trunc_voxel_multiplier=8.0
        )

        # now we can process the voxels -> scene integration
        if self.USE_COLOR:
            self.vbg.integrate(
                frustum_block_coords,
                depth,
                color,
                intrinsics,
                intrinsics,
                extrinsic_o3d,
                1000.0,  # depth scale
                2.0,  # depth max
            )
        else:
            self.vbg.integrate(
                frustum_block_coords,
                depth,
                intrinsics,
                extrinsic_o3d,
                1000.0,
                2.0,
            )

    def render_depth(
        self, w: int, h: int, intrinsics: np.ndarray, extrinsics: np.ndarray, depth_scale=1000.0
    ) -> np.ndarray:
        mesh = self.vbg.extract_triangle_mesh().cpu()
        scene = o3d.t.geometry.RaycastingScene()
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(intrinsics, extrinsics, w, h)
        scene.add_triangles(mesh)
        result = scene.cast_rays(rays)
        depth = result["t_hit"].cpu().numpy()
        depth[depth > 5.0] = 0.0  # 5m is the max depth
        return depth * depth_scale

    def save(self, path: Path = Path("vbg.npz")):
        self.vbg.save(str(path))

    def load(self, path: Path = Path("vbg.npz")):
        self.vbg = self.vbg.load(str(path))
        #self.vbg = self.vbg.cuda()
