import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import tensorflow as tf
import open3d.core as o3c
import simpose as sp
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from functools import cached_property


class SceneIntegration:
    def __init__(self, use_color: bool = True, voxel_size=3 / 1024, block_resolution=16):
        self.device = o3c.Device("cuda:0")
        # self.device = o3c.Device("cpu:0")

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

    @cached_property
    def scene_mesh(self) -> o3d.geometry.TriangleMesh:
        # return self.vbg.extract_triangle_mesh().cpu()
        print("Extracting pointcloud...")
        pcd = self.vbg.extract_point_cloud().to_legacy()

        # o3d.visualization.draw_geometries([pcd])

        # pcd, _ = pcd.remove_statistical_outlier(20, 1.0)

        # at least n points in the sphere of radius r
        print("Cleaning up...")
        pcd, _ = pcd.remove_radius_outlier(500, 0.025, print_progress=True)

        # o3d.visualization.draw_geometries([pcd])

        print("Converting to mesh...")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=11, width=0, scale=1.1, linear_fit=False, n_threads=-1
        )
        # o3d.visualization.draw_geometries([mesh])

        print("Removing big triangles...")
        # calculate triangle surface areas
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        tri_indices = mesh_t.triangle.indices.cuda()  # [N, 3] indices into vertices
        vertices = mesh_t.vertex.positions.cuda()  # [M, 3] vertex positions

        # calculate triangle areas
        a = vertices[tri_indices[:, 0]]
        b = vertices[tri_indices[:, 1]]
        c = vertices[tri_indices[:, 2]]
        ab = b - a  # [N, 3]
        ac = c - a  # [N, 3]

        # cross product in open3d
        cross = o3d_cross(ab, ac)  # [N, 3]
        area = 0.5 * (cross * cross).sum(1).sqrt()  # [N]

        tri_mask = area > 1e-4

        mesh.remove_triangles_by_mask(tri_mask.cpu().numpy())
        mesh.remove_unreferenced_vertices()

        # o3d.visualization.draw_geometries([mesh])

        return mesh

    def render_depth(
        self, w: int, h: int, intrinsics: np.ndarray, extrinsics: np.ndarray, depth_scale=1000.0
    ) -> np.ndarray:
        scene = o3d.t.geometry.RaycastingScene()
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(intrinsics, extrinsics, w, h)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.scene_mesh))
        result = scene.cast_rays(rays)
        depth = result["t_hit"].cpu().numpy()
        depth[depth > 5.0] = 0.0  # 5m is the max depth
        return depth * depth_scale

    def save(self, path: Path = Path("vbg.npz")):
        self.vbg.save(str(path))

    def load(self, path: Path = Path("vbg.npz")):
        self.vbg = self.vbg.load(str(path)).cpu()
        # self.vbg = self.vbg.cuda()


def o3d_cross(a: o3d.core.Tensor, b: o3d.core.Tensor):
    """Cross product in open3d
    params:
        a: [N, 3]
        b: [N, 3]
    returns:
        [N, 3]
    """
    x = a[:, 1:2] * b[:, 2:3] - a[:, 2:3] * b[:, 1:2]
    y = a[:, 2:3] * b[:, 0:1] - a[:, 0:1] * b[:, 2:3]
    z = a[:, 0:1] * b[:, 1:2] - a[:, 1:2] * b[:, 0:1]
    return o3d.core.concatenate([x, y, z], axis=1)
