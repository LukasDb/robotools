import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import tensorflow as tf
import open3d.core as o3c
import simpose as sp
from scipy.spatial.transform import Rotation as R
from pathlib import Path


class SceneIntegration:
    def __init__(self, use_color: bool = True, voxel_size=2.0 / 1024, block_resolution=16):
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
                voxel_size=voxel_size,  # -> 3x3x3 m space with 512x512x512 voxels
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
        # pcl, _ = pcl.remove_statistical_outliers(20, 2.0)

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
                1000.0,
                2.0,
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
        self.vbg = self.vbg.cuda()
        result = self.vbg.ray_cast(
            block_coords=self.vbg.hashmap().key_tensor(),
            intrinsic=intrinsics,
            extrinsic=extrinsics,
            width=w,
            height=h,
            render_attributes=["depth"],  # "normal", "color", "index", "interp_ratio"],
            depth_scale=depth_scale,
            depth_min=0.0,
            depth_max=5.0,
            weight_threshold=1,
            range_map_down_factor=8,
        )
        return result["depth"].cpu().numpy()[..., 0]

    def save(self, path: Path = Path("vbg.npz")):
        self.vbg.save(str(path))

    def load(self, path: Path = Path("vbg.npz")):
        self.vbg = self.vbg.load(str(path))
        self.vbg = self.vbg.cuda()


# exit()

# pcd = vbg.extract_point_cloud()

# mesh = vbg.extract_triangle_mesh()
# mesh.compute_triangle_normals()
# mesh.compute_vertex_normals()
# mesh = mesh.to_legacy()

# mesh = mesh.filter_smooth_taubin(number_of_iterations=100)
# mesh.compute_vertex_normals()

# o3d.visualization.draw_geometries(vis)
# o3d.visualization.draw_geometries([pcd.to_legacy()])
# o3d.visualization.draw_geometries([mesh])


# result = vbg.ray_cast(
#     block_coords=frustum_block_coords,
#     intrinsic=intrinsics,
#     extrinsic=first_extrinsics,
#     width=w,
#     height=h,
#     render_attributes=["depth"],  # "normal", "color", "index", "interp_ratio"],
#     depth_scale=1000.0,
#     depth_min=0.0,
#     depth_max=5.0,
#     weight_threshold=1,
#     range_map_down_factor=8,
# )

# fig, axs = plt.subplots(3, 1)
# # Colorized depth
# d_min = 0.2
# d_max = 1.0
# depth_reconst = result["depth"]
# # as np array
# depth_reconst = depth_reconst.cpu().numpy()[..., 0]
# colorized_depth = o3d.t.geometry.Image(depth_reconst).colorize_depth(1000.0, d_min, d_max)
# axs[0].imshow(colorized_depth.as_tensor().cpu().numpy())
# axs[0].set_title("depth (Reconstructed)")

# colorized_depth_raw = o3d.t.geometry.Image(first_depth).colorize_depth(1000.0, d_min, d_max)
# axs[1].imshow(colorized_depth_raw.as_tensor().cpu().numpy())
# axs[1].set_title("depth (Realsense)")


# diff = np.abs(depth_reconst - first_depth)
# colorized_diff = o3d.t.geometry.Image(diff).colorize_depth(1000.0, 0, 0.1)
# axs[2].imshow(colorized_diff.as_tensor().cpu().numpy())
# axs[2].set_title("Difference")

# print(f"{np.mean(diff) = }; {np.max(diff) = }; {np.min(diff) = }")

# # axs[0, 1].imshow(result["normal"].cpu().numpy())
# # axs[0, 1].set_title("normal")

# # axs[1, 0].imshow(result["color"].cpu().numpy())
# # axs[1, 0].set_title("color via kernel")

# plt.show()

# color_img = o3d.t.geometry.Image(first_color.numpy())
# rgbd_reconst = o3d.t.geometry.RGBDImage(
#     color_img, o3d.t.geometry.Image(depth_reconst), aligned=True
# )
# rgbd_raw = o3d.t.geometry.RGBDImage(color_img, o3d.t.geometry.Image(first_depth), aligned=True)

# pcl_reconst = o3d.t.geometry.PointCloud().create_from_rgbd_image(
#     rgbd_reconst, intrinsics, first_extrinsics, depth_scale=1000.0
# )
# # pcl_reconst.paint_uniform_color((1.0, 0.0, 0.0))
# pcl_raw = o3d.t.geometry.PointCloud().create_from_rgbd_image(
#     rgbd_raw, intrinsics, first_extrinsics, depth_scale=1000.0
# )

# # pcl_raw.paint_uniform_color((0.0, 1.0, 0.0))
# pcl_raw.translate([0.0, 0.0, 0.3])


# o3d.visualization.draw_geometries([pcl_reconst.to_legacy(), pcl_raw.to_legacy(), origin])
