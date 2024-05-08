import numpy as np
from robotools.geometry import invert_homogeneous
import open3d as o3d


class MaskGenerator:

    def render_object_mask(
        self,
        obj: o3d.geometry.TriangleMesh,
        obj_pose: np.ndarray,
        cam_pose: np.ndarray,
        width: int,
        height: int,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        """render object mask from camera"""
        scene, rays = self._get_raycasting_scene(
            obj,
            obj_pose=obj_pose,
            cam_pose=cam_pose,
            width=width,
            height=height,
            intrinsics=intrinsics,
        )
        o3d_mask = scene.test_occlusions(rays).numpy()
        mask = np.where(o3d_mask == True, 1, 0).astype(np.uint8)
        return mask

    def calculate_occluded_mask(
        self,
        unoccluded_mask: np.ndarray,
        obj: o3d.geometry.TriangleMesh,
        obj_pose: np.ndarray,
        cam_pose: np.ndarray,
        width: int,
        height: int,
        intrinsics: np.ndarray,
        depth: np.ndarray,
        occlusion_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """by knowing what the depth image *should* look like, we can calculate the occluded mask"""

        scene, rays = self._get_raycasting_scene(
            obj,
            obj_pose=obj_pose,
            cam_pose=cam_pose,
            width=width,
            height=height,
            intrinsics=intrinsics,
        )
        ans = scene.cast_rays(rays)
        rendered_depth = ans["t_hit"].numpy()

        diff = np.abs(rendered_depth - depth)

        occluded_mask = np.where(
            np.logical_and(diff < occlusion_threshold, unoccluded_mask == 1), 1, 0
        ).astype(np.uint8)

        return occluded_mask, rendered_depth

    def _get_raycasting_scene(
        self,
        obj: o3d.geometry.TriangleMesh,
        obj_pose: np.ndarray,
        cam_pose: np.ndarray,
        width: int,
        height: int,
        intrinsics: np.ndarray,
        visualize_debug: bool = False,
    ) -> tuple[o3d.t.geometry.RaycastingScene, o3d.core.Tensor]:
        scene = o3d.t.geometry.RaycastingScene()

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(obj)
        mesh_t.transform(obj_pose)
        mesh_id = scene.add_triangles(mesh_t)

        if visualize_debug:
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                width,
                height,
                intrinsics,
                invert_homogeneous(cam_pose),
                1.0,
            )

            mesh = o3d.geometry.TriangleMesh(obj)  # copy
            mesh.compute_vertex_normals()
            mesh.transform(obj_pose)
            o3d.visualization.draw_geometries([mesh, frustum])  # type: ignore

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(intrinsics),
            o3d.core.Tensor(invert_homogeneous(cam_pose)),
            width,
            height,
        )

        return scene, rays
