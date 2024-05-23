import os

# QT_QPA_PLATFORM = offscreen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import tensorflow as tf

from robotools.geometry import distance_from_matrices, invert_homogeneous

for dev in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(dev, True)
import numpy as np
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import click
import matplotlib.pyplot as plt
import open3d as o3d
import requests
import base64
import io
import json
import trimesh


import simpose as sp
import robotools as rt
from robotools.trajectory import SphericalTrajectory, TrajectoryExecutor


"""
1) go through the dataset and run yolo + pvn3d on every frame
2) cluster predictions and find object instances
2.1) show some predictions and let user confirm/correct
3) for each object instance, run ICP on the depth data to get the pose
4) for each object instance, run multi frame stabilization to get the mask
4.1 show some predictions and let the user confirm/correct
5 save the poses/scene to file


WIP
"""

IOU_THRESHOLD = 0.7


@click.command()
@click.option("--capture", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/scene_construction")
def main(capture: bool, output: Path) -> None:
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene_combined.yaml")))

    # extract names from scene.yaml
    # cam: rt.camera.Camera = scene._entities["Realsense_121622061798"]
    cam: rt.camera.Camera = scene._entities["ZED-M"]
    bg: rt.utility.BackgroundMonitor = scene._entities["Background Monitor"]
    bg.disable()
    robot: rt.robot.Robot = scene._entities["crx"]

    tfds = sp.data.TFRecordDataset
    dataset = tfds.get(
        output,
        get_keys=[
            tfds.CAM_LOCATION,
            tfds.CAM_ROTATION,
            tfds.CAM_MATRIX,
            tfds.OBJ_POS,
            tfds.OBJ_ROT,
            tfds.RGB,
            tfds.DEPTH,
        ],
    ).enumerate()

    scene_integration = rt.SceneIntegration(use_color=True)
    scene_integration.load()

    print("Loaded.")
    scene_mesh = scene_integration.vbg.extract_triangle_mesh()
    scene_mesh.compute_triangle_normals()
    scene_mesh.compute_vertex_normals()
    scene_mesh = scene_mesh.to_legacy()

    scene_pcl = scene_integration.vbg.extract_point_cloud()

    scene_mesh = scene_pcl.to_legacy()  # for testing

    mesh_root_dir = Path("/home/aismart/Desktop/RGBD-capture/robotools/0_meshes").expanduser().glob("*")
    objs = {}
    for mesh_dir in mesh_root_dir:
        mesh_path = mesh_dir.joinpath(f"{mesh_dir.name}.obj")
        objs[mesh_dir.name] = o3d.io.read_triangle_mesh(str(mesh_path), True)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Test iou_mesh
    # obj = objs["cpsduck"]
    # obj.compute_triangle_normals()
    # obj.compute_vertex_normals()

    # obj1 = o3d.geometry.TriangleMesh(obj)
    # obj2 = o3d.geometry.TriangleMesh(obj)
    # obj2.translate([1e-4] * 3)

    # iou = iou_mesh(obj1, obj2)
    # print(f"{iou = }")

    # obj2.rotate(R.from_euler("xyz", [0.0, 0.0, 30.0], degrees=True).as_matrix(), [0, 0, 0])
    # iou = iou_mesh(obj1, obj2)
    # print(f"{iou = }")

    # obj2.rotate(R.from_euler("xyz", [0.0, 0.0, -30.0], degrees=True).as_matrix(), [0, 0, 0])
    # obj2.translate([0.0, 0.0, 0.05])
    # iou = iou_mesh(obj1, obj2)
    # print(f"{iou = }")

    # obj2.translate([0.0, 0.0, 1.0])
    # iou = iou_mesh(obj1, obj2)
    # print(f"{iou = }")
    # exit()

    # all_obj_poses = {}
    obj_poses = {}
    show = True
    for step, data in dataset:
        extrinsics = np.eye(4)
        extrinsics[:3, 3] = data[tfds.CAM_LOCATION].numpy()
        extrinsics[:3, :3] = R.from_quat(data[tfds.CAM_ROTATION].numpy()).as_matrix()
        extrinsics_inv = invert_homogeneous(extrinsics)
        h, w = data[tfds.DEPTH].shape[:2]

        intrinsics = data[tfds.CAM_MATRIX].numpy().astype(np.float64)

        depth_reconst = (
            scene_integration.render_depth(
                w,
                h,
                data[tfds.CAM_MATRIX].numpy().astype(np.float64),
                extrinsics_inv.astype(np.float64),
            )
            / 1000.0
        )  # to m
        depth_raw = data[tfds.DEPTH].numpy()

        rgb = data[tfds.RGB].numpy().astype(np.uint8).copy()

        detections = infer_poses(rgb, depth_reconst, intrinsics)

        # draw known objects faintly
        faint_map = np.zeros_like(rgb)
        for pose_list in obj_poses.values():
            for pose in pose_list:
                affine_matrix = invert_homogeneous(extrinsics) @ pose
                rot = affine_matrix[:3, :3]
                t = affine_matrix[:3, 3]
                rvec = cv2.Rodrigues(rot)
                rvec = rvec[0].flatten()
                tvec = t.flatten()
                cv2.drawFrameAxes(faint_map, intrinsics, np.zeros((4, 1)), rvec, tvec, 0.1, 3)  # type: ignore

        for det in detections:
            cls = det["cls"]
            bbox = det["bbox"]
            bbox = [int(x) for x in bbox]
            confidence = float(det["confidence"])

            # ICP refinement of non-inferred objects for each image is not necessary, since we use the scene pcl anyway (view independent)
            affine_matrix = np.array(det["pose"])
            pose = extrinsics @ affine_matrix

            transformed_obj = o3d.t.geometry.TriangleMesh.from_legacy(objs[cls])
            transformed_obj.transform(pose)
            obj_bbox = transformed_obj.get_axis_aligned_bounding_box()
            # grow obj_bbox by 10%
            obj_bbox = o3d.t.geometry.AxisAlignedBoundingBox(
                obj_bbox.min_bound - 0.1 * (obj_bbox.max_bound - obj_bbox.min_bound),
                obj_bbox.max_bound + 0.1 * (obj_bbox.max_bound - obj_bbox.min_bound),
            )

            obj_pcl = o3d.t.geometry.PointCloud(scene_pcl).cpu().crop(obj_bbox).cuda()

            is_valid_detection, pose = icp_refinement(pose, obj_pcl, objs[cls])

            # since we use ICP with the scene PCL, we do not need to fuse multi-frame poses (we already use a multi-view PCL!)
            obj = objs[cls]
            ious = [iou_pose(obj, pose, x) for x in obj_poses.get(cls, [])]
            if len(ious) > 0:
                print(f"{max(ious) = }")
            is_new_detection = len(ious) == 0 or max(ious) < IOU_THRESHOLD

            # make this consider rotation as well? -> no! then rotational symmetry would be a problem

            # bbox
            #           valid  |  invalid
            # new        green | f_red
            # not new   f_blue | f_red

            prefix = ""
            if is_valid_detection:
                if is_new_detection:
                    color = (0, 255, 0)
                    to_draw = rgb
                    prefix = "NEW: "
                else:
                    color = (0, 255, 0)
                    to_draw = faint_map
                    prefix = "Known: "
            else:
                color = (255, 0, 0)
                to_draw = faint_map
                prefix = "Invalid: "

            cv2.rectangle(to_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(
                to_draw,
                f"{prefix}{cls} {confidence:.2f}",
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            if not is_new_detection or not is_valid_detection:
                continue

            obj_poses.setdefault(cls, [])
            obj_poses[cls].append(pose)
            # only draw detected!
            affine_matrix = invert_homogeneous(extrinsics) @ pose
            rot = affine_matrix[:3, :3]
            t = affine_matrix[:3, 3]
            rvec = cv2.Rodrigues(rot)
            rvec = rvec[0].flatten()
            tvec = t.flatten()
            cv2.drawFrameAxes(rgb, intrinsics, np.zeros((4, 1)), rvec, tvec, 0.05, 3)  # type: ignore

        # blend over rgb
        rgb = cv2.addWeighted(rgb, 1.0, faint_map, 0.4, 0)

        print(f"Found {len(detections)} objects in the image")
        [
            print(f"Found {len(poses)} objects of <{cls}> in the scene")
            for cls, poses in obj_poses.items()
        ]
        print()
        cv2.imshow("img", rgb[:, :, ::-1])  # display as 'bgr'
        cv2.waitKey(1)
        while show:
            key = cv2.waitKey(0)
            if key == ord(" "):
                break
            elif key == ord("q"):
                show = False
                break
    cv2.destroyAllWindows()

    # fuse objs
    vis_objs = []

    for cls, poses in obj_poses.items():
        print(f"Found {len(poses)} objects of <{cls}> in the scene")
        obj = objs[cls]
        for pose in poses:
            obj_copy = o3d.geometry.TriangleMesh(obj)
            obj_copy.transform(pose)
            vis_objs.append(obj_copy)

            obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            obj_frame.transform(pose)
            vis_objs.append(obj_frame)

    o3d.visualization.draw_geometries([scene_mesh, origin] + vis_objs)

    # TODO dump things to disk somehow (scene.yaml?)
    with open("labelled_objects.json", "w") as f:
        json.dump(obj_poses, f, default=lambda x: x.tolist(), indent=2)


import numba


# @numba.njit
def icp(
    scene_pcl: np.ndarray, vertices: np.ndarray, triangles: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    pose, _, cost = trimesh.registration.icp(
        scene_pcl,
        mesh,
        initial=invert_homogeneous(pose),
        threshold=1e-05,
        max_iterations=2,
    )
    return invert_homogeneous(pose), cost


def iou_pose(mesh, pose1, pose2):
    mesh1 = o3d.geometry.TriangleMesh(mesh)
    mesh2 = o3d.geometry.TriangleMesh(mesh)
    mesh1.transform(pose1)
    mesh2.transform(pose2)
    return iou_mesh(mesh1, mesh2)


def iou_mesh(mesh1, mesh2):
    mesh1.paint_uniform_color([1, 0, 0])
    mesh2.paint_uniform_color([0, 1, 0])
    mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(mesh1)
    mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh2)

    # get oriented bounding boxes
    _bbox1 = mesh1.get_oriented_bounding_box()
    _bbox2 = mesh2.get_oriented_bounding_box()

    _bbox1 = trimesh_from_oriented_bbox(_bbox1)
    _bbox2 = trimesh_from_oriented_bbox(_bbox2)

    bbox1 = trimesh.Trimesh(
        vertices=_bbox1.vertex.positions.cpu().numpy(), faces=_bbox1.triangle.indices.cpu().numpy()
    )
    bbox2 = trimesh.Trimesh(
        vertices=_bbox2.vertex.positions.cpu().numpy(), faces=_bbox2.triangle.indices.cpu().numpy()
    )

    # visualize using trimesh
    intersection = bbox1.intersection(bbox2)
    if intersection.is_empty:
        return 0.0
    if not intersection.is_watertight:
        print("Warning: intersection not watertight")

    # intersection.show()
    inter_volume = intersection.volume
    bbox1_volume = bbox1.volume
    bbox2_volume = bbox2.volume

    # inter_volume = intersection.get_volume()
    union_volume = bbox1_volume + bbox2_volume - inter_volume
    iou = np.clip(inter_volume / union_volume, 0, 1)
    return iou


def trimesh_from_oriented_bbox(bbox: o3d.t.geometry.OrientedBoundingBox, color=[1, 0, 0]):
    # convert oriented bounding boxes to triangle mesh
    box_dims = bbox.extent.cpu().numpy()
    bbox_mesh = o3d.t.geometry.TriangleMesh.create_box(*box_dims)
    # box starts at origin and extends positive
    bbox_mesh.translate(-box_dims / 2)
    bbox_mesh.rotate(bbox.rotation.cpu().numpy(), center=[0, 0, 0])
    bbox_mesh.translate(bbox.center.cpu().numpy())

    bbox_mesh.vertex.colors = o3d.core.Tensor(
        [color] * bbox_mesh.vertex.positions.shape[0], dtype=o3d.core.Dtype.Float32
    )
    return bbox_mesh


def icp_refinement(
    initial_pose, scene_pcl: o3d.t.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh
) -> tuple[bool, np.ndarray]:

    # data
    # source = scene_pcl.voxel_down_sample(0.001).cuda()
    source = scene_pcl.cuda()
    target = o3d.t.geometry.PointCloud.from_legacy(mesh.sample_points_poisson_disk(10_000))
    target = target.cuda()

    if source.is_empty() or target.is_empty():
        return False, initial_pose

    # source_vis, target_vis = source.to_legacy(), target.to_legacy()
    # source_vis.paint_uniform_color([1, 0, 0])
    # target_vis.paint_uniform_color([0, 1, 0])
    # target_vis.transform(initial_pose)
    # o3d.visualization.draw_geometries([target_vis, source_vis])

    # params
    MAX_ITERATIONS = 30
    FITNESS_THRESHOLD = 0.021
    INLIER_RMSE_THRESHOLD = 0.0065
    MAX_CORRESPONDENCE_DISTANCE = 0.015

    mu, sigma = 0, 0.1  # mean and standard deviation
    treg = o3d.t.pipelines.registration
    estimation = treg.TransformationEstimationPointToPlane(
        treg.robust_kernel.RobustKernel(treg.robust_kernel.RobustKernelMethod.TukeyLoss, sigma)
    )

    criteria = treg.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=MAX_ITERATIONS
    )

    result = treg.icp(
        source,
        target,
        MAX_CORRESPONDENCE_DISTANCE,
        invert_homogeneous(initial_pose),
        estimation,
        criteria,
        -1,
    )

    print(
        f"{result.fitness = :.4f}, {result.inlier_rmse = :.4f};  {result.num_iterations = }"
    )

    # source_vis, target_vis = source.to_legacy(), target.to_legacy()
    # source_vis.paint_uniform_color([1, 0, 0])
    # target_vis.paint_uniform_color([0, 1, 0])
    # target_vis.transform(invert_homogeneous(result.transformation.numpy()))
    # o3d.visualization.draw_geometries([target_vis, source_vis])

    if (
        result.fitness < FITNESS_THRESHOLD
        or result.num_iterations >= MAX_ITERATIONS
        or result.inlier_rmse > INLIER_RMSE_THRESHOLD
    ):
        return False, initial_pose

    reg_point_to_plane = result.transformation.numpy()
    return True, invert_homogeneous(reg_point_to_plane)


def infer_poses(rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> dict:
    endpoint = "http://localhost:5000/predict"

    # raw_bytes = open(test_img_path, "rb").read()
    rgb_raw_bytes = cv2.imencode(".jpg", rgb[:, :, ::-1])[1].tobytes()  # encode as 'BGR'
    rgb_encoded_bytes = base64.b64encode(rgb_raw_bytes)

    memfile = io.BytesIO()
    np.save(memfile, depth)
    depth_encoded_bytes = base64.b64encode(memfile.getvalue())

    memfile = io.BytesIO()
    np.save(memfile, intrinsics)
    intrinsics_encoded_bytes = base64.b64encode(memfile.getvalue())

    # POST to endpoint
    response = requests.post(
        endpoint,
        files={
            "rgb": rgb_encoded_bytes,
            "depth": depth_encoded_bytes,
            "intrinsic": intrinsics_encoded_bytes,
        },
    )

    return response.json()

if __name__ == "__main__":
    main()
