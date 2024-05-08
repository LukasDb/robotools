import os
import json

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
import asyncio
import matplotlib.pyplot as plt
import open3d as o3d
import requests
import base64
import io

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


async def async_main(capture: bool, output: Path) -> None:
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene.yaml")))

    # extract names from scene.yaml
    cam: rt.camera.Camera = scene._entities["Realsense_121622061798"]
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

    print("Done")
    scene_mesh = scene_integration.vbg.extract_triangle_mesh()
    print("Mesh extracted. Now processing...")
    scene_mesh.compute_triangle_normals()
    scene_mesh.compute_vertex_normals()
    scene_mesh = scene_mesh.to_legacy()

    scene_pcl = scene_integration.vbg.extract_point_cloud()

    scene_mesh = scene_pcl.to_legacy()  # for testing

    mesh_root_dir = Path("~/data/6IMPOSE/0_meshes/").expanduser().glob("*")
    objs = {}
    for mesh_dir in mesh_root_dir:
        mesh_path = mesh_dir.joinpath(f"{mesh_dir.name}.obj")
        objs[mesh_dir.name] = o3d.io.read_triangle_mesh(str(mesh_path), True)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

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

        for det in detections:
            cls = det["cls"]
            bbox = det["bbox"]
            bbox = [int(x) for x in bbox]
            confidence = float(det["confidence"])

            if confidence < 10.0:
                continue

            cv2.rectangle(rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(
                rgb,
                f"{cls} {confidence:.2f}",
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # ICP refinement of non-inferred objects for each image is not necessary, since we use the scene pcl anyway (view independent)
            affine_matrix = np.array(det["pose"])
            pose = extrinsics @ affine_matrix

            pose = icp_refinement(pose, scene_pcl, objs[cls])

            # since we use ICP with the scene PCL, we do not need to fuse multi-frame poses (we already use a multi-view PCL!)
            obj = objs[cls]
            dist_threshold = np.linalg.norm(obj.get_max_bound() - obj.get_min_bound()) * 0.5
            dists = [
                distance_from_matrices(pose, x, rotation_factor=0.1)
                for x in obj_poses.get(cls, [])
            ]
            if len(dists) == 0 or min(dists) > dist_threshold:
                # "new" object
                print("Adding new object!")
                obj_poses.setdefault(cls, [])
                obj_poses[cls].append(pose)
            else:
                print("Already found the object: ignoring.")

            # all_obj_poses.setdefault(cls, [])
            # all_obj_poses[cls].append(obj_pose)

            affine_matrix = invert_homogeneous(extrinsics) @ pose

            rot = affine_matrix[:3, :3]
            t = affine_matrix[:3, 3]

            rvec = cv2.Rodrigues(rot)
            rvec = rvec[0].flatten()
            tvec = t.flatten()

            # draw frame
            cv2.drawFrameAxes(rgb, intrinsics, np.zeros((4, 1)), rvec, tvec, 0.1, 3)  # type: ignore

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


def icp_refinement(
    initial_pose, scene_pcl: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh
):
    # performs significantly worse
    gpu = o3d.core.Device("CUDA:0")
    o3d_float64 = o3d.core.Dtype.Float32
    source = scene_pcl

    # the scene is only the visible part of the object
    # therefore we need to align the object to the scene and then remove the 'backside'

    target = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.asarray(mesh.vertices), o3d_float64))
    target.point.normals = o3d.core.Tensor(np.asarray(mesh.vertex_normals), o3d_float64)
    target = target.cuda()

    treg = o3d.t.pipelines.registration
    estimation = treg.TransformationEstimationPointToPlane()
    criteria = treg.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=5
    )

    max_correspondence_distance = 0.01
    voxel_size = 0.005  # voxel downsampling!
    result = treg.icp(
        source,
        target,
        max_correspondence_distance,
        invert_homogeneous(initial_pose),
        estimation,
        criteria,
        voxel_size,
    )
    # print(
    #     f"{result.converged}; {result.fitness = :.4f}, {result.inlier_rmse = :.4f};  {result.num_iterations = }"
    # )

    # if not result.converged:
    #     return initial_pose
    reg_point_to_plane = result.transformation.numpy()
    return invert_homogeneous(reg_point_to_plane)


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


@click.command()
@click.option("--capture", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/scene_construction")
def main(capture: bool, output: Path):
    asyncio.run(async_main(capture, output))


if __name__ == "__main__":
    main()
