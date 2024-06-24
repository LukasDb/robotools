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
import copy
from collections import OrderedDict
import io
import json
import trimesh
import open3d.visualization.gui as gui


import simpose as sp
import robotools as rt
from robotools.trajectory import SphericalTrajectory, TrajectoryExecutor

IOU_THRESHOLD = 0.7


@click.command()
@click.option("--capture", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/scene_construction")
def main(capture: bool, output: Path) -> None:
    print(
        """-- 3a Scene Labelling --
This script is used to label objects in the scene. Using the voxel grid of the reconstructed scene of script #2, here you have to manually annotated the pose of the objects.
You will be asked to pick points from the object mesh and from the scene. This will act as correspondences to estimate the initial pose of the object. ICP then refines the pose.
1) Choose an object mesh and pick 3 points. Pick points that are easy to remember and to pick (like the front, the back and the rightmost point).
2) Pick the corresponding points in the scene.
3) View the refined pose and either accept or reject it.
4) Add more of the same objects, choose different objects or exit the program."""
    )
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene_combined.yaml")))

    # extract names from scene.yaml
    bg: rt.utility.BackgroundMonitor = scene._entities["Background Monitor"]
    bg.disable()
    robot: rt.robot.Robot = scene._entities["crx"]

    scene_integration = rt.SceneIntegration(use_color=True)
    scene_integration.load()

    print("Loaded.")

    print("Extracting scene...")
    scene_pcd_t = scene_integration.vbg.extract_point_cloud()
    scene_pcd_t = scene_pcd_t.voxel_down_sample(0.002)
    scene_pcd = scene_pcd_t.to_legacy()
    
    """ #alternative: using mesh
    scene_mesh_t = scene_integration.vbg.cpu().extract_triangle_mesh()
    print("Sampling points...")
    scene_pcd = scene_mesh_t.to_legacy().sample_points_poisson_disk(200_000)
    print("Copying to GPU...")
    scene_pcd_t = o3d.t.geometry.PointCloud.from_legacy(scene_pcd).cuda() """

    mesh_root_dir = Path("/home/aismart/Desktop/robotools/0_meshes").expanduser().glob("*")
    objs = OrderedDict()
    for mesh_dir in mesh_root_dir:
        mesh_path = mesh_dir.joinpath(f"{mesh_dir.name}.obj")
        objs[mesh_dir.name] = o3d.io.read_triangle_mesh(str(mesh_path), True)

    obj_poses: dict[str, list[np.ndarray]] = {}

    chosen_mesh_name, mesh_pcd, mesh_picked_points = define_target(objs)
    chosen_mesh = objs[chosen_mesh_name]

    while True:
        # pick points from two point clouds and builds correspondences
        source_points = pick_points(scene_pcd, objs, obj_poses)
        T_registered = register_via_correspondences(
            scene_pcd, mesh_pcd, source_points, mesh_picked_points
        )
        is_valid_detection, T_refined = icp_refinement(T_registered, scene_pcd_t, chosen_mesh)
        print("Pose refined. Verify results...")
        print("Press Q to exit the visualizer")

        vis = []
        scene_pcd_temp = copy.deepcopy(scene_pcd)
        vis.append(scene_pcd_temp)
        mesh_temp = copy.deepcopy(chosen_mesh)
        mesh_temp.transform(T_refined)
        vis.append(mesh_temp)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        axis.transform(T_refined)
        vis.append(axis)
        o3d.visualization.draw_geometries(vis)

        if input("Accept the pose? (y/n):") == "y":
            print("Pose accepted")
            obj_poses[chosen_mesh_name] = obj_poses.get(chosen_mesh_name, []) + [T_refined]
        else:
            print("Pose rejected")
        print("")

        print("All registered objects (Press 'Q' to continue):")
        draw_registration_results(scene_pcd, objs, obj_poses)

        option = handle_user_input(obj_poses)
        if option == "pick_object":
            chosen_mesh_name, mesh_pcd, mesh_picked_points = define_target(objs)
            chosen_mesh = objs[chosen_mesh_name]
        elif option == "quit":
            break


def handle_user_input(obj_poses) -> str:
    print("1: Add the same object again")
    print("2: Choose a different object or pick different points")
    print("3: Exit and write to file")
    print("4: Exit without writing to file")
    option = input("Choose an option [1-5]:")

    if option == "1":
        pass
    elif option == "2":
        return "pick_object"
    elif option == "3":
        with open("labelled_objects.json", "w") as f:
            json.dump(obj_poses, f, default=lambda x: x.tolist(), indent=2)
        return "quit"
    elif option == "4":
        if input("Are you sure? (y/n):") == "y":
            return "quit"
        else:
            handle_user_input(obj_poses)
    else:
        print("Invalid option. Please choose again.")
        handle_user_input(obj_poses)


def define_target(
    objs: OrderedDict[str, o3d.geometry.TriangleMesh]
) -> tuple[str, o3d.geometry.PointCloud, list]:
    print(f"Available meshes:")
    print("\n".join(f"{i}: {k}" for i, k in enumerate(objs.keys())))
    chosen_index = input(f"Choose mesh ([0-{len(objs)-1}]):")

    try:
        mesh_name = list(objs.keys())[int(chosen_index)]
    except Exception:
        print("Invalid index. Please choose again.")
        return define_target(objs)
    print(f"Chose {mesh_name}.\n")
    chosen_mesh = objs[mesh_name]
    mesh_pcd = chosen_mesh.sample_points_poisson_disk(20_000)
    mesh_picked_points = pick_points(mesh_pcd)
    return mesh_name, mesh_pcd, mesh_picked_points


def draw_registration_results(
    scene_pcd, objs: dict[str, o3d.geometry.TriangleMesh], obj_poses: dict[str, list[np.ndarray]]
):
    vis = []
    scene_pcd_temp = copy.deepcopy(scene_pcd)
    vis.append(scene_pcd_temp)

    for mesh_name, mesh in objs.items():
        for pose in obj_poses.get(mesh_name, []):
            mesh_temp = copy.deepcopy(mesh)
            mesh_temp.transform(pose)
            vis.append(mesh_temp)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            axis.transform(pose)
            vis.append(axis)

    o3d.visualization.draw_geometries(vis)


def pick_points(
    geometry,
    objs: dict[str, o3d.geometry.TriangleMesh] = {},
    obj_poses: dict[str, list[np.ndarray]] = {},
):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(geometry)

    for mesh_name, mesh in objs.items():
        for i, pose in enumerate(obj_poses.get(mesh_name, [])):
            mesh_temp = copy.deepcopy(mesh)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            # this does not work ith the deprecated VisualizerWithEditing
            mesh_temp.transform(pose)
            axis.transform(pose)
            vis.add_geometry(mesh_temp)
            vis.add_geometry(axis)

    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


def register_via_correspondences(source, target, source_points, target_points):
    corr = np.zeros((len(source_points), 2))
    corr[:, 0] = source_points
    corr[:, 1] = target_points
    # estimate rough transformation using correspondences
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))
    return invert_homogeneous(trans_init)


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

    # params
    MAX_ITERATIONS = 100
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

    print(f"{result.fitness = :.4f}, {result.inlier_rmse = :.4f};  {result.num_iterations = }")

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


if __name__ == "__main__":
    main()
