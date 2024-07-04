import cv2
import tensorflow as tf

from robotools.geometry import invert_homogeneous

for dev in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(dev, True)
import numpy as np
import json
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import click
import asyncio
import matplotlib.pyplot as plt
import open3d as o3d
import time
import re
import os


import simpose as sp
import robotools as rt
from robotools.trajectory import SphericalTrajectory, TrajectoryExecutor, CartesianTrajectory
from robotools.camera import Realsense, HandeyeCalibrator, zed, live_recal



"""
we use
- the calibration from 1_handeye_calibration (needs to be run only a single time)
- the object poses from 3_scene_labelling (needs to be run for every scene)
to create the final dataset with all annotations

In this script:
- we capture data similar to 2_scene_reconstrution, but without the projector and with different lighting/backgrounds etc
- WE USE THE SAME POSES! therefore we can use the GT depth data from 2!
- we take calibration from 1, the depth from 2, the poses from 3 and write them all to the output
    -> should we then even capture ZED w/o projection in 2?


WIP
"""


async def async_main(capture: bool,label:bool, output: Path) -> None:
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene_combined.yaml")))

    # extract names from scene.yaml
    # cam: rt.camera.Camera = scene._entities["Realsense_121622061798"]
    cam: rt.camera.Camera = scene._entities["ZED-M"]

    bg: rt.utility.BackgroundMonitor = scene._entities["Background Monitor"]
    projector: rt.utility.BackgroundMonitor = scene._entities["Projector"]
    robot: rt.robot.Robot = scene._entities["crx"]

    scene_integration = rt.SceneIntegration(use_color=True)
    #scene_integration.load()
    calibrator = HandeyeCalibrator()
    world2marker = bg.get_pose()
    

    print(f"Loaded scene from file: {cam.calibration}")
    trajectory = SphericalTrajectory(
        thetas=np.linspace(30,160 , 6, endpoint=False).tolist(),
        pitchs=[40, 55, 65, 75],
        radius=[0.3],
        center_point=(0.83, 0, -0.05),
        view_jitter=(5, 5, 5),
    )
    trajectory += SphericalTrajectory(
        thetas=np.linspace(200,330 , 6, endpoint=False).tolist(),
        pitchs=[40, 55, 65, 75],
        radius=[0.3],
        center_point=(0.83, 0, -0.05),
        view_jitter=(5, 5, 5),
    )
    trajectory += CartesianTrajectory(
            (180, 0, 90),
            np.linspace(0.83 - 0.1, 0.83 + 0.1, 3).tolist(), 
            np.linspace(-0.3, 0.3, 3).tolist(),
            0.5,
            view_jitter=(5, 5, 5),
        ) 

    if capture:
        executor = TrajectoryExecutor()
        trajectory.visualize()
        input("Please move the window to the background screen and press enter")
        bg.setup_window()
        input("Please move the window to the projector screen and press enter")
        projector.setup_window()
        projector.set_to_black()


        # input("Please move the window to the second screen and press enter")
        # bg.setup_window()
        # bg.set_to_depth_texture()

        # 2) acquire
        captured_data = []
        
           
        async for step in executor.execute(robot, trajectory, cam=cam):
            #recallibrate
            world2robot = await robot.get_pose()
            world2cam_raw = world2robot @ cam.calibration.extrinsic_matrix
            bg.set_to_charuco(
            chessboard_size=calibrator.chessboard_size,
            marker_size=calibrator.marker_size,
            n_markers=calibrator.n_markers,
            charuco_dict=calibrator.aruco_dict,
            )
            time.sleep(2)
            frame = cam.get_frame()
            img = frame.rgb
            retval, world2cam_refined = live_recal.charucoPnP(calibrator, cam, img, world2cam_raw, world2marker)
            if retval:
                diff = world2cam_raw @ invert_homogeneous(world2cam_refined)
                #there could be an error because of this. In case that happens,we need
                if np.max(np.abs(diff)) < 2:
                    bg.set_to_black()
                    time.sleep(2)

                    cam_rot = R.from_matrix(world2cam_refined[:3, :3]).as_quat()
                    cam_pos = world2cam_refined[:3, 3]

                    #get frame
                    
                    frame = cam.get_frame()             
                    """ rgb=frame.rgb
                    rgb = rgb.copy()
                    rgb_R=frame.rgb_R
                    rgb_R = rgb_R.copy
                    depth = frame.depth
                    depth = depth.copy()
                    depth_R = frame.depth_R
                    depth_R = depth_R.copy()
                    """


                    rp = sp.RenderProduct(
                        rgb=frame.rgb,
                        rgb_R= frame.rgb_R,
                        depth=frame.depth,
                        depth_R=frame.depth_R,
                        intrinsics=cam.calibration.intrinsic_matrix,
                        cam_position=cam_pos,
                        cam_quat_xyzw=cam_rot,
                    )  
                    captured_data.append(rp)
                else:
                    print("the offset between the two position values was too big")
        output_raw = Path("data/labelled_raw")
        
        output_raw.mkdir(parents=True, exist_ok=True)
        writer_params = sp.writers.WriterConfig(
            output_dir=output_raw,
            overwrite=True,
            start_index=0,
            end_index=len(captured_data) - 1,
        )
        with sp.writers.TFRecordWriter(writer_params, None) as writer:
            for step in range(len(captured_data)):
                writer.write_data(step, render_product=captured_data[step])
        print("Captured and stored new set of raw images")
        
        
    if label:
        mask_gen = rt.MaskGenerator()

        with open("labelled_objects.json", "r") as f:
            obj_poses = json.load(f)

        obj_poses = {
            k: list([np.array(pose) for pose in pose_list]) for k, pose_list in obj_poses.items()
        }

        for cls, pose_list in obj_poses.items():
            print(f"Found {len(pose_list)} {cls}s")



        # 3) reconstruct
        files = os.listdir(output_raw.joinpath("data"))
        tfrecord_file = None
        for filename in files:
            if filename.endswith(".tfrecord"):
                tfrecord_file = filename
        match = re.search(r'_(\d+)\.tfrecord$', tfrecord_file)
        total = int(match.group(1))

        tfds = sp.data.TFRecordDataset
        dataset = tfds.get(
            output.with_name(output.name + "_raw"),
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

        mesh_root_dir = Path("/home/aismart/Desktop/robotools/0_meshes").expanduser().glob("*")
        objs = {}
        for mesh_dir in mesh_root_dir:
            mesh_path = mesh_dir.joinpath(f"{mesh_dir.name}.obj")
            objs[mesh_dir.name] = o3d.io.read_triangle_mesh(str(mesh_path), True)

        output.mkdir(parents=True, exist_ok=True)
        writer_params = sp.writers.WriterConfig(
            output_dir=output,
            overwrite=True,
            start_index=0,
            end_index=total,  # AS LONG AS THE RECORDED DATASET
        )

        


        with sp.writers.TFRecordWriter(writer_params, None) as writer:
            for step, data in tqdm(dataset, total=total, desc="Analyzing..."):
                cam_pose = np.eye(4)
                cam_pose[:3, 3] = data[tfds.CAM_LOCATION].numpy()
                cam_pose[:3, :3] = R.from_quat(data[tfds.CAM_ROTATION].numpy()).as_matrix()
                h, w = data[tfds.DEPTH].shape[:2]

                rgb = data[tfds.RGB].numpy()
                rgb_annotated = rgb.copy()

                try:
                    rgb_R = data[tfds.RGB_R].numpy()
                except KeyError:
                    rgb_R = None

                intrinsics = data[tfds.CAM_MATRIX].numpy().astype(np.float64)

                depth_reconst = scene_integration.render_depth(
                    w,
                    h,
                    data[tfds.CAM_MATRIX].numpy().astype(np.float64),
                    invert_homogeneous(cam_pose).astype(np.float64),
                    depth_scale=1.0,
                )

                depth_raw = data[tfds.DEPTH].numpy()

                depth_rendered = np.ones_like(depth_raw) * 1000.0

                occlusion_mask = np.zeros_like(depth_raw)

                object_labels = []
                img_obj_id = 1
                for cls, pose_list in obj_poses.items():
                    obj = objs[cls]
                    for obj_pose in pose_list:
                        unoccluded_mask = mask_gen.render_object_mask(
                            obj, obj_pose, cam_pose, w, h, intrinsics
                        )
                        bbox_obj = rt.BBox.get_from_mask(unoccluded_mask)  # x1y1x2y2
                        occluded_mask, depth_from_pose = mask_gen.calculate_occluded_mask(
                            unoccluded_mask=unoccluded_mask,
                            obj=obj,
                            obj_pose=obj_pose,
                            cam_pose=cam_pose,
                            width=w,
                            height=h,
                            intrinsics=intrinsics,
                            depth=depth_reconst,
                            occlusion_threshold=0.01,
                        )

                        depth_rendered = np.minimum(depth_rendered, depth_from_pose)
                        occlusion_mask = np.maximum(occlusion_mask, occluded_mask * img_obj_id)

                        bbox = rt.BBox.get_from_mask(occluded_mask)  # x1y1x2y2
                        rgb_annotated = bbox.draw(rgb_annotated, color=(255, 0, 0))

                        cam2obj = invert_homogeneous(cam_pose) @ obj_pose
                        rot = cam2obj[:3, :3]
                        t = cam2obj[:3, 3]
                        rvec = cv2.Rodrigues(rot)
                        rvec = rvec[0].flatten()
                        tvec = t.flatten()
                        cv2.drawFrameAxes(rgb_annotated, intrinsics, np.zeros((4, 1)), rvec, tvec, 0.1, 3)  # type: ignore

                        obj_pos = obj_pose[:3, 3]
                        obj_rot = R.from_matrix(obj_pose[:3, :3]).as_quat(canonical=True)

                        px_count_visib = np.count_nonzero(occluded_mask)
                        px_count_valid = np.count_nonzero(depth_reconst[occluded_mask == 1])
                        px_count_all = np.count_nonzero(unoccluded_mask == 1)
                        visible_fraction = 0.0 if px_count_all == 0 else px_count_visib / px_count_all

                        object_labels.append(
                            sp.ObjectAnnotation(
                                cls=cls,
                                object_id=img_obj_id,
                                position=tuple(obj_pos),
                                quat_xyzw=tuple(obj_rot),
                                bbox_visib=bbox.astuple(),
                                bbox_obj=bbox_obj.astuple(),
                                px_count_visib=px_count_visib,
                                px_count_valid=px_count_valid,
                                px_count_all=px_count_all,
                                visib_fract=visible_fraction,
                            )
                        )
                        img_obj_id += 1
                depth_rendered[depth_rendered > 100.0] = 0.0

                cam_pos = cam_pose[:3, 3]
                cam_rot = R.from_matrix(cam_pose[:3, :3]).as_quat(True)
                datapoint = sp.RenderProduct(
                    rgb=rgb,
                    rgb_R=rgb_R,
                    depth=depth_raw,
                    depth_GT=depth_reconst,
                    # depth_GT_R=depth_rendered,
                    mask=occlusion_mask,
                    cam_position=cam_pos,
                    cam_quat_xyzw=cam_rot,
                    intrinsics=intrinsics,
                    objs=object_labels,
                )
                writer.write_data(step, render_product=datapoint)
        print("saved labelled dataset")
    return


def color_depth(depth: np.ndarray, d_min=0.0, d_max=1.0) -> np.ndarray:
    return o3d.t.geometry.Image(depth).colorize_depth(1.0, d_min, d_max).as_tensor().cpu().numpy()


@click.command()
@click.option("--capture", is_flag=True)
@click.option("--label", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/labelled")
def main(capture: bool,label:bool, output: Path):
    asyncio.run(async_main(capture, label, output))


if __name__ == "__main__":
    main()