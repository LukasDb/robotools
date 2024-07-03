import cv2
import tensorflow as tf

from robotools.geometry import invert_homogeneous

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
import time
import re
import os

import simpose as sp
import robotools as rt
from robotools.trajectory import SphericalTrajectory, TrajectoryExecutor, CartesianTrajectory
from robotools.camera import Realsense, HandeyeCalibrator, zed, live_recal

async def async_main(capture: bool, output: Path) -> None:
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene_combined.yaml"))) # make sure to have the right calibration file

    # extract names from scene.yaml
    cam_RS: rt.camera.Camera = scene._entities["Realsense_121622061798"] 
    cam_ZED: rt.camera.Camera = scene._entities["ZED-M"]
    
    bg: rt.utility.BackgroundMonitor = scene._entities["Background Monitor"]
    projector: rt.utility.BackgroundMonitor = scene._entities["Projector"]
    robot: rt.robot.Robot = scene._entities["crx"]

    calibrator = HandeyeCalibrator()
    world2marker = bg.get_pose()

    

    trajectory = SphericalTrajectory(
        thetas=np.linspace(45, 315, 8, endpoint=True).tolist(),
        pitchs=[45, 70],
        radius=[0.35],
        center_point=(0.83, 0, 0),
        #view_jitter=(5, 5, 5),
    )
    trajectory += SphericalTrajectory(
        thetas=np.linspace(60, 300, 8, endpoint=True).tolist(),
        pitchs=[60],
        radius=[0.45],
        center_point=(0.83, 0, 0),
        #view_jitter=(5, 5, 5),
    )
    trajectory += CartesianTrajectory(
            (180, 0, 90),
            np.linspace(0.83 - 0.1, 0.83 + 0.1, 3).tolist(), 
            np.linspace(-0.3, 0.3, 3).tolist(),
            0.5,
            #view_jitter=(5, 5, 5),
        )
    trajectory += CartesianTrajectory(
            (180, 30, 90),
            0.83, 
            np.linspace(-0.3, 0.3, 5).tolist(),
            0.25,
            #view_jitter=(5, 5, 5),
        ) 
    trajectory += CartesianTrajectory(
            (180, 330, 90),
            0.93, 
            np.linspace(-0.3, 0.3, 5).tolist(),
            0.25,
            #view_jitter=(5, 5, 5),
        )   

    
    if capture:
        executor = TrajectoryExecutor()
        #trajectory.visualize()
        
        
        input("Please move the window to the background screen and press enter")
        bg.setup_window()
        input("Please move the window to the projector screen and press enter")
        projector.setup_window()

          # 2) acquire
        
        projector.set_to_black()
        bg.set_to_black()       
        
        captured_data = []              #buffer for captured data
        
        print("Starting capturing with Realsense...")
        input("Please turn on the ceiling lights and press enter")
        async for step in executor.execute(robot, trajectory, cam=cam_RS):
            #recallibrate
            world2robot = await robot.get_pose()
            world2cam_raw = world2robot @ cam_RS.calibration.extrinsic_matrix
            bg.set_to_charuco(
                chessboard_size=calibrator.chessboard_size,
                marker_size=calibrator.marker_size,
                n_markers=calibrator.n_markers,
                charuco_dict=calibrator.aruco_dict,
            )
            time.sleep(2)
            frame = cam_RS.get_frame()
            img = frame.rgb
            img =img.copy()
            cv2.imshow("Preview", img[::2, ::2, ::-1])
            cv2.waitKey(1)
            retval, world2cam_refined = live_recal.charucoPnP(calibrator, cam_RS, img, world2cam_raw, world2marker)
            if retval:
                diff = world2cam_raw @ invert_homogeneous(world2cam_refined)
                
                if np.max(np.abs(diff)) < 2:
                    bg.set_to_black()
                    

                    cam_rot = R.from_matrix(world2cam_refined[:3, :3]).as_quat()
                    cam_pos = world2cam_refined[:3, 3]

                    #get frame
                    time.sleep(2)
                    frame = cam_RS.get_frame()
                    rgb=frame.rgb
                    rgb = rgb.copy()
                    rgb_R = frame.rgb_R
                    #rgb_R = rgb_R.copy
                    depth = frame.depth
                    depth = depth.copy()
                    depth_R = frame.depth_R
                    #depth_R = depth_R.copy()



                    rp = sp.RenderProduct(
                        rgb=rgb,
                        rgb_R= None,
                        depth=depth,
                        depth_R=None,
                        intrinsics=cam_RS.calibration.intrinsic_matrix,
                        cam_position=cam_pos,
                        cam_quat_xyzw=cam_rot,
                    )
                    captured_data.append(rp)  #dynamically store captured data
                    
                else:
                    print("the offset between the two position values was too big")
       
             
        
       
        print("Starting capturing with Realsense...")
        input("Please turn OFF the ceiling lights and press enter")
        async for step in executor.execute(robot, trajectory, cam=cam_RS):
            #recallibrate
            world2robot = await robot.get_pose()
            world2cam_raw = world2robot @ cam_RS.calibration.extrinsic_matrix
            projector.set_to_black()
            bg.set_to_charuco(
                chessboard_size=calibrator.chessboard_size,
                marker_size=calibrator.marker_size,
                n_markers=calibrator.n_markers,
                charuco_dict=calibrator.aruco_dict,
            )
            time.sleep(2)
            frame = cam_RS.get_frame()
            img = frame.rgb
            retval, world2cam_refined = live_recal.charucoPnP(calibrator, cam_RS, img, world2cam_raw, world2marker)
            if retval:
                diff = world2cam_raw @ invert_homogeneous(world2cam_refined)
                #there could be an error because of this. In case that happens,we need
                if np.max(np.abs(diff)) < 2:
                    bg.set_to_black()
                    projector.set_to_depth_texture() 

                    cam_rot = R.from_matrix(world2cam_refined[:3, :3]).as_quat()
                    cam_pos = world2cam_refined[:3, 3]

                    #get frame, the copying is to prevent a RS bug
                    time.sleep(2)                    
                    frame = cam_RS.get_frame()             
                    rgb=frame.rgb
                    rgb = rgb.copy()
                    rgb_R=frame.rgb_R
                    #rgb_R = rgb_R.copy
                    depth = frame.depth
                    depth = depth.copy()
                    depth_R = frame.depth_R
                    #depth_R = depth_R.copy()



                    rp = sp.RenderProduct(
                        rgb=rgb,
                        rgb_R= None,
                        depth=depth,
                        depth_R=None,
                        intrinsics=cam_RS.calibration.intrinsic_matrix,
                        cam_position=cam_pos,
                        cam_quat_xyzw=cam_rot,
                    )
                    captured_data.append(rp)
                else:
                    print("the offset between the two position values was too big")


        
        #projector.set_to_depth_texture()
       
        print("Starting capturing with ZED and Pattern...")
        input("Make sure the ceiling lights are turned OFF and press enter")
        async for step in executor.execute(robot, trajectory, cam=cam_ZED):
            #recallibrate
            world2robot = await robot.get_pose()
            world2cam_raw = world2robot @ cam_ZED.calibration.extrinsic_matrix
            projector.set_to_black()
            bg.set_to_charuco(
                chessboard_size=calibrator.chessboard_size,
                marker_size=calibrator.marker_size,
                n_markers=calibrator.n_markers,
                charuco_dict=calibrator.aruco_dict,
            )
            time.sleep(2)
            frame = cam_ZED.get_frame()
            img = frame.rgb
            retval, world2cam_refined = live_recal.charucoPnP(calibrator, cam_ZED, img, world2cam_raw, world2marker)
            if retval:
                diff = world2cam_raw @ invert_homogeneous(world2cam_refined)
                #there could be an error because of this. In case that happens,we need
                if np.max(np.abs(diff)) < 2:
                    projector.set_to_depth_texture()
                    bg.set_to_black()
                    

                    cam_rot = R.from_matrix(world2cam_refined[:3, :3]).as_quat()
                    cam_pos = world2cam_refined[:3, 3]

                    #get frame
                    time.sleep(2)
                    frame = cam_ZED.get_frame()             
                    rgb=frame.rgb
                    rgb = rgb.copy()
                    rgb_R=frame.rgb_R
                    rgb_R = rgb_R.copy
                    depth = frame.depth
                    depth = depth.copy()
                    depth_R = frame.depth_R
                    depth_R = depth_R.copy()

                    rp = sp.RenderProduct(
                        rgb=rgb,
                        rgb_R= None,
                        depth=depth,
                        depth_R=None,
                        intrinsics=cam_ZED.calibration.intrinsic_matrix,
                        cam_position=cam_pos,
                        cam_quat_xyzw=cam_rot,
                    
                    )  
                    captured_data.append(rp)
                else:
                    print("the offset between the two position values was too big")
        
        output.mkdir(parents=True, exist_ok=True)
        writer_params = sp.writers.WriterConfig(
            output_dir=output,
            overwrite=True,
            start_index=0,
            end_index=len(captured_data) - 1,
        )
        with sp.writers.TFRecordWriter(writer_params, None) as writer:
            for step in range(len(captured_data)):
                writer.write_data(step, render_product=captured_data[step])
        #projector.set_to_black()
      
    # 3) reconstruct
    # output = Path("~/data/real/occluded_cpsduck_realsense_121622061798").expanduser()
    # output = Path("~/data/real/single_cpsduck_realsense_121622061798").expanduser()

    tfds = sp.data.TFRecordDataset
    output = output 
    dataset= tfds.get(
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
   


    # only select every nth image (for demo data)
    # the dataset contains always secquences of 5 images with only backdrop changing
    # nth = 5
    # dataset = dataset.filter(lambda i, _: tf.math.floormod(i, nth) == 0)

    scene_integration = rt.SceneIntegration(use_color=True)
    
    if True: 
        

        files = os.listdir(output.joinpath("data"))
        tfrecord_file = None
        for filename in files:
            if filename.endswith(".tfrecord"):
                tfrecord_file = filename
        match = re.search(r'_(\d+)\.tfrecord$', tfrecord_file)
        total = int(match.group(1))

        for i, data in tqdm(dataset, total=total, desc="Reconstructing..."):
            t = data[tfds.CAM_LOCATION]
            r = R.from_quat(data[tfds.CAM_ROTATION].numpy())  # xyzw quaternion
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = r.as_matrix()
            extrinsic[:3, 3] = t.numpy()

            scene_integration.integrate(
                data[tfds.RGB].numpy(),
                data[tfds.DEPTH].numpy(),
                data[tfds.CAM_MATRIX].numpy(),
                extrinsic,
            )
        
              
        scene_integration.save(output.parent.parent.joinpath("vbg.npz"))
    else:
        scene_integration.load()

    print("Done_1")
    mesh = scene_integration.vbg.extract_triangle_mesh()
    print("Mesh extracted. Now processing...")
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh = mesh.to_legacy()
    """ obj = o3d.io.read_triangle_mesh(
        str(Path("~/data/6IMPOSE/0_meshes/cpsduck/cpsduck.obj").expanduser()), True
    )
    obj = obj.translate((0.68, 0.16, -0.16))
    """
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    o3d.visualization.draw_geometries([mesh, origin])

    n_prev = 5
    fig, axs = plt.subplots(n_prev, 3)
    mean_diffs = []
    outlier_ratios = []
    for step, data in tqdm(dataset, total=len(captured_data), desc="Analyzing..."):
        extrinsics = np.eye(4)
        extrinsics[:3, 3] = data[tfds.CAM_LOCATION].numpy()
        extrinsics[:3, :3] = R.from_quat(data[tfds.CAM_ROTATION].numpy()).as_matrix()
        extrinsics = invert_homogeneous(extrinsics)
        h, w = data[tfds.DEPTH].shape[:2]

        depth_reconst = scene_integration.render_depth(
            w, h, data[tfds.CAM_MATRIX].numpy().astype(np.float64), extrinsics.astype(np.float64)
        )
        depth_raw = data[tfds.DEPTH].numpy() * 1000

        # simple hole filler
        depth_fused = np.copy(depth_reconst)
        depth_fused[depth_reconst < 100] = depth_raw[depth_reconst < 100]  # is in mm!

        diff = np.abs(depth_reconst - depth_raw)
        threshold = 10
        mean_diff = np.mean(diff[diff < threshold])
        mean_diffs.append(mean_diff)
        num_outliers = np.count_nonzero(diff > threshold)
        outlier_ratios.append(num_outliers / diff.size)

        if step >= n_prev:
            continue

        colorized_depth = o3d.t.geometry.Image(
            depth_reconst
        )  # .colorize_depth(1000.0, d_min, d_max)
        axs[step, 0].imshow(colorized_depth.as_tensor().cpu().numpy())
        axs[step, 0].set_title("depth (Reconstructed)")

        colorized_depth_raw = o3d.t.geometry.Image(
            depth_raw
        )  # .colorize_depth(1000.0, d_min, d_max)
        axs[step, 1].imshow(colorized_depth_raw.as_tensor().cpu().numpy())
        axs[step, 1].set_title("depth (Realsense)")

        colorized_depth_fused = o3d.t.geometry.Image(
            depth_fused
        )  # .colorize_depth(            1000.0, d_min, d_max        )
        axs[step, 2].imshow(colorized_depth_fused.as_tensor().cpu().numpy())
        axs[step, 2].set_title("depth (fused)")

        continue

        print(f"Mean diff: {mean_diff}")
        colorized_diff = o3d.t.geometry.Image(diff).colorize_depth(1000.0, 0, 0.1)
        axs[2].imshow(colorized_diff.as_tensor().cpu().numpy())
        axs[2].set_title("Difference")
        plt.show(block=False)
        input()
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
    print(f"Mean mean diff: {np.mean(mean_diffs)}")
    print(f"Mean num outliers: {np.mean(outlier_ratios)*100:.2f}%")
    plt.show()


@click.command()
@click.option("--capture", is_flag=True)

@click.option("--output", type=click.Path(path_type=Path), default="data/scene_construction")
def main(capture: bool,output: Path):
    asyncio.run(async_main(capture, output))


if __name__ == "__main__":
    main()
