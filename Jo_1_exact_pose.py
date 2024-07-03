import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from cv2 import aruco
import open3d as o3d

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import click
import asyncio
import time

import robotools as rt
from robotools.camera import Realsense, HandeyeCalibrator, zed, live_recal
from robotools.trajectory import SphericalTrajectory, TrajectoryExecutor, CartesianTrajectory
from robotools.robot import FanucCRX10iAL
from robotools.geometry import (
    get_6d_vector_from_affine_matrix,
    invert_homogeneous,
    get_affine_matrix_from_6d_vector,
)

import coloredlogs
import logging

import threading

async def async_main(display: bool,capture: bool,save:bool, output: Path) -> None:
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene_combined.yaml"))) # make sure to have the right calibration file

    # extract names from scene_combined.yaml
    cam: rt.camera.Camera = scene._entities["ZED-M"]
    #cam: rt.camera.Camera = scene._entities["Realsense_121622061798"]
    bg: rt.utility.BackgroundMonitor = scene._entities["Background Monitor"]
    robot: rt.robot.Robot = scene._entities["crx"]
    
    calibrator = HandeyeCalibrator()
   
    #initalize arrays
    w2c_array_raw = []
    w2c_array_refined = []
    image_array = []
    #get the staic matrices
    world2marker = bg.get_pose()
    robot2cam = cam.calibration.extrinsic_matrix

    executor = TrajectoryExecutor()
    #generate trajectory
    trajectory = SphericalTrajectory(
        thetas=np.linspace(30,160 , 8, endpoint=False).tolist(),
        pitchs=[40, 55, 65, 75],
        radius=[0.3],
        center_point=(0.83, 0, -0.16),
        view_jitter=(5, 5, 5),
    )
    trajectory += SphericalTrajectory(
        thetas=np.linspace(45, 315, 8, endpoint=True).tolist(),
        pitchs=[45, 70],
        radius=[0.35],
        center_point=(0.83, 0, -0.16),
        view_jitter=(5, 5, 5),
    )
    trajectory += SphericalTrajectory(
        thetas=np.linspace(60, 300, 8, endpoint=True).tolist(),
        pitchs=[60],
        radius=[0.45],
        center_point=(0.83, 0, -0.16),
        view_jitter=(5, 5, 5),
    )
    
    trajectory += CartesianTrajectory(
            (180, 0, 90),
            np.linspace(0.83 - 0.1, 0.83 + 0.1, 4).tolist(), 
            np.linspace(-0.25, 0.25, 4).tolist(),
            0.5,
            view_jitter=(5, 5, 5),
        )
    if display:
        trajectory.visualize()

    #create arrays to store the respective positions
    num_steps = len(trajectory)
    #w2c_array_raw= np.zeros((num_steps, 4, 4), dtype='float64')
    #w2c_array_refined = np.zeros((num_steps, 4, 4), dtype='float64')
    
    #setup the charuco window
    input("Please move the window to the second screen and press enter")
    bg.setup_window()
    bg.set_to_charuco(
        chessboard_size=calibrator.chessboard_size,
        marker_size=calibrator.marker_size,
        n_markers=calibrator.n_markers,
        charuco_dict=calibrator.aruco_dict,
        )
   
   

    #drive the robot along the trajectory, take a picture at each positon, save both the robot and the position
    output.mkdir(parents=True, exist_ok=True)
    async for step in executor.execute(robot, trajectory, cam=cam):
        time.sleep(2)
        frame = cam.get_frame()
        img = frame.rgb
        img = img.copy()
        cv2.imshow("Preview", img[::2, ::2, ::-1])
        cv2.waitKey(1)
        world2robot = await robot.get_pose()
        world2cam_raw = world2robot @ robot2cam
        #if world2robot is not None:
            #cv2.imwrite(str(output.joinpath(f"{step:06}.png")), frame.rgb)     
            #np.savetxt(str(output.joinpath(f"cam_raw{step:06}.txt")), world2cam_raw)
        #readimage
        retval, world2cam_refined = live_recal.charucoPnP(calibrator, cam, img, world2cam_raw, world2marker)
        if retval:

            #save estimated postion
            #np.savetxt(str(output.joinpath(f"cam_refined{step:06}.txt")), world2cam_refined)
            if display:
                display_camera_poses([world2marker, world2cam_raw,world2cam_refined],trajectory)
                
            #append both raw and refined camera poses to arrays
            w2c_array_refined.append(world2cam_refined)
            w2c_array_raw.append(world2cam_raw)
            image_array.append(img)
            #cv2.imwrite(str(output.joinpath(f"Cal of pic{step:03}.png")), img)
        
        print(step)
    
    #save agglomerated arrays for later usage
    np.save(str(output.joinpath("world2cam_array_raw.npy")),w2c_array_raw)
    np.save(str(output.joinpath("world2cam_array_refined.npy")),w2c_array_refined)
    np.save(str(output.joinpath("frames.npy")),image_array)
    print("Done")
    



def display_camera_poses(transformations,trajectory):
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis = [origin]
    i = 1
    for pose in transformations:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size= 0.1)
        frame.transform(pose)
        vis.append(frame)
        
        i += 1
    for pose in trajectory:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        frame.transform(pose)
        vis.append(frame)

    o3d.visualization.draw_geometries(vis)



@click.command()
@click.option("--capture", is_flag=True)
@click.option("--display", is_flag=True)
@click.option("--save", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/recalibration")
def main(display: bool,capture: bool,save: bool, output: Path):
    asyncio.run(async_main (display, capture,save, output))


if __name__ == "__main__":
    main()
