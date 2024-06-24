import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import click
import asyncio

import robotools as rt
from robotools.camera import Realsense, HandeyeCalibrator, zed
from robotools.trajectory import SphericalTrajectory, TrajectoryExecutor, CartesianTrajectory
from robotools.robot import FanucCRX10iAL

import coloredlogs
import logging

async def async_main(calibrate: bool,capture: bool, output: Path) -> None:
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene_combined.yaml"))) # make sure to have the right calibration file

    # extract names from scene_combined.yaml
    cam_ZED: rt.camera.Camera = scene._entities["ZED-M"]
    bg: rt.utility.BackgroundMonitor = scene._entities["Background Monitor"]
    robot: rt.robot.Robot = scene._entities["crx"]
    
    calibrator = HandeyeCalibrator()
    extrinsic_guess = np.eye(4)
    extrinsic_guess[:3, :3] = R.from_euler("zx", [-90, -5], degrees=True).as_matrix()

    if capture: 
        executor = TrajectoryExecutor()
        #generate trajectory
        trajectory = SphericalTrajectory(
            thetas=np.linspace(45, 315, 8, endpoint=True).tolist(),
            pitchs=[45, 70],
            radius=[0.35],
            center_point=(0.83, 0, -0.16),
            view_jitter=(5, 5, 5),
        )
        trajectory.transform(extrinsic_guess, local=True)
        trajectory.visualize()
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
        async for step in executor.execute(robot, trajectory, cam=cam_ZED):
            frame = cam_ZED.get_frame()
            cv2.imshow("Preview", frame.rgb[::2, ::2, ::-1])
            cv2.waitKey(1)
            robot_pose = await robot.get_pose()
            if robot_pose is not None:
                cv2.imwrite(str(output.joinpath(f"{step:06}.png")), frame.rgb)
                np.savetxt(str(output.joinpath(f"raw{step:06}.txt")), robot_pose)
    if calibrate:
        for img_path in output.glob("*.png"):
            img = cv2.imread(str(img_path))
            robot_pose = np.loadtxt(str(img_path.with_suffix(".txt")))
            vis = calibrator.capture(img, robot_pose)
            #calibrate the exact position
            # ????
            result = calibrator.calibrate(extrinsic_guess=extrinsic_guess)
            robot_pose_refined = result.
            #save the exact postion
            np.savetxt(str(output.joinpath(f"refined{step:06}.txt")), robot_pose_refined)
            cv2.imshow("vis", vis[::2, ::2, ::-1])
            cv2.waitKey(1)
           
            
            calibrator.reset()        


@click.command()
@click.option("--capture", is_flag=True)
@click.option("--calibrate", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/recalibration")
def main(calibrate: bool,capture: bool, output: Path):
    asyncio.run(async_main (calibrate, capture, output))


if __name__ == "__main__":
    main()
