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

coloredlogs.install(level="DEBUG")


async def async_main(capture: bool, output: Path) -> None:
    scene = rt.Scene()

    robot: FanucCRX10iAL = scene.add_entity(FanucCRX10iAL())
    #
    # cam: Realsense = scene.add_entity(Realsense.get_available_devices()[0])
    cam: zed = scene.add_entity(zed.ZedCamera.get_available_devices()[0])
    bg = scene.add_entity(rt.utility.BackgroundMonitor())

    calibrator = HandeyeCalibrator()
    extrinsic_guess = np.eye(4)
    extrinsic_guess[:3, :3] = R.from_euler("zx", [-90, -5], degrees=True).as_matrix()
    print("extrinsic guess:\n", extrinsic_guess)
    # 2) acquire
    if capture:
        executor = TrajectoryExecutor()
        trajectory = SphericalTrajectory(
            thetas=np.linspace(60, 300, 8, endpoint=True).tolist(),
            pitchs=[50, 70],
            radius=[0.45],
            center_point=(0.83, 0, -0.16),
            view_jitter=(5, 5, 5),
        )
        trajectory += CartesianTrajectory(
            (180, 0, 90),
            np.linspace(0.83 - 0.1, 0.83 + 0.1, 3).tolist(), 
            np.linspace(-0.3, 0.3, 6).tolist(),
            0.5,
            view_jitter=(5, 5, 5),
        )

        trajectory.transform(extrinsic_guess, local=True)

        #trajectory.visualize()

        # wait for user to move window to second screen
        input("Please move the window to the second screen and press enter")

        bg.setup_window()
        bg.set_to_charuco(
            chessboard_size=calibrator.chessboard_size,
            marker_size=calibrator.marker_size,
            n_markers=calibrator.n_markers,
            charuco_dict=calibrator.aruco_dict,
        )

        output.mkdir(parents=True, exist_ok=True)
        async for step in executor.execute(robot, trajectory, cam=cam):
            frame = cam.get_frame()
            cv2.imshow("Preview", frame.rgb[::2, ::2, ::-1])
            cv2.waitKey(1)
            robot_pose = await robot.get_pose()
            if robot_pose is not None:
                cv2.imwrite(str(output.joinpath(f"{step:06}.png")), frame.rgb)
                np.savetxt(str(output.joinpath(f"{step:06}.txt")), robot_pose)

    for img_path in output.glob("*.png"):
        img = cv2.imread(str(img_path))
        robot_pose = np.loadtxt(str(img_path.with_suffix(".txt")))
        vis = calibrator.capture(img, robot_pose)
        cv2.imshow("vis", vis[::2, ::2, ::-1])
        cv2.waitKey(1)

    result = calibrator.calibrate(extrinsic_guess=extrinsic_guess)
    bg.set_pose(result.world2markers)
    cam.calibration = result.calibration
    yaml.dump(scene.to_config(), open("test.yaml", "w"))

    calibrator.visualize_calibration(
        world2markers=result.world2markers,
        extrinsics=result.calibration.extrinsic_matrix,
        intrinsics=result.calibration.intrinsic_matrix,
        dist_coeffs=result.calibration.dist_coeffs,
    )


@click.command()
@click.option("--capture", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/calibration")
def main(capture: bool, output: Path):
    asyncio.run(async_main(capture, output))


if __name__ == "__main__":
    main()
