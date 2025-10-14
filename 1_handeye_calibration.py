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
from robotools.camera import Realsense, GeneralCalibrator, zed
from robotools.trajectory import SphericalTrajectory, TrajectoryExecutor, CartesianTrajectory
from robotools.robot import FanucCRX10iAL

import coloredlogs
import logging

coloredlogs.install(level="DEBUG")


async def async_main(capture: bool, output: Path, is_eye_to_hand:bool = False) -> None:
    scene = rt.Scene()

    robot: FanucCRX10iAL = scene.add_entity(FanucCRX10iAL())
    cam: Realsense = scene.add_entity(Realsense.get_available_devices()[0])

    calibrator = GeneralCalibrator(is_eye_to_hand=is_eye_to_hand)
    extrinsic_guess = np.eye(4)
    extrinsic_guess[:3, :3] = R.from_euler("xz", [-18,180], degrees=True).as_matrix()
    print("extrinsic guess:\n", extrinsic_guess)

    # 2) acquire
    if capture:
        executor = TrajectoryExecutor()
        if is_eye_to_hand:

            trajectory = SphericalTrajectory(
                thetas=np.linspace(-45,25, 12, endpoint=True).tolist(),
                pitchs=[55, 65],
                radius=[0.35, 0.45],
                center_point=(0.5, -0.3, -0.2),
                view_jitter=(5,5,5),
            )
        else:
            trajectory = SphericalTrajectory(
                thetas=np.linspace(-30,110, 6, endpoint=True).tolist(),
                pitchs=[55,65, 70],
                radius=[0.35, 0.50],
                center_point=(0.65, -0.3, -0.2),
                view_jitter=(5,5,5),
            )


        trajectory.transform(extrinsic_guess, local=True)

        trajectory.visualize()

        home_pose = await robot.get_pose()
        print("Home position:",home_pose)
        await robot.robotmotion_start(home_pose)

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

    cam.calibration = result.calibration
    yaml.dump(scene.to_config(), open("scene.yaml", "w"))

    if is_eye_to_hand:
        calibrator.visualize_calibration(
            world2markers=None,
            extrinsics=result.calibration.extrinsic_matrix,   # W_C (world->camera)
            intrinsics=result.calibration.intrinsic_matrix,
            dist_coeffs=result.calibration.dist_coeffs,
            ee_to_marker=result.ee_to_marker,                 # EE->Marker
        )
    else:
        calibrator.visualize_calibration(
            world2markers=result.world2markers,               # W->Marker
            extrinsics=result.calibration.extrinsic_matrix,   # EE->Camera (original case)
            intrinsics=result.calibration.intrinsic_matrix,
            dist_coeffs=result.calibration.dist_coeffs,
        )


@click.command()
@click.option("--capture", is_flag=True)
@click.option("--is_eye_in_hand", is_flag=True)

def main(capture: bool, is_eye_in_hand: bool) -> None:

    if is_eye_in_hand:
        output = Path("data_eye_in_hand/calibration")
    else:
        output = Path("data_eye_to_hand/calibration")

    asyncio.run(async_main(capture, output, is_eye_to_hand=not is_eye_in_hand))


if __name__ == "__main__":
    main()
