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
    scene.from_config(yaml.safe_load(open("scene_crx.yaml"))) # make sure to have the right calibration file

    # extract names from scene_combined.yaml

    robot: rt.robot.Robot = scene._entities["crx"]
    calibrator = HandeyeCalibrator()
    executor = TrajectoryExecutor()
    #generate trajectory
    trajectory = SphericalTrajectory(
        thetas=np.linspace(30,160 , 12, endpoint=False).tolist(),
        pitchs=[40, 55, 65 ],
        radius=[0.3],
        center_point=(0.83, 0, -0.06),
        view_jitter=(5, 5, 60),
    )
    trajectory += SphericalTrajectory(
        thetas=np.linspace(200,330 , 12, endpoint=False).tolist(),
        pitchs=[40, 55, 65, 75],
        radius=[0.3],
        center_point=(0.83, 0, -0.05),
        view_jitter=(5, 5, 60),
    )
    """ trajectory += SphericalTrajectory(
        thetas=np.linspace(45, 315, 16, endpoint=True).tolist(),
        pitchs=[45, 70],
        radius=[0.35],
        center_point=(0.83, 0, -0.06),
        view_jitter=(5, 5, 60),
    ) """
    trajectory += SphericalTrajectory(
        thetas=np.linspace(60, 300, 16, endpoint=True).tolist(),
        pitchs=[60],
        radius=[0.45],
        center_point=(0.83, 0, -0.06),
        view_jitter=(5, 5, 60),
    )
    
    trajectory += CartesianTrajectory(
            (180, 0, 90),
            np.linspace(0.83 - 0.1, 0.83 + 0.1, 3).tolist(), 
            np.linspace(-0.25, 0.25, 3).tolist(),
            0.3,
            view_jitter=(5, 5, 5),
        ) 
    trajectory += CartesianTrajectory(
            (180, 30, 90),
            0.83, 
            np.linspace(-0.3, 0.3, 4).tolist(),
            0.25,
            view_jitter=(5, 5, 5),
        ) 
    trajectory += CartesianTrajectory(
            (180, 330, 90),
            0.93, 
            np.linspace(-0.3, 0.3, 4).tolist(),
            0.25,
            view_jitter=(5, 5, 5),
        )   
    iter = 0
    async for step in executor.execute(robot, trajectory, cam=None):
        iter = iter+1
        print(iter)
    print("moving sequence ended")



@click.command()
@click.option("--capture", is_flag=True)
@click.option("--display", is_flag=True)
@click.option("--save", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/recalibration")
def main(display: bool,capture: bool,save: bool, output: Path):
    asyncio.run(async_main (display, capture,save, output))


if __name__ == "__main__":
    main()
