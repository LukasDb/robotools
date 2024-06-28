import os
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from cv2 import aruco
from cv2 import drawFrameAxes
import open3d as o3d

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
from robotools.geometry import (
    get_6d_vector_from_affine_matrix,
    invert_homogeneous,
    get_affine_matrix_from_6d_vector,
)

async def async_main(display: bool,capture: bool, output: Path) -> None:
    # get calibration values
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene_combined.yaml"))) # make sure to have the right calibration file
    cam: rt.camera.Camera = scene._entities["ZED-M"]
    bg: rt.utility.BackgroundMonitor = scene._entities["Background Monitor"]
    
    w2c_array_raw = np.load(str(output.joinpath("world2cam_array_raw.npy")))
    w2c_array_refined = np.load(str(output.joinpath("world2cam_array_refined.npy")))
    image_array = np.load(str(output.joinpath("frames.npy")))
    w2m = bg.get_pose()
    length = w2c_array_raw.shape[0]
    
  
   
    for step in range(length):
    

        img = image_array[step]
        # compute vectors
        c2m_raw = invert_homogeneous(w2c_array_raw[step,:,:]) @ w2m
        rvec_raw, _ = cv2.Rodrigues(c2m_raw[:3,:3])
        tvec_raw = c2m_raw[:3,3]

        c2m_refined = invert_homogeneous(w2c_array_refined[step,:,:]) @ w2m
        rvec_refined, _ = cv2.Rodrigues(c2m_refined[:3,:3])
        tvec_refined = c2m_refined[:3,3]
    
        # duplicate image
        raw = img.copy()
        refined = img.copy()
        # label
        
        cv2.putText(raw, "Raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # type: ignore
        cv2.putText(refined, "Refined", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # type: ignore
        # draw corner with cam2marker_raw
        raw = cv2.drawFrameAxes(
            image= raw,
            cameraMatrix= cam.calibration.intrinsic_matrix, 
            distCoeffs= cam.calibration.dist_coeffs, 
            rvec= rvec_raw, 
            tvec= tvec_raw, 
            length = 0.1,
            thickness = 3
        )
        # draw corner with cam2marker_refined
        refined = cv2.drawFrameAxes(
            image= refined,
            cameraMatrix= cam.calibration.intrinsic_matrix, 
            distCoeffs= cam.calibration.dist_coeffs, 
            rvec= rvec_refined, 
            tvec= tvec_refined, 
            length = 0.1,
            thickness = 3
        )
        annotated = np.hstack([raw, refined])

        cv2.imshow(f"Calibration {step}", annotated[::2, ::2, ::-1])
        key = cv2.waitKey(0)
        
        







   
@click.command()
@click.option("--capture", is_flag=True)
@click.option("--display", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/recalibration")
def main(display: bool,capture: bool, output: Path):
    asyncio.run(async_main (display, capture, output))


if __name__ == "__main__":
    main()
