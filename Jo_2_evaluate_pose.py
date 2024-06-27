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
    #load arrays
    w2c_array_raw = np.load(str(output.joinpath("world2cam_array_raw.npy")))
    w2c_array_refined = np.load(str(output.joinpath("world2cam_array_refined.npy")))

    #variables
    output_lines =["All distance parameters are given in meters, all rotation parameters in degrees.\nAlways raw - refined \n"]

    #extract coordinates
    x_raw = w2c_array_raw[:, 0, 3] 
    x_refined =w2c_array_refined[:, 0, 3]

    y_raw = w2c_array_raw[:, 1 ,3]
    y_refined = w2c_array_refined[:,1 ,3]

    z_raw = w2c_array_refined[:, 2, 3]
    z_refined = w2c_array_refined[:, 2, 3]

    d_eucl_raw = np.sqrt(x_raw**2 + y_raw**2 + z_raw**2)
    d_eucl_refined = np.sqrt(x_refined**2 + y_refined**2 + z_refined**2)

    
    length = len(x_raw)
    Rx_diff = np.zeros(length)

    Ry_diff = np.zeros(length)

    Rz_diff= np.zeros(length)

    for step in range(length):
        affine_combined = w2c_array_raw[step] @ invert_homogeneous(w2c_array_refined[step])
        euler_combined, _ = rt.geometry.get_euler_from_affine_matrix(affine_combined)
        Rx_diff[step] = euler_combined[0]
        Ry_diff[step] = euler_combined[1]
        Rz_diff[step] = euler_combined[2]
        
        text_out = step
        print(text_out)
        output_lines.append(f"{text_out}")
        text_out = euler_combined
        print(text_out)
        output_lines.append(f"{text_out}")


        


    #get values x
    bias_x, std_dev_x, max_diff_x = analysis(x_raw,x_refined)
    bias_y, std_dev_y, max_diff_y = analysis(y_raw,y_refined)
    bias_z, std_dev_z, max_diff_z = analysis(z_raw,z_refined)

    bias_d_eucl, std_dev_d_eucl, max_diff_d_eucl = analysis(d_eucl_raw,d_eucl_refined)

    bias_Rx = np.mean(Rx_diff)
    std_dev_Rx = np.std(Rx_diff)
    max_diff_Rx = np.max(np.abs(Rx_diff))
    
    bias_Ry = np.mean(Ry_diff)
    std_dev_Ry = np.std(Ry_diff)
    max_diff_Ry = np.max(np.abs(Ry_diff))
    
    bias_Rz = np.mean(Rz_diff)
    std_dev_Rz = np.std(Rz_diff)
    max_diff_Rz = np.max(np.abs(Rz_diff))

    text_out = f"We evaluated {length} Images. \n \n \n"
    print(text_out)
    output_lines.append(text_out)

    print_output('x',bias_x,std_dev_x, max_diff_x, output_lines)
    print_output('y',bias_y,std_dev_y, max_diff_y, output_lines)
    print_output('z',bias_z,std_dev_z, max_diff_z, output_lines)
    print_output('Euclidian Distance',bias_d_eucl,std_dev_d_eucl, max_diff_d_eucl, output_lines)
    print_output('Rx',bias_Rx,std_dev_Rx, max_diff_Rx, output_lines)
    print_output('Ry',bias_Ry,std_dev_Ry, max_diff_Ry, output_lines)
    print_output('Rz',bias_Rz,std_dev_Rz, max_diff_Rz, output_lines)

    #save to file
    with open(str(output.joinpath("evaluation_results.txt")), 'w') as file:
        file.write("\n".join(output_lines))
    print("Done")


    


def analysis(array_raw, array_refined):
    bias = np.mean(array_raw -array_refined)
    std_dev = np.std(array_raw - array_refined)
    max_diff = np.max(np.abs(array_raw - array_refined))
    
    return bias, std_dev, max_diff
def print_output(param, bias, std_dev, max_diff,output_lines):
    text_out = f"For the parameter {param} we obtained: \n Bias: {bias} \n Standard deviation: {std_dev} \n Maximum difference: {max_diff} \n"
    print(text_out)
    output_lines.append(text_out)


    







@click.command()
@click.option("--capture", is_flag=True)
@click.option("--display", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/recalibration")
def main(display: bool,capture: bool, output: Path):
    asyncio.run(async_main (display, capture, output))


if __name__ == "__main__":
    main()
