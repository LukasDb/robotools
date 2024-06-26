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
    output_lines =["All distance parameters are given in meters, all rotation parameters in degrees.\n"]

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
    Rx_raw = np.zeros(length)
    Rx_refined = np.zeros(length)

    Ry_raw = np.zeros(length)
    Ry_refined = np.zeros(length)

    Rz_raw = np.zeros(length)
    Rz_refined = np.zeros(length)

    for step in range(length):
        euler_raw, _ = rt.geometry.get_euler_from_affine_matrix(w2c_array_raw[step])
        Rx_raw[step] = euler_raw[0]
        Ry_raw[step] = euler_raw[1]
        Rz_raw[step] = euler_raw[2]

        euler_refined, _ = rt.geometry.get_euler_from_affine_matrix(w2c_array_refined[step])
        Rx_refined[step] = euler_refined[0]
        Ry_refined[step] = euler_refined[1]
        Rz_refined[step] = euler_refined[2]

        # simple check to filter out datapoints at the -180 / +180 border
        # includes hacky trick to get the differeces right anyways
        if abs(Rx_refined[step] - Rx_raw[step]) > 50:
            print(f"problem in {step}")
            print(f"raw{Rx_raw[step]}")
            print(f"refined{Rx_refined[step]}")
            if Rx_refined[step] < 150:
                Rx_refined[step] = Rx_refined[step] + 180
                Rx_raw[step] = Rx_raw[step] - 180
            else:
                Rx_refined[step] = Rx_refined[step] - 180
                Rx_raw[step] = Rx_raw[step] + 180
            print(f"raw{Rx_raw[step]}")
            print(f"refined{Rx_refined[step]}")


        


    #get values x
    bias_x, std_dev_x, max_diff_x = analysis(x_raw,x_refined)
    bias_y, std_dev_y, max_diff_y = analysis(y_raw,y_refined)
    bias_z, std_dev_z, max_diff_z = analysis(z_raw,z_refined)

    bias_d_eucl, std_dev_d_eucl, max_diff_d_eucl = analysis(d_eucl_raw,d_eucl_refined)

    bias_Rx, std_dev_Rx, max_diff_Rx = analysis(Rx_raw,Rx_refined)
    bias_Ry, std_dev_Ry, max_diff_Ry = analysis(Ry_raw,Ry_refined)
    bias_Rz, std_dev_Rz, max_diff_Rz = analysis(Rz_raw,Rz_refined)

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
