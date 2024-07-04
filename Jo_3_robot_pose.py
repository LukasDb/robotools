import numpy as np
import time
from robotools.robot import FanucCRX10iAL
import robotools as rt
import yaml
import asyncio
import click
from pathlib import Path
from robotools.camera import HandeyeCalibrator
import subprocess

async def async_main(output:Path):
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene_RS.yaml"))) # make sure to have the right calibration file
    robot: rt.robot.Robot = scene._entities["crx"]
    bg: rt.utility.BackgroundMonitor = scene._entities["Background Monitor"]
    duration = 3
    data = []
    calibrator = HandeyeCalibrator()
    robot_mover = "Jo_3a_move_robot.py"
    
    input("Please move the window to the background screen and press enter")
    bg.setup_window()
    option = input("If you want to display charucos press 1")
    if option == "1":
        bg.set_to_charuco(
                chessboard_size=calibrator.chessboard_size,
                marker_size=calibrator.marker_size,
                n_markers=calibrator.n_markers,
                charuco_dict=calibrator.aruco_dict,)
    else:
        bg.set_to_black()
    
    input(f"This function will capture the robot pose for {duration} minutes and then write it to a file. Start by pressing enter")
    subprocess.Popen(["python",robot_mover])
    time0 = time.time()
    while (time0 + 60*duration) > time.time():
        pose = await robot.get_pose()  # Ensure the coroutine is awaited
        time_array = np.array([time.time()-time0, 0, 0, 0])  # Create a time_array with current time and three zeros
        fused = np.column_stack((pose, time_array))  # Combine 4x4 pose with 4x1 time_array to form 4x5 array
        data.append(fused)
        #print(fused)
        
    np.save("data/robot_tracking/robot_pose_data.npy", data)
    print("done")


@click.command()
@click.option("--output", type=click.Path(path_type=Path), default="data/robot_tracking")
def main(output:Path):
    asyncio.run(async_main(output))

if __name__ == "__main__":
    main()