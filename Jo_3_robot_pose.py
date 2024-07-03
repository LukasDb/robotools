import numpy as np
import time
from robotools.robot import FanucCRX10iAL
import robotools as rt
import yaml
import asyncio
import click
from pathlib import Path

async def async_main(output:Path):
    scene = rt.Scene()
    scene.from_config(yaml.safe_load(open("scene_combined.yaml"))) # make sure to have the right calibration file
    robot: rt.robot.Robot = scene._entities["crx"]
    duration = 3
    data = []
    
    input(f"This function will capture the robot pose for {duration} minutes and then write it to a file. Start by pressing enter")
    time0 = time.time()
    while (time0 + 60*duration) > time.time():
        pose = await robot.get_pose()  # Ensure the coroutine is awaited
        time_array = np.array([time.time()-time0, 0, 0, 0])  # Create a time_array with current time and three zeros
        fused = np.column_stack((pose, time_array))  # Combine 4x4 pose with 4x1 time_array to form 4x5 array
        data.append(fused)
        print(fused)
        time.sleep(1)
    np.save("robot_pose_data.npy", data)
    print("done")


@click.command()
@click.option("--output", type=click.Path(path_type=Path), default="data/robot_tracking")
def main(output:Path):
    asyncio.run(async_main(output))

if __name__ == "__main__":
    main()