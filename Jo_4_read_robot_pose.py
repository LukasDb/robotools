import numpy as np
import click
from pathlib import Path



@click.command()
@click.option("--capture", is_flag=True)
@click.option("--display", is_flag=True)
@click.option("--save", is_flag=True)
@click.option("--output", type=click.Path(path_type=Path), default="data/robot_tracking")
def main(save: bool, display: bool,capture: bool, output: Path):
   robot_pose= np.load(str(output.joinpath("robot_pose_data.npy")))

if __name__ == "__main__":
    main()