import numpy as np
import time
import logging
import asyncio
import tkinter as tk
import itertools as it
import asyncio
from typing import AsyncGenerator

import robotools as rt
from robotools.geometry import invert_homogeneous


class TrajectoryExecutor:
    def __init__(self) -> None:
        pass

    async def execute(
        self,
        robot: rt.robot.Robot,
        trajectory: list[np.ndarray],
        cam: rt.camera.Camera | None = None,
    ) -> AsyncGenerator[int, None]:
        """exectues asynchronoulsy the trajectory on the given robot. If additional a camera is specified, the trajectory poses refer to the calibrated camera pose instead"""

        to_cam = np.eye(4)
        if cam is not None:
            if cam.calibration is None:
                logging.warning(
                    "Camera needs to be calibrated before executing trajectory, using directly robot flange as target!"
                )
            else:
                to_cam = invert_homogeneous(cam.calibration.extrinsic_matrix)

        home_pose = await robot.get_pose()
        
        if home_pose is None:
            logging.warning("Could not get robot home pose!")
            return

        logging.info("Reached home pose")

        #await asyncio.sleep(0.2)

        for idx_trajectory, pose in enumerate(trajectory):
            # generate randomized bg and lights settings, to be re-used for all cameras

            robot_target = pose @ to_cam

            logging.warning(f"Moving robot to {robot_target[:3,3]}")
            await robot.move_to(robot_target, timeout=30)

            #await asyncio.sleep(1.0)

            yield idx_trajectory

        await robot.move_to(home_pose, timeout=50)
