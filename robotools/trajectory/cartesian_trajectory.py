from .trajectory import Trajectory
import numpy as np
from scipy.spatial.transform import Rotation as R


class CartesianTrajectory(Trajectory):
    def __init__(
        self,
        view_euler_xyz: tuple[float, float, float],
        x_positions: list[float] | float,
        y_positions: list[float] | float,
        z_positions: list[float] | float,
        view_jitter: tuple[float, float, float] | None = None,
        **kwargs: dict
    ) -> None:
        super().__init__(**kwargs)
        self.x_positions = x_positions if isinstance(x_positions, list) else [x_positions]
        self.y_positions = y_positions if isinstance(y_positions, list) else [y_positions]
        self.z_positions = z_positions if isinstance(z_positions, list) else [z_positions]
        self.view_dir = view_euler_xyz
        self.view_jitter = view_jitter if view_jitter is not None else (0, 0, 0)

        for x in self.x_positions:
            for y in self.y_positions:
                for z in self.z_positions:

                    view_x = self.view_dir[0] + np.random.uniform(
                        -self.view_jitter[0], self.view_jitter[0]
                    )
                    view_y = self.view_dir[1] + np.random.uniform(
                        -self.view_jitter[1], self.view_jitter[1]
                    )
                    view_z = self.view_dir[2] + np.random.uniform(
                        -self.view_jitter[2], self.view_jitter[2]
                    )

                    pose_mat = np.eye(4)
                    pose_mat[:3, :3] = R.from_euler(
                        "XYZ", [view_x, view_y, view_z], degrees=True
                    ).as_matrix()
                    pose_mat[:3, 3] = (x, y, z)

                    if self.in_exclusion_box(pose_mat):
                        continue

                    self._trajectory.append(pose_mat)
