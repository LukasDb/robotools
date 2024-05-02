from .trajectory import Trajectory
import numpy as np
from scipy.spatial.transform import Rotation as R


class SphericalTrajectory(Trajectory):
    def __init__(
        self,
        thetas: list[float] | float,
        pitchs: list[float] | float,
        radius: list[float] | float,
        center_point: tuple[float, float, float],
        view_jitter: tuple[float, float, float] | None = None,
        **kwargs: dict
    ) -> None:
        super().__init__(**kwargs)
        self.thetas = thetas if isinstance(thetas, list) else [thetas]
        self.pitchs = pitchs if isinstance(pitchs, list) else [pitchs]
        self.radii = radius if isinstance(radius, list) else [radius]
        self.view_jitter = view_jitter if view_jitter is not None else (0, 0, 0)
        self.center_point = center_point

        for _pitch in self.pitchs:
            for _theta in self.thetas:
                for radius in self.radii:
                    theta = _theta + np.random.uniform(-self.view_jitter[0], self.view_jitter[0])
                    pitch = _pitch + np.random.uniform(-self.view_jitter[1], self.view_jitter[1])
                    roll = np.random.uniform(-self.view_jitter[2], self.view_jitter[2])

                    x = (
                        radius * np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(pitch))
                        + center_point[0]
                    )
                    y = (
                        radius * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(pitch))
                        + center_point[1]
                    )
                    z = radius * np.sin(np.deg2rad(pitch)) + center_point[2]

                    pose_mat = np.eye(4)
                    pose_mat[:3, :3] = R.from_euler(
                        "ZXZ", [theta + 90, -90 - pitch, roll], degrees=True
                    ).as_matrix()
                    pose_mat[:3, 3] = (x, y, z)

                    if self.in_exclusion_box(pose_mat):
                        continue

                    self._trajectory.append(pose_mat)
