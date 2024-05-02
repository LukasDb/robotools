from abc import ABC
import numpy as np
from typing import Iterable
import open3d as o3d


class Trajectory(ABC):

    def __init__(
        self,
        exclusion_boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
            ((-0.3, -0.3, -0.0), (0.3, 0.3, 1.0))
        ],
    ) -> None:
        self._trajectory: list[np.ndarray] = []
        self._exclusion_boxes = exclusion_boxes

    def __next__(self) -> np.ndarray:
        return next(self._trajectory)

    def __iter__(self) -> Iterable[np.ndarray]:
        return iter(self._trajectory)

    def visualize(self) -> None:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis = [origin]
        for pose in self._trajectory:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame.transform(pose)
            vis.append(frame)

        for box in self._exclusion_boxes:
            box = o3d.geometry.AxisAlignedBoundingBox(box[0], box[1])
            box.color = [1.0, 0.0, 0.0]
            vis.append(box)
        o3d.visualization.draw_geometries(vis)

    def __len__(self) -> int:
        return len(self._trajectory)

    def transform(self, transform: np.ndarray, local: bool = True) -> None:
        """apply transform to all poses"""
        if local:
            for i in range(len(self)):
                self._trajectory[i] = self._trajectory[i] @ transform
        else:
            for i in range(len(self)):
                self._trajectory[i] = transform @ self._trajectory[i]

    def in_exclusion_box(self, pose: np.ndarray) -> bool:
        for box in self._exclusion_boxes:
            if (
                box[0][0] <= pose[0, 3] <= box[1][0]
                and box[0][1] <= pose[1, 3] <= box[1][1]
                and box[0][2] <= pose[2, 3] <= box[1][2]
            ):
                return True
        return False

    def __add__(self, other: "Trajectory") -> "Trajectory":
        self._trajectory += other._trajectory
        return self
