from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from enum import Enum, auto
import numpy.typing as npt

import robotools as rt
from ..entity import AsyncEntity


@dataclass
class CamFrame:
    rgb: npt.NDArray[np.float64] | None = None
    rgb_R: npt.NDArray[np.float64] | None = None
    depth: npt.NDArray[np.float64] | None = None
    depth_R: npt.NDArray[np.float64] | None = None


@dataclass
class HandeyeCalibration:
    intrinsic_matrix: npt.NDArray[np.float64]
    dist_coeffs: npt.NDArray[np.float64]
    extrinsic_matrix: npt.NDArray[np.float64]

    def to_config(self) -> dict:
        return {
            "intrinsic_matrix": self.intrinsic_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "extrinsic_matrix": self.extrinsic_matrix.tolist(),
        }

    def from_config(self, config: dict) -> "HandeyeCalibration":
        self.intrinsic_matrix = np.array(config["intrinsic_matrix"])
        self.dist_coeffs = np.array(config["dist_coeffs"])
        self.extrinsic_matrix = np.array(config["extrinsic_matrix"])
        return self


class Camera(AsyncEntity, ABC):
    width: int
    height: int

    def __init__(
        self,
        name: str,
        calibration: HandeyeCalibration | None = None,
    ):
        AsyncEntity.__init__(self, name)

        self.calibration: HandeyeCalibration | None = calibration

    def to_config(self) -> dict:
        out = super().to_config()
        out["calibration"] = "none" if self.calibration is None else self.calibration.to_config()
        return out

    def from_config(self, config: dict) -> None:

        if config["calibration"] == "none":
            calibration = None
        else:
            calibration = HandeyeCalibration(np.eye(4), np.eye(1), np.eye(4)).from_config(
                config["calibration"]
            )
        self.calibration = calibration
        return super().from_config(config)

    def is_calibrated(self) -> bool:
        return self.calibration is not None

    @abstractmethod
    def get_frame(self) -> CamFrame:
        pass
