import cv2
from .camera import Camera, CamFrame, HandeyeCalibration
import numpy as np
import json
from pathlib import Path
import pyrealsense2 as rs
import logging
import threading


# mappings
occ_speed_map = {
    "very_fast": 0,
    "fast": 1,
    "medium": 2,
    "slow": 3,
    "wall": 4,
}
tare_accuracy_map = {
    "very_high": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}
scan_map = {
    "intrinsic": 0,
    "extrinsic": 1,
}
fl_adjust_map = {"right_only": 0, "both_sides": 1}


class Realsense(Camera):
    height = 1080
    width = 1920
    DEPTH_H = 720
    DEPTH_W = 1280

    @staticmethod
    def get_available_devices() -> list["Realsense"]:
        ctx = rs.context()  # type: ignore
        devices = ctx.query_devices()
        cams = []
        for dev in devices:
            logging.info(f"Found device: {dev.get_info(rs.camera_info.name)}")  # type: ignore
            serial_number = dev.get_info(rs.camera_info.serial_number)  # type: ignore

            cams.append(Realsense(serial_number))

        return cams

    def __init__(
        self,
        serial_number: str | None = None,
        calibration: HandeyeCalibration | None = None,
    ) -> None:
        super().__init__(f"Realsense_{serial_number}", calibration)
        if serial_number is not None:
            self.start(serial_number)

    def start(self, serial_number: str) -> None:
        self._serial_number = serial_number
        self._lock = threading.Lock()

        self._pipeline = rs.pipeline()  # type: ignore
        self._config = config = rs.config()  # type: ignore
        config.enable_device(self._serial_number)

        try:
            pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)  # type: ignore
            pipeline_profile = config.resolve(pipeline_wrapper)
        except RuntimeError:
            logging.error(f"Could not start realsense {serial_number}")
            return

        self.device = pipeline_profile.get_device()

        if self.device.get_info(rs.camera_info.firmware_version) != self.device.get_info(  # type: ignore
            rs.camera_info.recommended_firmware_version  # type: ignore
        ):
            logging.warn(f"Camera {self.name} firmware is out of date")

        config.enable_stream(rs.stream.depth, self.DEPTH_W, self.DEPTH_H, rs.format.z16, 30)  # type: ignore
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, 30)  # type: ignore
        self.align_to_rgb = rs.align(rs.stream.color)  # type: ignore

        self.temporal_filter = rs.temporal_filter(  # type: ignore
            smooth_alpha=0.2,
            smooth_delta=20,
            persistence_control=3,
        )

        # self.spatial_filter = rs.spatial_filter(  # type: ignore
        #     smooth_alpha=0.5,
        #     smooth_delta=20,
        #     magnitude=1,
        #     hole_fill=2,
        # )

        # TODO rework this
        device_name = self.device.get_info(rs.camera_info.name)  # type: ignore
        if "D415" in device_name:
            profile_path = Path("realsense_profiles/d415_HQ.json")
        elif "D435" in device_name:
            profile_path = Path("realsense_profiles/d435_HQ.json")

        logging.debug(f"[{self.name}] Loading configuration: {profile_path.name}")
        rs.rs400_advanced_mode(self.device).load_json(profile_path.read_text())  # type: ignore

        self.sensor = self.device.first_depth_sensor()
        self.depth_scale = self.sensor.get_depth_scale()

        self.sensor.set_option(rs.option.hdr_enabled, False)  # type: ignore

        # Start streaming
        self._pipeline.start(self._config)

    def to_config(self) -> dict:
        out = super().to_config()
        out["serial_number"] = self._serial_number
        return out

    def from_config(self, config: dict) -> None:
        serial_number = config["serial_number"]
        self.start(serial_number)
        return super().from_config(config)

    def get_frame(self) -> CamFrame:
        with self._lock:
            output = CamFrame()
            frames = self._pipeline.wait_for_frames()

            # makes depth frame same resolution as rgb frame
            aligned_frames = self.align_to_rgb.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                logging.warn("Could not get camera frame")
                return output

            depth_frame = self.temporal_filter.process(depth_frame)
            # depth_frame = self.spatial_filter.process(depth_frame)

            depth_image = np.asarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
            color_image = np.asarray(color_frame.get_data())  # type: ignore

            output.depth = depth_image
            output.rgb = color_image

            return output
