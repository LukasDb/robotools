from .camera import Camera, CamFrame
import numpy as np
from typing import List
import cv2
import logging

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    logging.error("Could not import pyzed. ZED cameras not available.")


""" NOT UP TO DATE"""

""" try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    logging.error("Could not import pyzed. ZED cameras not available.")
 """


class ZedCamera(Camera):
    width = 1920
    height = 1080

    @staticmethod
    def get_available_devices() -> List["ZedCamera"]:
        cams: List["ZedCamera"] = []
        logging.info("Getting ZED devices...")
        logging.warn(
            "If you get segmentation fault here, reverse the USB type C cable on the ZED camera."
        )
        """   if not "sl" in locals():
            logging.warn("sl not in locals")
            return cams """

        try:
            dev_list = sl.Camera.get_device_list()
        except Exception as e:
            return []

        for dev in dev_list:  # list[DeviceProperties]
            logging.info(f"Found device: {dev}")
            cams.append(ZedCamera(dev.serial_number))
        return cams

    def __init__(self, serial_number: str = "9999") -> None:
        # added a default serial number to prevent an error in script 2
        self._serial_number = serial_number
        # return
        try:
            self.init_params = sl.InitParameters()
        except Exception:
            return
        self.init_params.sdk_verbose = 0  # 1 for verbose
        self.init_params.camera_resolution = sl.RESOLUTION.HD2K
        self.init_params.camera_fps = 15
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
        self.init_params.coordinate_units = (
            sl.UNIT.METER
        )  # Use millimeter units (for depth measurements)
        self.init_params.depth_maximum_distance = 1  # Set maximum distance to 1m
        self.init_params.depth_minimum_distance = 0.1  # Set minimum distance to 10cm
        self.init_params.depth_stabilization = 20
        # self.init_params.set_from_serial_number(serial_number)

        self.device = sl.Camera()
        err = self.device.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            logging.error("Error opening ZED camera: ", err)
            exit()

        info = self.device.get_camera_information()
        super().__init__(str(info.camera_model))
        # super().__init__(str(info.camera_model) + f"-{self._serial_number}")

        self._rgb_buffer = sl.Mat()
        self._depth_buffer = sl.Mat()

    # def get_frame(self, depth_quality: DepthQuality) -> CamFrame:
    # TODO update depth quality
    def get_frame(self) -> CamFrame:
        output = CamFrame()

        if self.device.grab() == sl.ERROR_CODE.SUCCESS:
            self.device.retrieve_image(self._rgb_buffer, sl.VIEW.LEFT)
            output.rgb = cv2.cvtColor(self._rgb_buffer.get_data(deep_copy=True), cv2.COLOR_BGR2RGB)  # type: ignore

            self.device.retrieve_image(self._rgb_buffer, sl.VIEW.RIGHT)
            output.rgb_R = cv2.cvtColor(self._rgb_buffer.get_data(deep_copy=True), cv2.COLOR_BGR2RGB)  # type: ignore

            self.device.retrieve_measure(self._depth_buffer, sl.MEASURE.DEPTH)
            output.depth = self._depth_buffer.get_data(deep_copy=True)
            output.depth = np.nan_to_num(
                output.depth, copy=False, nan=0, posinf=0, neginf=0
            )  # converting erroneous pixels to zero, to comply with the realsense standard

            self.device.retrieve_measure(self._depth_buffer, sl.MEASURE.DEPTH_RIGHT)
            output.depth_R = self._depth_buffer.get_data(deep_copy=True)
            output.depth_R = np.nan_to_num(
                output.depth_R, copy=False, nan=0, posinf=0, neginf=0
            )  # converting erroneous pixels to zero, to comply with the realsense standard

        return output

    @property
    def unique_id(self) -> str:
        return "zed_" + str(self._serial_number)

    def _set_hq_depth(self) -> None:
        self.is_hq_depth = True
        # TODO possibly change other parameters, too

    def _set_lq_depth(self) -> None:
        self.is_hq_depth = False
        # TODO possibly change other parameters, too
