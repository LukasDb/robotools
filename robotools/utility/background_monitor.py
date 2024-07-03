from typing import Any, Literal
import numpy as np
import numpy.typing as npt
import tkinter as tk
from PIL import Image, ImageTk
import screeninfo
import pathlib
import cv2
from cv2 import aruco
import logging

import robotools as rt


class DemoMonitor:
    height = 2160
    width = 3840
    diagonal_16_by_9 = float(np.linalg.norm((16, 9)))
    width_mm = 16 / diagonal_16_by_9 * 800  # 0.69726
    height_mm = 9 / diagonal_16_by_9 * 800  # 0.3922


class BackgroundMonitor(rt.Entity):
    """Background Monitor, the pose refers to the center of the monitor
    with coordinate system as if it was a cv2 image
    """

    def __init__(self,name:str = 'Default_BG_Name') -> None:
        super().__init__(name = name)
        # get secondary monitor info
        self.window = tk.Toplevel()
        self.window.title(f"{self.name} -> MOVE THIS WINDOW TO SECONDARY MONITOR")

        self._is_demo_mode = False
        # create fullscreen canvas
        self.canvas = ResizableImage(self.window, mode="zoom")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # move window to second screen
        self.window.geometry("640x480")
        self.window.geometry("+0+0")

        self._monitor = self._get_current_monitor()
        self.setup_window(set_fullscreen=False)

        self.aruco_dict = None
        self.charuco_board = None

    def disable(self):
        # close tk window
        self.window.destroy()

    def enable(self):
        self.__init__()

    def to_config(self) -> dict:
        out = super().to_config()
        return out

    def from_config(self, config: dict) -> None:
        super().from_config(config)
        self.window.title(f"{self.name} -> MOVE THIS WINDOW TO SECONDARY MONITOR")
        return
    
    def _get_current_monitor(self) -> screeninfo.Monitor | DemoMonitor:
        if self._is_demo_mode:
            logging.warn("USING MOCK MONITOR DIMENSIONS")
            return DemoMonitor()

        monitors = screeninfo.get_monitors()
        x, y = self.window.winfo_x(), self.window.winfo_y()
        monitor = None
        for m in reversed(monitors):
            if m.x <= x < m.width + m.x and m.y <= y < m.height + m.y:
                monitor = m
                break
        logging.debug(f"Using monitor {monitor}")
        assert monitor is not None, "Could not find monitor"
        return monitor

    @property
    def screen_width(self) -> int:
        return self._monitor.width

    @property
    def screen_height(self) -> int:
        return self._monitor.height

    @property
    def screen_width_m(self) -> float:
        width_mm = self._monitor.width_mm
        assert width_mm is not None, "Could not get monitor width in mm"
        return width_mm / 1000.0

    @property
    def screen_height_m(self) -> float:
        height_mm = self._monitor.height_mm
        assert height_mm is not None, "Could not get monitor height in mm"
        return height_mm / 1000.0

    def set_to_depth_texture(self) -> None:
        # from https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance
        """ textured_folder = pathlib.Path("./intel_textured_patterns")
        textured_paths = list(textured_folder.iterdir())
        # choose the one that is closest to the current resolution
        textured_widths = [int(p.name.split("_")[5]) - self.width for p in textured_paths]
        selected = np.argmin(np.abs(textured_widths))
        textured_path = textured_paths[selected] """

        textured_path = pathlib.Path("./intel_textured_patterns/Random_image_10_90_10_1280_720.png")
        self._load_image_to_full_canvas(textured_path)

    def set_to_black(self) -> None:
        textured_path = pathlib.Path("./intel_textured_patterns/plain_black.jpg")
        self._load_image_to_full_canvas(textured_path)

    def _load_image_to_full_canvas(self, path: pathlib.Path) -> None:
        self.setup_window(set_fullscreen=True)
        with path.open("rb") as f:
            bg = np.asarray(Image.open(f))
        # scale image to fill the screen
        bg = cv2.resize(  # type: ignore
            bg,
            (
                int(self.window.winfo_width()),
                int(self.window.winfo_height()),
            ),
        )
        self.set_image(bg)

    def setup_window(self, set_fullscreen: bool = True) -> None:
        # get window size
        # self.window.geometry("+0+0")
        if set_fullscreen:
            self.window.attributes("-fullscreen", True)
        self.window.update()
        self.width = self.window.winfo_width()
        self.height = self.window.winfo_height()

        self._monitor = self._get_current_monitor()

        logging.debug(
            f"Setting window up for screen with ({self.screen_width}, {self.screen_height}) pixels"
        )

        logging.debug(
            f"Setting window up for screen with ({self.screen_width_m}, {self.screen_height_m}) meters"
        )

    def set_image(self, image: npt.NDArray[np.float64]) -> None:
        """Set the image of the background monitor"""
        # self.image_tk = ImageTk.PhotoImage(image=Image.fromarray(image))
        # self.canvas.itemconfig(self.image_container, image=self.image_tk)
        self.canvas.set_image(image)
        self.canvas.update()

    def set_to_charuco(
        self,
        chessboard_size: float = 0.05,
        marker_size: float = 0.04,
        n_markers: tuple[int, int] = (7, 5),
        charuco_dict: dict = {},
    ) -> None:
        width = self.screen_width
        width_m = self.screen_width_m
        height = self.screen_height
        height_m = self.screen_height_m

        pixel_w = width_m / width
        pixel_h = height_m / height
        logging.debug(f"Pixel dimensions: {pixel_w} x {pixel_h} m")

        chessboard_size_scaled = chessboard_size // pixel_w * pixel_w  # in m
        marker_size_scaled = marker_size // pixel_w * pixel_w

        self.aruco_dict: Any = charuco_dict
        self.charuco_board: Any = aruco.CharucoBoard(
            n_markers, chessboard_size, marker_size_scaled, self.aruco_dict
        )

        # create an appropriately sized image
        charuco_img_width_m = n_markers[0] * chessboard_size_scaled  # in m
        charuco_img_width = charuco_img_width_m / width_m * width  # in pixel

        charuco_img_height_m = n_markers[1] * chessboard_size_scaled  # in m
        charuco_img_height = charuco_img_height_m / height_m * height

        # charuco board is created with pixel_w as square size
        # the actual pixel dimensions can vary so image needs to stretched/compressed in y
        y_factor = pixel_h / pixel_w
        charuco_img_height *= y_factor

        hor_pad = round((width - charuco_img_width) / 2)
        vert_pad = round((height - charuco_img_height) / 2)

        logging.debug(f"Creating Charuco image with size: {charuco_img_width, charuco_img_height}")
        charuco_img = self.charuco_board.generateImage(
            (round(charuco_img_width), round(charuco_img_height))
        )

        full_img = np.zeros((height, width), dtype=np.uint8)
        full_img[
            vert_pad : vert_pad + charuco_img.shape[0], hor_pad : hor_pad + charuco_img.shape[1]
        ] = charuco_img

        # charuco_img = cv2.copyMakeBorder(  # type: ignore
        #     charuco_img,
        #     vert_pad,
        #     vert_pad,
        #     hor_pad,
        #     hor_pad,
        #     cv2.BORDER_CONSTANT,  # type: ignore
        #     value=(0, 0, 0),
        # )

        self.set_image(full_img)

        # calculate transform to the center of the screen
        # same orientation, translated by half of charuco width and height
        self.markers2monitor = np.eye(4)
        self.markers2monitor[0, 3] = charuco_img_width_m / 2.0
        self.markers2monitor[1, 3] = charuco_img_height_m / 2.0

        logging.warn(
            f"Confirm the dimensions of the chessboard in the image: {chessboard_size_scaled}"
        )
        logging.warn(f"Confirm the dimensions of the markers in the image: {marker_size_scaled}")


class ResizableImage(tk.Canvas):
    def __init__(
        self,
        master: tk.Misc,
        image: None | npt.NDArray[np.uint8] = None,
        mode: Literal["stretch", "zoom", "fit"] = "fit",
        **kwargs: Any,
    ) -> None:
        tk.Canvas.__init__(self, master, kwargs)
        self._canvas_img = self.create_image(0, 0, anchor=tk.NW)
        self._img: npt.NDArray[np.uint8] | None = None
        self.mode = mode
        if image is not None:
            self._img = image.copy()
        self.bind(sequence="<Configure>", func=self._on_refresh)

    def _on_refresh(self, event: None | tk.Event = None) -> None:
        if event is not None:
            self.widget_width = int(event.width)
            self.widget_height = int(event.height)

        if self._img is None or not hasattr(self, "widget_width"):
            return

        img_width = self._img.shape[1]
        img_height = self._img.shape[0]
        if self.mode == "fit":
            scale = min(self.widget_width / img_width, self.widget_height / img_height)
            scaled_img_width = int(img_width * scale)
            scaled_img_height = int(img_height * scale)
        elif self.mode == "zoom":
            scale = max(self.widget_width / img_width, self.widget_height / img_height)
            scaled_img_width = int(img_width * scale)
            scaled_img_height = int(img_height * scale)

        elif self.mode == "stretch":
            scaled_img_width = self.widget_width
            scaled_img_height = self.widget_height
        else:
            raise ValueError(f"Unknown image mode: {self.mode}")

        img_resized = cv2.resize(  # type: ignore
            self._img.copy(), (scaled_img_width, scaled_img_height)
        )

        self._img_tk = ImageTk.PhotoImage(Image.fromarray(img_resized))
        self.itemconfig(self._canvas_img, image=self._img_tk)  # , anchor=tk.CENTER)

    def set_image(self, image: np.ndarray) -> None:
        self._img = image
        if self._canvas_img not in self.children:
            self._canvas_img = self.create_image(0, 0, anchor=tk.NW)
        self._on_refresh()  # resize to canvas

    def clear_image(self) -> None:
        if self.winfo_exists():
            self.delete(self._canvas_img)
