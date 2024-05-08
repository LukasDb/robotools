import numpy as np
import cv2


class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @staticmethod
    def get_from_mask(mask: np.ndarray, object_id: int = 1) -> "BBox":
        y, x = np.where(mask == object_id)
        if len(y) == 0:
            return (0, 0, 0, 0)
        x1 = np.min(x).tolist()
        x2 = np.max(x).tolist()
        y1 = np.min(y).tolist()
        y2 = np.max(y).tolist()
        return BBox(x1, y1, x2, y2)

    def draw(self, rgb: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
        cv2.rectangle(rgb, (self.x1, self.y1), (self.x2, self.y2), color, 2)
        return rgb

    def astuple(self):
        return self.x1, self.y1, self.x2, self.y2
