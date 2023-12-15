from typing import Union

import cv2
import numpy as np

from supervision.annotators.utils import ColorLookup, get_color_by_index
from supervision.draw.color import Color, ColorPalette

from icecream import ic

# COLOR_LIST = sv.ColorPalette.from_hex([
#     '#ff2d55',
#     '#0f7f07',
#     '#0095ff',
#     '#ffcc00',
#     '#46f0f0',
#     '#ff9500',
#     '#d2f53c',
#     '#cf52de',
# ])

class PoseAnnotator:
    """
    A class for pose elements on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        radius: int = 4,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            thickness (int): Thickness of the bounding box lines.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.radius: int = radius
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
            self,
            scene: np.ndarray,
            ultralytics_results,
        ) -> np.ndarray:

        poses = ultralytics_results.keypoints.data.cpu().numpy()

        # puntos
        circles = {
            "head": [0,1,2,3,4],
            "arms": [5,6,7,8,9,10],
            "trunk": [11,12],
            "legs": [13,14,15,16]
        }

        # lineas
        lines = {
            "head": [(0,1),(0,2),(1,2),(1,3),(2,4)],
            "arms": [(9,7),(7,5),(5,6),(6,8),(8,10)],
            "trunk": [(5,11),(6,12),(11,12)],
            "legs": [(11,13),(13,15),(12,14),(14,16)]
        }

        colors = {
            "head": get_color_by_index(color=self.color, idx=0),
            "arms": get_color_by_index(color=self.color, idx=1),
            "trunk": get_color_by_index(color=self.color, idx=2),
            "legs": get_color_by_index(color=self.color, idx=3)
        }

        for pose in poses:
            for key, value in circles.items():
                for index in value:
                    x, y, score = pose[index]
                    if score >= 0.7:
                        cv2.circle(scene, (int(x),int(y)), self.radius, colors[key].as_bgr(), -1)

            for key, value in lines.items():
                for index_1, index_2 in value:
                    x1, y1, score_1 = pose[index_1]
                    x2, y2, score_2 = pose[index_2]
                    if score_1 >= 0.7 and score_2 >= 0.7:
                        cv2.line(scene, (int(x1), int(y1)), (int(x2), int(y2)), colors[key].as_bgr(), self.thickness, cv2.LINE_AA)

        return scene