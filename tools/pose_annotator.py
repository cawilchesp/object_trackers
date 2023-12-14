from typing import Union

import cv2
import numpy as np

from supervision.annotators.utils import ColorLookup, get_color_by_index
from supervision.draw.color import Color, ColorPalette

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
        self.color_lookup: ColorLookup = color_lookup

    def annotate(
            self,
            scene: np.ndarray,
            ultralytics_results,
        ) -> np.ndarray:

        poses = ultralytics_results.keypoints.data.cpu().numpy().astype(int)
    
        for keypoints in poses:
            color_head = get_color_by_index(color=self.color, idx=0)
            nose =      (keypoints[0][0], keypoints[0][1])
            left_eye =  (keypoints[1][0], keypoints[1][1])
            right_eye = (keypoints[2][0], keypoints[2][1])
            left_ear =  (keypoints[3][0], keypoints[3][1])
            right_ear = (keypoints[4][0], keypoints[4][1])
            # Draw points
            cv2.circle(scene, nose, 4, color_head.as_bgr(), -1)
            cv2.circle(scene, left_eye, 4, color_head.as_bgr(), -1)
            cv2.circle(scene, right_eye, 4, color_head.as_bgr(), -1)
            cv2.circle(scene, left_ear, 4, color_head.as_bgr(), -1)
            cv2.circle(scene, right_ear, 4, color_head.as_bgr(), -1)
            # Draw lines
            cv2.line(scene, nose, left_eye, color_head.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, nose, right_eye, color_head.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, right_eye, left_eye, color_head.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, left_ear, left_eye, color_head.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, right_ear, right_eye, color_head.as_bgr(), 2, cv2.LINE_AA)

            color_arms = get_color_by_index(color=self.color, idx=1)
            left_shoulder =  (keypoints[5][0], keypoints[5][1])
            right_shoulder = (keypoints[6][0], keypoints[6][1])
            left_elbow =     (keypoints[7][0], keypoints[7][1])
            right_elbow =    (keypoints[8][0], keypoints[8][1])
            left_hand =      (keypoints[9][0], keypoints[9][1])
            right_hand =     (keypoints[10][0], keypoints[10][1])

            # Draw points
            cv2.circle(scene, left_shoulder, 4, color_arms.as_bgr(), -1)
            cv2.circle(scene, right_shoulder, 4, color_arms.as_bgr(), -1)
            cv2.circle(scene, left_elbow, 4, color_arms.as_bgr(), -1)
            cv2.circle(scene, right_elbow, 4, color_arms.as_bgr(), -1)
            cv2.circle(scene, left_hand, 4, color_arms.as_bgr(), -1)
            cv2.circle(scene, right_hand, 4, color_arms.as_bgr(), -1)
            # Draw lines
            cv2.line(scene, left_hand, left_elbow, color_arms.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, left_elbow, left_shoulder, color_arms.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, left_shoulder, right_shoulder, color_arms.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, right_shoulder, right_elbow, color_arms.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, right_elbow, right_hand, color_arms.as_bgr(), 2, cv2.LINE_AA)

            color_trunk = get_color_by_index(color=self.color, idx=2)
            left_hip = (keypoints[11][0], keypoints[11][1])
            right_hip = (keypoints[12][0], keypoints[12][1])
            # Draw points
            cv2.circle(scene, left_hip, 4, color_trunk.as_bgr(), -1)
            cv2.circle(scene, right_hip, 4, color_trunk.as_bgr(), -1)
            # Draw lines
            cv2.line(scene, left_shoulder, left_hip, color_trunk.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, left_hip, right_hip, color_trunk.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, right_hip, right_shoulder, color_trunk.as_bgr(), 2, cv2.LINE_AA)

            color_legs = get_color_by_index(color=self.color, idx=3)
            left_knee = (keypoints[13][0], keypoints[13][1])
            right_knee = (keypoints[14][0], keypoints[14][1])
            left_foot = (keypoints[15][0], keypoints[15][1])
            right_foot = (keypoints[16][0], keypoints[16][1])
            # Draw points
            cv2.circle(scene, left_knee, 4, color_legs.as_bgr(), -1)
            cv2.circle(scene, right_knee, 4, color_legs.as_bgr(), -1)
            cv2.circle(scene, left_foot, 4, color_legs.as_bgr(), -1)
            cv2.circle(scene, right_foot, 4, color_legs.as_bgr(), -1)
            # Draw lines
            cv2.line(scene, left_hip, left_knee, color_legs.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, left_knee, left_foot, color_legs.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, right_hip, right_knee, color_legs.as_bgr(), 2, cv2.LINE_AA)
            cv2.line(scene, right_knee, right_foot, color_legs.as_bgr(), 2, cv2.LINE_AA)

        return scene