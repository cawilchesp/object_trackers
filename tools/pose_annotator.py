import supervision as sv
from supervision.detection.core import Detections

import cv2
import numpy as np
from collections import deque

from icecream import ic

COLOR_LIST = sv.ColorPalette.from_hex([
    '#ff2d55',
    '#0f7f07',
    '#0095ff',
    '#ffcc00',
    '#46f0f0',
    '#ff9500',
    '#d2f53c',
    '#cf52de',
])

def pose_annotations(scene: np.array, poses: np.array, pose_config: dict) -> None:
    """
    Draw object pose on frame
    """
  
    for keypoints in poses:
        if pose_config['HEAD']:
            color = (0, 255, 0)
            nose = (keypoints[0][0], keypoints[0][1])
            left_eye = (keypoints[1][0], keypoints[1][1])
            right_eye = (keypoints[2][0], keypoints[2][1])
            left_ear = (keypoints[3][0], keypoints[3][1])
            right_ear = (keypoints[4][0], keypoints[4][1])
            # Draw points
            cv2.circle(scene, nose, 4, color, -1)
            cv2.circle(scene, left_eye, 4, color, -1)
            cv2.circle(scene, right_eye, 4, color, -1)
            cv2.circle(scene, left_ear, 4, color, -1)
            cv2.circle(scene, right_ear, 4, color, -1)
            # Draw lines
            cv2.line(scene, nose, left_eye, color, 2, cv2.LINE_AA)
            cv2.line(scene, nose, right_eye, color, 2, cv2.LINE_AA)
            cv2.line(scene, right_eye, left_eye, color, 2, cv2.LINE_AA)
            cv2.line(scene, left_ear, left_eye, color, 2, cv2.LINE_AA)
            cv2.line(scene, right_ear, right_eye, color, 2, cv2.LINE_AA)

        if pose_config['ARMS']:
            color = (255, 0, 0)
            left_shoulder =     (keypoints[5][0], keypoints[5][1])
            right_shoulder =    (keypoints[6][0], keypoints[6][1])
            left_elbow =        (keypoints[7][0], keypoints[7][1])
            right_elbow =       (keypoints[8][0], keypoints[8][1])
            left_hand =         (keypoints[9][0], keypoints[9][1])
            right_hand =        (keypoints[10][0], keypoints[10][1])

            # Draw points
            cv2.circle(scene, left_shoulder, 4, color, -1)
            cv2.circle(scene, right_shoulder, 4, color, -1)
            cv2.circle(scene, left_elbow, 4, color, -1)
            cv2.circle(scene, right_elbow, 4, color, -1)
            cv2.circle(scene, left_hand, 4, color, -1)
            cv2.circle(scene, right_hand, 4, color, -1)
            # Draw lines
            cv2.line(scene, left_hand, left_elbow, color, 2, cv2.LINE_AA)
            cv2.line(scene, left_elbow, left_shoulder, color, 2, cv2.LINE_AA)
            cv2.line(scene, left_shoulder, right_shoulder, color, 2, cv2.LINE_AA)
            cv2.line(scene, right_shoulder, right_elbow, color, 2, cv2.LINE_AA)
            cv2.line(scene, right_elbow, right_hand, color, 2, cv2.LINE_AA)

        if pose_config['TRUNK']:
            color = (0, 0, 255)
            left_hip = (keypoints[11][0], keypoints[11][1])
            right_hip = (keypoints[12][0], keypoints[12][1])
            # Draw points
            cv2.circle(scene, left_hip, 4, color, -1)
            cv2.circle(scene, right_hip, 4, color, -1)
            # Draw lines
            cv2.line(scene, left_shoulder, left_hip, color, 2, cv2.LINE_AA)
            cv2.line(scene, left_hip, right_hip, color, 2, cv2.LINE_AA)
            cv2.line(scene, right_hip, right_shoulder, color, 2, cv2.LINE_AA)

        if pose_config['LEGS']:
            color = (0, 255, 255)
            left_knee = (keypoints[13][0], keypoints[13][1])
            right_knee = (keypoints[14][0], keypoints[14][1])
            left_foot = (keypoints[15][0], keypoints[15][1])
            right_foot = (keypoints[16][0], keypoints[16][1])
            # Draw points
            cv2.circle(scene, left_knee, 4, color, -1)
            cv2.circle(scene, right_knee, 4, color, -1)
            cv2.circle(scene, left_foot, 4, color, -1)
            cv2.circle(scene, right_foot, 4, color, -1)
            # Draw lines
            cv2.line(scene, left_hip, left_knee, color, 2, cv2.LINE_AA)
            cv2.line(scene, left_knee, left_foot, color, 2, cv2.LINE_AA)
            cv2.line(scene, right_hip, right_knee, color, 2, cv2.LINE_AA)
            cv2.line(scene, right_knee, right_foot, color, 2, cv2.LINE_AA)

    return scene