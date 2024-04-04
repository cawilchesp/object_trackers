import json
from typing import Generator, List

import cv2
import numpy as np


def get_stream_frames_generator(rtsp_url: str) -> Generator[np.ndarray, None, None]:
    """
    Generator function to yield frames from an RTSP stream.

    Args:
        rtsp_url (str): URL of the RTSP video stream.

    Yields:
        np.ndarray: The next frame from the video stream.
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise Exception("Error: Could not open video stream.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or error reading frame.")
                break
            yield frame
    finally:
        cap.release()