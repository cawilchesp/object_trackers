from ultralytics import YOLO
import supervision as sv

import torch
import numpy as np
from typing import Tuple


class KeyPointSink:
    def __init__(
        self,
        weights_path: str,
        image_size: int = 640,
        confidence: float = 0.5
    ) -> None:
        self.model = YOLO(weights_path)
        self.image_size = image_size        
        self.confidence = confidence

    def detect(self, image: np.array) -> Tuple[sv.Detections, sv.KeyPoints]:
        results = self.model(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
        )[0]
        key_points = sv.KeyPoints.from_ultralytics(results)
        detections = sv.Detections.from_ultralytics(results)
        
        return (detections, key_points)