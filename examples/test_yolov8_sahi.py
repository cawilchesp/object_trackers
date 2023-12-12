import supervision as sv
import numpy as np
from ultralytics import YOLO

import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
import time
from pathlib import Path
from collections import deque

from tools.print_info import print_video_info, print_progress
from tools.annotators import box_annotations, track_annotations
from tools.write_csv import csv_list, write_csv

# For debugging
from icecream import ic


# class names
class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush' ]



# Obtain parent folder to start app from any folder
source_path = Path(__file__).resolve()
source_dir = source_path.parent

weights = f'{source_dir}/weights/yolov8/yolov8x.pt'
# source = "C:/Users/cwilches/OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P/Dev/ruta_costera/C3_detenido.mp4"
source = "C:/Users/cwilches/OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P/Videos/Proyectos/Drones_Secretaria_Movilidad/Secr_Movilidad_01-12-2022-Carrera_45_calle_88_20221201-141306-542075-0.mp4"
class_filter = [0,1,2,3,5,7]


# Initialize YOLOv8 Model
model = YOLO(weights)
print('\n\033[32m[1]\033[0m YOLOv8 Model Initialized\n')

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(
        source=image_slice,
        conf=0.25,
        device=0,
        agnostic_nms=True,
        classes=class_filter,
        # verbose=False
    )[0]
    return sv.Detections.from_ultralytics(result)

# Initialize video capture
print('\033[32m[4]\033[0m Initializing Video Source')
source_type = 'camera' if source.lower().startswith('rtsp://') else 'video'
video_info = sv.VideoInfo.from_video_path(video_path=source)
print_video_info(source, source_type, video_info)
cap = cv2.VideoCapture(f"{source}")


# Image capture
cap.set(cv2.CAP_PROP_POS_FRAMES, 1320)
success, image = cap.read()

annotated_image = image.copy()

slicer = sv.InferenceSlicer(callback = callback)

sliced_detections = slicer(image)

# labels = [f"{slicer.names[class_id]} - {tracker_id}" for _, _, _, class_id, tracker_id in sliced_detections]

box_annotator = sv.BoxAnnotator()

sliced_image = box_annotator.annotate(annotated_image, detections=sliced_detections)

sv.plot_image(sliced_image)
