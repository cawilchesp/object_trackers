from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import supervision as sv

import yaml
import cv2
from time import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from tools.print_info import print_video_info, print_progress, step_message
from tools.write_csv import output_data_list, write_csv
from tools.video_info import VideoInfo, from_video_path, from_camera

# For debugging
from icecream import ic


class ObjectDetection:
    def __init__(
            self,
            source,
            output,
            weights: str = "yolov8n",
            mode: str = 'predict'
        ):
        # Step count
        self.step_count = 0
        
        # input parameters
        self.source = source

        # model information
        self.weights = weights
        self.model = YOLO(f"weights/{self.weights}.pt")
        step_message(self.process_step(), f'{self.weights.upper()} Model Initialized')

        # visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # device information
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # inference mode
        self.mode = mode

        # output
        self.output = output

        # Annotators
        self.label_annotator = None
        self.bounding_box_annotator = None
        self.mask_annotator = None
        self.trace_annotator = None
        self.heatmap_annotator = None


    def predict(self, im0):
        results = self.model(
            source=im0,
            device=self.device,
            verbose=False )[0]
        return results
    
    def track(self, im0):
        results = self.model.track(
            source=im0,
            device=self.device,
            persist=True,
            verbose=False )[0]
        return results
    
    def display_fps(self, image):
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        cv2.rectangle(
            img=image,
            pt1=(0, 0),
            pt2=(text_size[0] + 10, text_size[1] + 10),
            color=(255, 255, 255),
            thickness=-1 )
        
        cv2.putText(
            img=image,
            text=text,
            org=(5, text_size[1] + 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 0),
            thickness=2 )

    def plot(self, image, results, detections):
        object_labels = [f"{results.names[class_id]} {tracker_id or ''} ({score:.2f})" for _, _, score, class_id, tracker_id, _ in detections]

        image = self.label_annotator.annotate(
            scene=image,
            detections=detections,
            labels=object_labels )

        image = self.bounding_box_annotator.annotate(
            scene=image,
            detections=detections )
        
        image = self.trace_annotator.annotate(
            scene=image,
            detections=detections )
        
        return image

    
    def process_step(self) -> str:
        """ Simple step counter """
        self.step_count = str(int(self.step_count) + 1)
        return self.step_count

    def __call__(self):
        cap = cv2.VideoCapture(self.source)
        assert cap.isOpened(), "Source cannot be opened"

        if self.source == 0 or self.source.lower().startswith('rtsp://'):
            source_info = from_camera(cap)
        else:
            source_info = from_video_path(cap)
        print_video_info(self.source, source_info)

        # Annotators
        self.line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(source_info.width,source_info.height)) * 0.5)
        self.text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(source_info.width,source_info.height)) * 0.5

        self.label_annotator = sv.LabelAnnotator(text_scale=self.text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=self.line_thickness)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=self.line_thickness)
        self.mask_annotator = sv.MaskAnnotator()
        self.trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=50, thickness=self.line_thickness)
        self.heatmap_annotator = sv.HeatMapAnnotator()

        frame_count = 0
        while cap.isOpened():
            self.start_time = time()

            ret, image = cap.read()
            assert ret, "File closed or reached final frame"

            results = self.predict(image) if self.mode == 'predict' else self.track(image)
            detections = sv.Detections.from_ultralytics(results)
            
            image = self.plot(image, results, detections)

            self.display_fps(image)
            
            frame_count += 1
            
            # Show results
            cv2.imshow('YOLOv8 Detection', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()