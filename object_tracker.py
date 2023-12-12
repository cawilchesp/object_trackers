from ultralytics import YOLO
import supervision as sv

import yaml
import cv2
import torch
import time
from tqdm import tqdm
from pathlib import Path
from collections import deque

from tools.print_info import print_video_info, step_message
from tools.write_csv import csv_tracks_list, write_csv

# For debugging
from icecream import ic

def main():
    # Initialize YOLOv8 Model
    model = YOLO(f"weights/{WEIGHTS}.pt")
    step_message('1', 'YOLOv8 Model Initialized')

    # Initialize Byte Tracker
    byte_tracker = sv.ByteTrack()

    # Initialize video capture
    video_info = sv.VideoInfo.from_video_path(video_path=f"{FOLDER}/{SOURCE}")
    frame_generator = sv.get_video_frames_generator(source_path=f"{FOLDER}/{SOURCE}")
    target = f"{FOLDER}/{Path(SOURCE).stem}"

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_scale=0.3, text_padding=2, text_position=sv.Position.TOP_LEFT)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.BOTTOM_CENTER, trace_length=128, thickness=1)
    heat_map_annotator = sv.HeatMapAnnotator()

    results_data = []
    with sv.VideoSink(target_path=f"{target}.avi", video_info=video_info, codec="H264") as sink:
        for frame_number, image in enumerate(tqdm(frame_generator, total=video_info.total_frames, unit='frames')):
            annotated_image = image.copy()

            # Process YOLOv8 detections
            results = model(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=0,
                agnostic_nms=True,
                classes=CLASS_FILTER,
                retina_masks=True,
                verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)

            # Update tracks
            tracks = byte_tracker.update_with_detections(detections)

            # Draw labels
            object_labels = [f"{results.names[class_id]} - {tracker_id}" for _, _, _, class_id, tracker_id in tracks] if DRAW_LABELS else None
            annotated_image = label_annotator.annotate(
                scene=annotated_image,
                detections=tracks,
                labels=object_labels
            )

            # Draw boxes
            annotated_image = bounding_box_annotator.annotate(
                scene=annotated_image,
                detections=tracks
            )

            # Draw tracks
            annotated_image = trace_annotator.annotate(
                scene=annotated_image,
                detections=tracks
            )

            # Draw heat map
            annotated_image = heat_map_annotator.annotate(
                scene=annotated_image,
                detections=tracks
            )

            sink.write_frame(frame=annotated_image)

            # Save data in list
            results_data = csv_tracks_list(results_data, frame_number, tracks, results.names)
            
            # View live results
            if SHOW_RESULTS:
                cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Output', annotated_image)
                cv2.resizeWindow('Output', 1280, 720)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    # Saving data in CSV
    if SAVE_CSV:
        write_csv(f"{target}.csv", results_data)



if __name__ == "__main__":
    # Initialize Configuration File
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get configuration parameters
    FOLDER = config['INPUT']['FOLDER']
    SOURCE = config['INPUT']['SOURCE']
    WEIGHTS = config['YOLO']['YOLO_WEIGHTS']
    CONFIDENCE = config['DETECTION']['CONFIDENCE']
    CLASS_FILTER = config['DETECTION']['CLASS_FILTER']
    IMAGE_SIZE = config['DETECTION']['IMAGE_SIZE']
    DRAW_BOXES = config['DRAW']['BOXES']
    DRAW_LABELS = config['DRAW']['LABELS']
    DRAW_MASKS = config['DRAW']['MASKS']
    DRAW_TRACKS = config['DRAW']['TRACKS']
    TRACK_LENGTH = config['DRAW']['TRACK_LENGTH']
    SHOW_RESULTS = config['SHOW']
    SAVE_VIDEO = config['SAVE']['VIDEO']
    SAVE_CSV = config['SAVE']['CSV']

    main()
