from ultralytics import YOLO
import supervision as sv

import yaml
import cv2
import torch
import time
from tqdm import tqdm
from pathlib import Path

from tools.print_info import print_video_info, step_message
from tools.write_csv import csv_detections_list, write_csv
from tools.pose_annotator import PoseAnnotator

# For debugging
from icecream import ic


def main():
    # Initialize YOLOv8 Model
    model = YOLO(f"{MODEL_FOLDER}/{MODEL_WEIGHTS}.pt")
    step_count = 1
    step_message(str(step_count), 'YOLOv8 Pose Model Initialized')

    # Initialize video capture
    step_count += 1
    step_message(str(step_count), 'Initializing Video Source')
    video_info = sv.VideoInfo.from_video_path(video_path=f"{FOLDER}/{SOURCE}")
    frame_generator = sv.get_video_frames_generator(source_path=f"{FOLDER}/{SOURCE}")
    # print_video_info(f"{FOLDER}/{SOURCE}", video_info)
    
    target = f"{FOLDER}/{Path(SOURCE).stem}_pose"
    
    label_annotator = sv.LabelAnnotator(text_scale=0.3, text_padding=2, text_position=sv.Position.TOP_LEFT)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    pose_annotator = PoseAnnotator(thickness=4, radius=8)

    # Start video processing
    step_count += 1
    step_message(str(step_count), 'Video Processing Start')
    t_start = time.time()
    results_data = []
    frame_number = 0
    with sv.VideoSink(target_path=f"{target}.avi", video_info=video_info, codec="H264") as sink:
        for image in tqdm(frame_generator, total=video_info.total_frames, unit='frames'):
            annotated_image = image.copy()

            # Run YOLOv8 inference
            results = model(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=0,
                agnostic_nms=True,
                retina_masks=True,
                verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)

            # Draw labels
            if DRAW_LABELS:
                object_labels = [f"{results.names[class_id]} - {score:.2f}" for _, _, score, class_id, _ in detections]
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections,
                    labels=object_labels
                )

            # Draw boxes
            if DRAW_BOXES:
                annotated_image = bounding_box_annotator.annotate(
                    scene=annotated_image,
                    detections=detections
                )

            # Draw poses
            annotated_image = pose_annotator.annotate(
                annotated_image,
                results
            )
            
            # Save video
            if SAVE_VIDEO: sink.write_frame(frame=annotated_image)

            # Save data in list
            results_data = csv_detections_list(results_data, frame_number, detections, results.names)
            frame_number += 1
    
            # Visualization
            if SHOW_RESULTS:
                cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Output', annotated_image)
                cv2.resizeWindow('Output', 1280, 720)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    # Saving data in CSV
    if SAVE_CSV:
        step_count += 1
        step_message(str(step_count), 'Saving Results in CSV file')
        write_csv(f"{target}.csv", results_data)

    # Print total time elapsed
    step_message('Final', f"Total Time: {(time.time() - t_start):.2f} s")


if __name__ == "__main__":
    # Initialize Configuration File
    with open('pose_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get configuration parameters
    MODEL_FOLDER = config['MODEL']['YOLOV8_FOLDER']
    MODEL_WEIGHTS = config['MODEL']['YOLOV8_WEIGHTS']
    FOLDER = config['INPUT']['FOLDER']
    SOURCE = config['INPUT']['SOURCE']
    CONFIDENCE = config['DETECTION']['CONFIDENCE']
    IMAGE_SIZE = config['DETECTION']['IMAGE_SIZE']
    DRAW_BOXES = config['DRAW']['BOXES']
    DRAW_LABELS = config['DRAW']['LABELS']
    SHOW_RESULTS = config['SHOW']
    SAVE_VIDEO = config['SAVE']['VIDEO']
    SAVE_CSV = config['SAVE']['CSV']

    with torch.no_grad():
        main()