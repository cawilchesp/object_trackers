from ultralytics import YOLO
import supervision as sv

import yaml
import cv2
import torch
import time
from tqdm import tqdm
from pathlib import Path

from tools.print_info import print_video_info, step_message
from tools.annotators import box_annotations, mask_annotations
from tools.write_csv import csv_detections_list, write_csv

# For debugging
from icecream import ic


def main():
    # Initialize YOLOv8 Model
    model = YOLO(f"weights/{WEIGHTS}.pt")
    step_message('1', 'YOLOv8 Model Initialized')

    # Initialize video capture
    step_message('2', 'Initializing Video Source')
    frame_generator = sv.get_video_frames_generator(source_path=f"{FOLDER}/{SOURCE}")
    video_info = sv.VideoInfo.from_video_path(video_path=f"{FOLDER}/{SOURCE}")
    print_video_info(f"{FOLDER}/{SOURCE}", video_info)
    target = f"{FOLDER}/{Path(SOURCE).stem}"
    
    # Start video processing
    step_message('3', 'Video Processing Start')
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
                classes=CLASS_FILTER,
                retina_masks=True,
                verbose=False
            )[0]

            ic(results.names)
            quit()

            detections = sv.Detections.from_ultralytics(results)

            # Visualization
            labels = [f"{results.names[class_id]} - {score:.2f}" for _, _, score, class_id, _ in detections] if DRAW_LABELS else None

            # Draw boxes
            if DRAW_BOXES: annotated_image = box_annotations(annotated_image, detections, labels)

            # Draw masks
            if DRAW_MASKS and detections.mask is not None:
                annotated_image = mask_annotations(annotated_image, detections)

            # Save video
            if SAVE_VIDEO: sink.write_frame(frame=annotated_image)

            # Save data in list
            results_data = csv_detections_list(results_data, frame_number, detections, results.names)
            frame_number += 1

            # View live results
            if SHOW_RESULTS:
                cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Output', annotated_image)
                cv2.resizeWindow('Output', 1280, 720)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    # Saving data in CSV
    if SAVE_CSV:
        step_message('4', 'Saving Results in CSV file')
        write_csv(f"{target}.csv", results_data)

    # Print total time elapsed
    step_message('Final', f"Total Time: {(time.time() - t_start):.2f} s")


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
    SHOW_RESULTS = config['SHOW']
    CLIP_LENGTH = config['SAVE']['CLIP_LENGTH']
    SAVE_VIDEO = config['SAVE']['VIDEO']
    SAVE_CSV = config['SAVE']['CSV']

    with torch.no_grad():
        main()