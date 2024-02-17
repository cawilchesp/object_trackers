from ultralytics import YOLO
import supervision as sv

import yaml
import cv2
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path

from tools.print_info import print_video_info, print_progress, step_message
from tools.write_csv import output_data_list, write_csv

# For debugging
from icecream import ic


def process_step() -> str:
    """ Simple step counter """
    global step_count
    step_count = str(int(step_count) + 1)
    return step_count


def main():
    # Initialize YOLOv8 Model
    model = YOLO(f"{MODEL_FOLDER}/{MODEL_WEIGHTS}.pt")
    step_message(process_step(), 'YOLOv8 Model Initialized')

    # Initialize Image capture
    step_message(process_step(), 'Initializing Image Source')
    source_name = Path(SOURCE).stem
    source_extension = Path(SOURCE).suffix
    image = cv2.imread(f"{FOLDER}/{source_name}{source_extension}")
    # print_image_info(SOURCE, image)
    target = f"{FOLDER}/{Path(SOURCE).stem}_output"

    label_annotator = sv.LabelAnnotator(text_scale=0.3, text_padding=2, text_position=sv.Position.TOP_LEFT)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    mask_annotator = sv.MaskAnnotator()
    
    # Start image processing
    step_message(process_step(), 'Image Processing Start')
    t_start = time.time()
    results_data = []
    annotated_image = image.copy()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2BGRA)
    annotated_image[:,:,0:4] = 0

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

    # Process YOLOv8 detections
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

    # Draw masks
    if DRAW_MASKS:
        annotated_image = mask_annotator.annotate(
            scene=annotated_image,
            detections=detections
        )

    # Save image
    if SAVE_IMAGE: cv2.imwrite(f"{target}_animal.png", annotated_image)
    
    # Save data in list
    results_data = output_data_list(results_data, None, detections, results.names)

    # Visualization
    if SHOW_RESULTS:
        cv2.imshow('Output', annotated_image)
        cv2.waitKey(0)

    # Saving data in CSV
    if SAVE_CSV:
        step_message(process_step(), 'Saving Results in CSV file')
        write_csv(f"{target}.csv", results_data)

    # Print total time elapsed
    step_message('Final', f"Total Time: {(time.time() - t_start):.2f} s")


if __name__ == "__main__":
    # Initialize Configuration File
    with open('image_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get configuration parameters
    MODEL_FOLDER = config['MODEL']['YOLOV8_FOLDER']
    MODEL_WEIGHTS = config['MODEL']['YOLOV8_WEIGHTS']
    FOLDER = config['INPUT']['FOLDER']
    SOURCE = config['INPUT']['SOURCE']
    CONFIDENCE = config['DETECTION']['CONFIDENCE']
    CLASS_FILTER = config['DETECTION']['CLASS_FILTER']
    IMAGE_SIZE = config['DETECTION']['IMAGE_SIZE']
    DRAW_BOXES = config['DRAW']['BOXES']
    DRAW_LABELS = config['DRAW']['LABELS']
    DRAW_MASKS = config['DRAW']['MASKS']
    SHOW_RESULTS = config['SHOW']
    SAVE_IMAGE = config['SAVE']['IMAGE']
    SAVE_CSV = config['SAVE']['CSV']

    step_count = '0'

    main()