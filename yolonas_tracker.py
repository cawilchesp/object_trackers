from super_gradients.training import models
import supervision as sv

import yaml
import cv2
import time
from tqdm import tqdm
from pathlib import Path

from tools.print_info import print_video_info, step_message
from tools.write_csv import csv_tracks_list, csv_detections_list, write_csv

# For debugging
from icecream import ic


def main():
    # Initialize YOLO-NAS Model
    model = models.get(model_name=MODEL_WEIGHTS, pretrained_weights='coco').cuda()
    step_count = 1
    step_message(str(step_count), 'YOLO-NAS Model Initialized')

    # Initialize Byte Tracker
    if TRACKING:
        byte_tracker = sv.ByteTrack()
        step_count = 1
        step_message(str(step_count), 'ByteTrack Tracker Initialized')

    # Initialize video capture
    step_count = 1
    step_message(str(step_count), 'Initializing Video Source')
    video_info = sv.VideoInfo.from_video_path(video_path=f"{FOLDER}/{SOURCE}")
    frame_generator = sv.get_video_frames_generator(source_path=f"{FOLDER}/{SOURCE}")
    # print_video_info(f"{FOLDER}/{SOURCE}", video_info)
    
    if TRACKING:
        target = f"{FOLDER}/{Path(SOURCE).stem}_tracking"
    else:
        target = f"{FOLDER}/{Path(SOURCE).stem}_detection"

    label_annotator = sv.LabelAnnotator(text_scale=0.3, text_padding=2, text_position=sv.Position.TOP_LEFT)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.BOTTOM_CENTER, trace_length=TRACK_LENGTH, thickness=1)
    heatmap_annotator = sv.HeatMapAnnotator()

    # Start video processing
    step_count = 1
    step_message(str(step_count), 'Video Processing Started')
    t_start = time.time()
    results_data = []
    frame_number = 0
    with sv.VideoSink(target_path=f"{target}.avi", video_info=video_info, codec="H264") as sink:
        for image in tqdm(frame_generator, total=video_info.total_frames, unit='frames'):
            annotated_image = image.copy()

            # Process YOLO-NAS detections
            results = list(model.predict(image, conf=CONFIDENCE, fuse_model=False))[0]
            detections = sv.Detections.from_yolo_nas(results)

            # Update tracks
            if TRACKING:
                tracks = byte_tracker.update_with_detections(detections)

            # Draw labels
            if DRAW_LABELS:
                if TRACKING:
                    object_labels = [f"{results.class_names[class_id]} - {tracker_id}" for _, _, _, class_id, tracker_id in tracks]
                else:
                    object_labels = [f"{results.class_names[class_id]} - {score:.2f}" for _, _, score, class_id, _ in detections]
                    
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=tracks if TRACKING else detections,
                    labels=object_labels
                )

            # Draw boxes
            if DRAW_BOXES:
                annotated_image = bounding_box_annotator.annotate(
                    scene=annotated_image,
                    detections=tracks if TRACKING else detections
                )

            # Draw tracks
            if DRAW_TRACKS and TRACKING:
                annotated_image = trace_annotator.annotate(
                    scene=annotated_image,
                    detections=tracks
                )

            # Draw heatmap
            if DRAW_HEATMAP:
                annotated_image = heatmap_annotator.annotate(
                    scene=annotated_image,
                    detections=tracks if TRACKING else detections
                )

            # Save video
            if SAVE_VIDEO: sink.write_frame(frame=annotated_image)

            # Save data in list
            if TRACKING:
                results_data = csv_tracks_list(results_data, frame_number, tracks, results.class_names)
            else:
                results_data = csv_detections_list(results_data, frame_number, detections, results.class_names)

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
        step_count = 1
        step_message(str(step_count), 'Saving Results in CSV file')
        write_csv(f"{target}.csv", results_data)

    # Print total time elapsed
    step_message('Final', f"Total Time: {(time.time() - t_start):.2f} s")


if __name__ == "__main__":
    # Initialize Configuration File
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get configuration parameters
    MODEL_WEIGHTS = config['MODEL']['NAS_WEIGHTS']
    FOLDER = config['INPUT']['FOLDER']
    SOURCE = config['INPUT']['SOURCE']
    CONFIDENCE = config['DETECTION']['CONFIDENCE']
    CLASS_FILTER = config['DETECTION']['CLASS_FILTER']
    IMAGE_SIZE = config['DETECTION']['IMAGE_SIZE']
    TRACKING = config['TRACKING']
    DRAW_BOXES = config['DRAW']['BOXES']
    DRAW_LABELS = config['DRAW']['LABELS']
    DRAW_MASKS = config['DRAW']['MASKS']
    DRAW_TRACKS = config['DRAW']['TRACKS']
    DRAW_HEATMAP = config['DRAW']['HEATMAP']
    TRACK_LENGTH = config['DRAW']['TRACK_LENGTH']
    SHOW_RESULTS = config['SHOW']
    SAVE_VIDEO = config['SAVE']['VIDEO']
    SAVE_CSV = config['SAVE']['CSV']

    main()
        